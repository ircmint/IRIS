"""
evaluation.py
=============
Standalone evaluation for the fine-tuned LLaVA-NeXT-Video QLoRA adapter.

This does NOT fine-tune anything. It loads:
  - the frozen base model (quantized 4-bit, same as training)
  - the already-trained LoRA adapter (BEST_ADAPTER_DIR from FineTune.py,
    or any adapter dir you point it at)
and runs generation + metrics ONLY, on the held-out validation split.

Metric set matches the reference report format:
  n_events_evaluated, n_generation_errors,
  accuracy,
  precision_macro / recall_macro / f1_macro,
  precision_weighted / recall_weighted / f1_weighted,
  bleu4_mean, meteor_mean, rougeL_mean, cider,
  json_parse_failure_rate, hallucination_rate

Run inside your llava_next_video env:

    conda activate $SCRATCH_ROOT/envs/llava_next_video
    python evaluation.py --adapter_path /path/to/adapter

Or just edit ADAPTER_PATH below and run with no args (hardcoded CONFIG
still drives everything -- argparse only lets you override the adapter
checkpoint for comparing runs, same as your other scripts).
"""

import os
import re
import json
import math
import argparse
import collections

import numpy as np
import pandas as pd
import torch
from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor, BitsAndBytesConfig
from peft import PeftModel

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from FineTune import (
    MODEL_ID, MODEL_CACHE_DIR, CSV_PATH, VIDEO_DIR, OUTPUT_DIR,
    BEST_ADAPTER_DIR, LABELS, SYSTEM_PROMPT, VAL_FRACTION, SEED,
    FRAMES_PER_EVENT, RESIZE_MAX_SIDE, MIN_EVENT_DURATION,
    classify_from_notes, build_telemetry_summary, sample_frames, clean_events_df,
)

try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Warning: Failed to load NLTK data: {e}")

# ----------------------------------------------------------------------
# Config -- edit these to match your setup, or override via CLI args
# ----------------------------------------------------------------------
RESULTS_DIR = f"{OUTPUT_DIR}/eval_results"
PREDICTIONS_CSV = f"{RESULTS_DIR}/predictions.csv"
OUTPUT_REPORT = f"{RESULTS_DIR}/evaluation_report.json"
MAX_NEW_TOKENS = 128

# Terms that should only appear in reasoning if the model actually saw them.
# Fill in from real predictions.csv output once you have a run -- see note
# at the bottom of this script. These are placeholders, same caveat as before.
HALLUCINATION_TERMS = [
    "pedestrian crossing sign", "school zone", "traffic light", "stop sign",
    "speed limit sign", "double yellow line",
]

os.makedirs(RESULTS_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# Model loading -- base model (QLoRA) + adapter ONLY, no training
# ----------------------------------------------------------------------
def load_model_and_processor(model_id, adapter_path):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    base = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id, cache_dir=MODEL_CACHE_DIR, quantization_config=quantization_config,
        torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    processor = LlavaNextVideoProcessor.from_pretrained(adapter_path)
    return model, processor


# ----------------------------------------------------------------------
# JSON parsing of model output
# ----------------------------------------------------------------------
def parse_prediction(text):
    """Returns (label, confidence, reasoning, parse_ok)."""
    cleaned = text.strip()
    try:
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        obj = json.loads(match.group(0) if match else cleaned)
        label = str(obj.get("label", "other")).strip().lower()
        confidence = float(obj.get("confidence", 0.0))
        reasoning = str(obj.get("reasoning", ""))
        if label not in LABELS:
            label = "other"
        return label, confidence, reasoning, True
    except Exception:
        return "other", 0.0, text, False


# ----------------------------------------------------------------------
# CIDEr (self-contained, standard TF-IDF n-gram implementation --
# no pycocoevalcap/Java dependency needed)
# ----------------------------------------------------------------------
def _ngrams(words, n_max=4):
    counts = collections.defaultdict(int)
    for n in range(1, n_max + 1):
        for i in range(len(words) - n + 1):
            counts[tuple(words[i:i + n])] += 1
    return counts


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class CiderScorer:
    """Standard CIDEr (Vedantam et al. 2015), single-reference-per-sample case."""

    def __init__(self, n_max=4, sigma=6.0):
        self.n_max = n_max
        self.sigma = sigma

    def compute(self, candidates, references):
        cand_ngrams = [_ngrams(_tokenize(c), self.n_max) for c in candidates]
        ref_ngrams = [_ngrams(_tokenize(r), self.n_max) for r in references]

        doc_freq = collections.defaultdict(int)
        for rg in ref_ngrams:
            for ngram in rg.keys():
                doc_freq[ngram] += 1
        n_docs = len(ref_ngrams)

        def counts_to_tfidf_vec(counts, total_len):
            vec = {}
            for ngram, count in counts.items():
                n = len(ngram)
                tf = count / max(1, total_len)
                idf = math.log(max(1.0, n_docs / max(1.0, doc_freq.get(ngram, 0) + 1)))
                vec[ngram] = tf * idf
            return vec

        scores = []
        for cand_c, ref_c in zip(cand_ngrams, ref_ngrams):
            cand_len = sum(cand_c.values())
            ref_len = sum(ref_c.values())
            cand_vec = counts_to_tfidf_vec(cand_c, cand_len)
            ref_vec = counts_to_tfidf_vec(ref_c, ref_len)

            per_n_scores = []
            for n in range(1, self.n_max + 1):
                c_n = {k: v for k, v in cand_vec.items() if len(k) == n}
                r_n = {k: v for k, v in ref_vec.items() if len(k) == n}
                common = set(c_n.keys()) & set(r_n.keys())
                numerator = sum(c_n[k] * r_n[k] for k in common)
                c_norm = math.sqrt(sum(v * v for v in c_n.values()))
                r_norm = math.sqrt(sum(v * v for v in r_n.values()))
                denom = c_norm * r_norm
                sim = numerator / denom if denom > 0 else 0.0
                per_n_scores.append(sim)
            score = (sum(per_n_scores) / self.n_max) * 10.0
            scores.append(score)
        return float(np.mean(scores)) if scores else 0.0


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def compute_all_metrics(pred_labels, true_labels, pred_reasons, true_reasons, parse_ok_flags):
    accuracy = accuracy_score(true_labels, pred_labels)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=LABELS, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=LABELS, average="weighted", zero_division=0
    )

    smooth = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    bleus, meteors, rouges = [], [], []
    for pred_r, true_r in zip(pred_reasons, true_reasons):
        pred_tok = nltk.word_tokenize(pred_r.lower()) if pred_r else []
        ref_tok = nltk.word_tokenize(true_r.lower()) if true_r else []
        if ref_tok and pred_tok:
            bleus.append(sentence_bleu([ref_tok], pred_tok, smoothing_function=smooth))
            try:
                meteors.append(meteor_score([ref_tok], pred_tok))
            except Exception:
                meteors.append(0.0)
        rouges.append(scorer.score(true_r or "", pred_r or "")['rougeL'].fmeasure)

    cider_score = CiderScorer().compute(pred_reasons, true_reasons)

    hall_rates = []
    for pred_r, true_r in zip(pred_reasons, true_reasons):
        p_lower, r_lower = (pred_r or "").lower(), (true_r or "").lower()
        hall, tot = 0, 0
        for elem in HALLUCINATION_TERMS:
            if elem in p_lower:
                tot += 1
                if elem not in r_lower:
                    hall += 1
        hall_rates.append(hall / tot if tot > 0 else 0.0)

    json_parse_failure_rate = 1.0 - (sum(parse_ok_flags) / len(parse_ok_flags)) if parse_ok_flags else 0.0

    return {
        "accuracy": round(float(accuracy), 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(precision_weighted), 4),
        "recall_weighted": round(float(recall_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "bleu4_mean": round(float(np.mean(bleus)) if bleus else 0.0, 4),
        "meteor_mean": round(float(np.mean(meteors)) if meteors else 0.0, 4),
        "rougeL_mean": round(float(np.mean(rouges)) if rouges else 0.0, 4),
        "cider": round(float(cider_score), 4),
        "json_parse_failure_rate": round(float(json_parse_failure_rate), 4),
        "hallucination_rate": round(float(np.mean(hall_rates)) if hall_rates else 0.0, 4),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained LLaVA-NeXT-Video LoRA adapter -- no training.")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--adapter_path", default=BEST_ADAPTER_DIR)
    parser.add_argument("--csv_path", default=CSV_PATH)
    parser.add_argument("--video_dir", default=VIDEO_DIR)
    parser.add_argument("--output_report", default=OUTPUT_REPORT)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading base model from: {args.model_id}")
    print(f"Loading adapter from:    {args.adapter_path}")
    model, processor = load_model_and_processor(args.model_id, args.adapter_path)
    model_device = next(model.parameters()).device

    df = pd.read_csv(args.csv_path)
    df = clean_events_df(df)
    df["label"] = df["notes"].apply(classify_from_notes)
    val_parts = [g.sample(frac=VAL_FRACTION, random_state=SEED) for _, g in df.groupby("label")]
    val_df = pd.concat(val_parts).reset_index(drop=True)
    print(f"Loaded {len(val_df)} held-out events from {args.csv_path}")

    pred_labels, true_labels = [], []
    pred_reasons, true_reasons = [], []
    parse_ok_flags = []
    rows_out = []
    n_generation_errors = 0

    with torch.no_grad():
        for i, row in val_df.iterrows():
            true_label = row["label"]
            true_reason = row.get("notes", "") if isinstance(row.get("notes", ""), str) else ""

            try:
                start, end = row["adjusted_start"], row["adjusted_end"]
                if end - start < MIN_EVENT_DURATION:
                    pad = (MIN_EVENT_DURATION - (end - start)) / 2
                    start, end = max(0, start - pad), end + pad

                video_path = os.path.join(args.video_dir, f"{row['clip_id']}.mp4")
                frames = sample_frames(video_path, start, end, FRAMES_PER_EVENT, RESIZE_MAX_SIDE)
                telemetry = build_telemetry_summary(row)

                conversation = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": telemetry}]},
                ]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(text=prompt, videos=[frames], return_tensors="pt").to(model_device)

                out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
                response = processor.tokenizer.decode(
                    out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
            except Exception as gen_err:
                print(f"  Warning: generation failed for clip {row.get('clip_id', i)} ({gen_err}); recording empty response.")
                response = ""
                n_generation_errors += 1

            pred_label, confidence, pred_reason, parse_ok = parse_prediction(response)

            pred_labels.append(pred_label)
            true_labels.append(true_label)
            pred_reasons.append(pred_reason)
            true_reasons.append(true_reason)
            parse_ok_flags.append(parse_ok)

            rows_out.append({
                "clip_id": row["clip_id"], "gold_label": true_label, "pred_label": pred_label,
                "confidence": confidence, "pred_reasoning": pred_reason, "raw_output": response,
            })

            if (i + 1) % 10 == 0 or (i + 1) == len(val_df):
                print(f"Generated {i + 1}/{len(val_df)} outputs...")

    pd.DataFrame(rows_out).to_csv(PREDICTIONS_CSV, index=False)

    if n_generation_errors == len(val_df):
        print("*** ALL generations failed -- aborting, metrics would be meaningless. ***")
        return

    metrics = compute_all_metrics(pred_labels, true_labels, pred_reasons, true_reasons, parse_ok_flags)

    report = {
        "n_events_evaluated": len(val_df),
        "n_generation_errors": n_generation_errors,
        **metrics,
    }

    print("\n" + "=" * 60 + "\nEVALUATION REPORT\n" + "=" * 60)
    for k, v in report.items():
        print(f"{k}: {v}")
    print("=" * 60)

    with open(args.output_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {args.output_report}")
    print(f"Predictions saved to: {PREDICTIONS_CSV}")
    print("NOTE: hallucination_rate only flags the placeholder terms in HALLUCINATION_TERMS.")
    print("Skim predictions.csv's pred_reasoning column and add real invented-detail phrases you see.")


if __name__ == "__main__":
    main()