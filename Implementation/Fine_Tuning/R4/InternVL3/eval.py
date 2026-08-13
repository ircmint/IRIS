"""
Evaluate the QLoRA-tuned InternVL3-8B-hf on the held-out val split.
Taxonomy is loaded from taxonomy.json (saved next to the adapter by
train.py) rather than hardcoded, since it's derived from `notes` and may
change if the keyword rules in labels.py are tuned.

Metrics: accuracy, precision/recall/F1 (macro), BLEU-4, METEOR, ROUGE-L,
CIDEr, json_parse_failure_rate, hallucination_rate.

Run:
    pip install scikit-learn nltk rouge-score pycocoevalcap
    python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
    python eval.py
"""

# ============================================================
# PATHS — edit or set environment variables before running
#   IRASTE_SCRATCH : root of your scratch/working directory
#   IRASTE_HOME    : your home directory on the compute node
#   HF_HOME        : HuggingFace cache directory
# ============================================================
import os as _os
SCRATCH_ROOT = _os.environ.get("IRASTE_SCRATCH", "/scratch/<your_username>")
HOME_ROOT    = _os.environ.get("IRASTE_HOME",    "/home/<your_username>")
HF_CACHE     = _os.environ.get("HF_HOME",        _os.path.join(SCRATCH_ROOT, "hf_cache"))

import json
import os
import re

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

from labels import build_taxonomy, load_taxonomy, build_target_json, build_telemetry_summary
from dataset import EvasiveEventDataset

MODEL_PATH = os.path.join(SCRATCH_ROOT, "models/internvl3-8b-hf")
ADAPTER_PATH = os.path.join(SCRATCH_ROOT, "runs/internvl3-8b-qlora-evasive/best_adapter")
VIDEO_DIR = os.path.join(HOME_ROOT, "IRASTE")
VIDEO_FILENAME = "GX019940.MP4"
CSV_PATH = os.path.join(HOME_ROOT, "IRASTE/DAY_2/candidate_events_updated.csv")
NUM_FRAMES = 2  # must match train.py's setting - the adapter was trained on this frame count
MIN_CLIP_DURATION = 1.0
VAL_SPLIT = 0.15
SEED = 42
NUM_TOLERANCE = 0.15


def load_model():
    processor = AutoProcessor.from_pretrained(ADAPTER_PATH)

    # MUST match train.py's quantization - InternVL3-8B in plain fp16 is
    # ~15-16GB of weights alone, which cannot fit this 10.9GB GPU regardless
    # of anything else running. This was the actual cause of the OOM at
    # model-load time (before any inference even started).
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head"],
    )

    base = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, quantization_config=bnb_config, dtype=torch.float16, device_map={"": 0}
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    return model, processor


def generate(model, processor, frames, prompt):
    inputs = processor(images=[frames], text=[prompt], return_tensors="pt", crop_to_patches=False).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    completion_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return processor.decode(completion_ids, skip_special_tokens=True)


def extract_numbers(text):
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", text)]


def is_hallucinated(pred_json, input_numbers, valid_labels):
    label = pred_json.get("label")
    confidence = pred_json.get("confidence")
    reasoning = pred_json.get("reasoning", "")

    if label not in valid_labels:
        return True
    if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
        return True
    for n in extract_numbers(reasoning):
        grounded = any(abs(n - m) <= NUM_TOLERANCE * max(abs(m), 1e-6) for m in input_numbers)
        if not grounded:
            return True
    return False


def main():
    model, processor = load_model()
    taxonomy = load_taxonomy(os.path.join(ADAPTER_PATH, "taxonomy.json"))
    print("Loaded taxonomy:", taxonomy)

    df = pd.read_csv(CSV_PATH)
    df, taxonomy_check = build_taxonomy(df)
    if sorted(taxonomy_check) != sorted(taxonomy):
        print("WARNING: current data's derived taxonomy differs from the saved "
              "adapter's taxonomy - labels.py may have changed since training. "
              f"saved={taxonomy} current={taxonomy_check}")

    try:
        _, val_df = train_test_split(df, test_size=VAL_SPLIT, random_state=SEED, stratify=df["_label"])
    except ValueError:
        _, val_df = train_test_split(df, test_size=VAL_SPLIT, random_state=SEED)
    val_df = val_df.reset_index(drop=True)

    video_path = os.path.join(VIDEO_DIR, VIDEO_FILENAME)
    dataset = EvasiveEventDataset(
        val_df, video_path, processor, taxonomy,
        cache_dir="/tmp/internvl_eval_cache", split_name="eval_val",
        num_frames=NUM_FRAMES, min_clip_duration=MIN_CLIP_DURATION,
    )

    gold_labels, pred_labels = [], []
    gold_reasoning, pred_reasoning = [], []
    parse_failures = 0
    hallucinations = 0
    generation_errors = 0
    cider_gts, cider_res = {}, {}
    per_event_rows = []

    smoothing = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    import time
    print(f"Starting eval over {len(dataset)} val examples...", flush=True)

    for idx in range(len(dataset)):
        t0 = time.time()
        row = dataset.df.iloc[idx]
        from labels import get_clip_bounds
        start, end = get_clip_bounds(row, MIN_CLIP_DURATION)
        target = json.loads(build_target_json(row))
        input_numbers = extract_numbers(build_telemetry_summary(row))

        try:
            frames = dataset._sample_frames(start, end)
            prompt = dataset._build_prompt(row)
            raw_output = generate(model, processor, frames, prompt)
        except Exception as e:
            generation_errors += 1
            print(f"[{idx + 1}/{len(dataset)}] GENERATION ERROR: {e}", flush=True)
            gold_labels.append(target["label"])
            gold_reasoning.append(target["reasoning"])
            pred_labels.append("GEN_ERROR")
            pred_reasoning.append("")
            hallucinations += 1
            per_event_rows.append({
                "idx": idx, "clip_id": row.get("clip_id", ""), "gold_label": target["label"],
                "pred_label": "GEN_ERROR", "gold_reasoning": target["reasoning"],
                "pred_reasoning": "", "parse_failed": False, "generation_error": True,
                "hallucinated": True,
            })
            continue

        print(f"[{idx + 1}/{len(dataset)}] {time.time() - t0:.1f}s  "
              f"gold={target['label']}  raw_output={raw_output[:80]!r}", flush=True)

        gold_labels.append(target["label"])
        gold_reasoning.append(target["reasoning"])

        try:
            match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            pred = json.loads(match.group(0)) if match else json.loads(raw_output)
        except (json.JSONDecodeError, AttributeError):
            parse_failures += 1
            pred_labels.append("PARSE_FAIL")
            pred_reasoning.append("")
            hallucinations += 1
            per_event_rows.append({
                "idx": idx, "clip_id": row.get("clip_id", ""), "gold_label": target["label"],
                "pred_label": "PARSE_FAIL", "gold_reasoning": target["reasoning"],
                "pred_reasoning": raw_output, "parse_failed": True, "generation_error": False,
                "hallucinated": True,
            })
            continue

        pred_label = pred.get("label", "PARSE_FAIL")
        pred_reason = pred.get("reasoning", "")
        pred_labels.append(pred_label)
        pred_reasoning.append(pred_reason)

        halluc = is_hallucinated(pred, input_numbers, taxonomy)
        if halluc:
            hallucinations += 1

        per_event_rows.append({
            "idx": idx, "clip_id": row.get("clip_id", ""), "gold_label": target["label"],
            "pred_label": pred_label, "gold_reasoning": target["reasoning"],
            "pred_reasoning": pred_reason, "parse_failed": False, "generation_error": False,
            "hallucinated": halluc,
        })

        key = str(idx)
        cider_gts[key] = [target["reasoning"]]
        cider_res[key] = [pred_reason]

    n = len(dataset)

    accuracy = accuracy_score(gold_labels, pred_labels)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        gold_labels, pred_labels, labels=taxonomy, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        gold_labels, pred_labels, labels=taxonomy, average="weighted", zero_division=0
    )

    bleu_scores, meteor_scores, rouge_scores = [], [], []
    for gold, pred in zip(gold_reasoning, pred_reasoning):
        if not pred:
            bleu_scores.append(0.0)
            meteor_scores.append(0.0)
            rouge_scores.append(0.0)
            continue
        gold_tok, pred_tok = gold.split(), pred.split()
        bleu_scores.append(sentence_bleu([gold_tok], pred_tok, smoothing_function=smoothing))
        meteor_scores.append(meteor_score([gold_tok], pred_tok))
        rouge_scores.append(rouge.score(gold, pred)["rougeL"].fmeasure)

    try:
        from pycocoevalcap.cider.cider import Cider
        cider_score, _ = Cider().compute_score(cider_gts, cider_res)
    except ImportError:
        cider_score = None

    results = {
        "n_events_evaluated": n,
        "n_generation_errors": generation_errors,
        "taxonomy": taxonomy,
        "accuracy": round(accuracy, 4),
        "precision_macro": round(precision_macro, 4),
        "recall_macro": round(recall_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(precision_weighted, 4),
        "recall_weighted": round(recall_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
        "bleu4_mean": round(sum(bleu_scores) / n, 4),
        "meteor_mean": round(sum(meteor_scores) / n, 4),
        "rougeL_mean": round(sum(rouge_scores) / n, 4),
        "cider_mean": round(cider_score, 4) if cider_score is not None else "install pycocoevalcap",
        "json_parse_failure_rate": round(parse_failures / n, 4),
        "hallucination_rate": round(hallucinations / n, 4),
    }

    csv_path = os.path.join(ADAPTER_PATH, "eval_per_event_predictions.csv")
    pd.DataFrame(per_event_rows).to_csv(csv_path, index=False)

    print("\n===== EVALUATION SUMMARY =====")
    for k, v in results.items():
        if k == "taxonomy":
            continue
        print(f"{k}: {v}")
    print(f"Per-event predictions saved to: {csv_path}")

    return results


if __name__ == "__main__":
    main()