
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


import os
os.environ["HF_HOME"] = os.path.join(HOME_ROOT, "IRASTE/hf_cache")
os.environ["HF_HUB_CACHE"] = os.path.join(HOME_ROOT, "IRASTE/hf_cache")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HOME_ROOT, "IRASTE/hf_cache")

import re
import json
import math
import argparse
import collections
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import PeftModel

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Warning: Failed to load NLTK data: {e}")

# ----------------------------------------------------------------------
# Config -- edit these to match your setup, or override via CLI args
# ----------------------------------------------------------------------
MODEL_ID = os.path.join(SCRATCH_ROOT, "InternVL2-8B")
ADAPTER_PATH = os.path.join(HOME_ROOT, "IRASTE/outputs/internvl2_8b_lora_v6_ckpt_select/BEST_CHECKPOINT")
VAL_JSON = os.path.join(HOME_ROOT, "IRASTE/outputs/val_sft.json")
MEDIA_ROOT = os.path.join(HOME_ROOT, "IRASTE")
OUTPUT_REPORT = os.path.join(HOME_ROOT, "IRASTE/outputs/evaluation_report.json")

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
ROAD_SAFETY_ELEMENTS = ["stop sign", "traffic signal", "red light", "pedestrian crossing",
                        "zebra crossing", "speed limit", "double yellow line"]
LABELS = ["swerve", "sudden brake", "lane drift", "none"]

# ----------------------------------------------------------------------
# Image preprocessing (matches FineTune.py exactly -- must stay in sync
# with how the model was trained, or evaluation numbers won't be valid)
# ----------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = []
    for i in range(min_num, max_num + 1):
        for j in range(min_num, max_num + 1):
            if i * j <= max_num:
                target_ratios.append((i, j))
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height), Image.Resampling.BILINEAR)
    processed_images = []
    for i in range(blocks):
        box = (
            (i % target_aspect_ratio[0]) * image_size,
            (i // target_aspect_ratio[0]) * image_size,
            ((i % target_aspect_ratio[0]) + 1) * image_size,
            ((i // target_aspect_ratio[0]) + 1) * image_size
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) > 1:
        processed_images.append(image.resize((image_size, image_size), Image.Resampling.BILINEAR))
    return processed_images

def load_pixel_values(image_paths, media_root):
    transform = build_transform(448)
    tiles = []
    for img_path in image_paths:
        full_path = os.path.join(media_root, img_path)
        image = Image.open(full_path).convert('RGB')
        blocks = dynamic_preprocess(image, min_num=1, max_num=1, image_size=448, use_thumbnail=False)
        tiles.extend([transform(b) for b in blocks])
    return torch.stack(tiles)

# ----------------------------------------------------------------------
# Model loading -- base model + adapter ONLY, no training
# ----------------------------------------------------------------------
def load_model_and_tokenizer(model_id, adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModel.from_pretrained(
        model_id, quantization_config=quantization_config,
        trust_remote_code=True, device_map="auto"
    )
    img_context_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_id

    model = PeftModel.from_pretrained(model, adapter_path)
    model.img_context_token_id = img_context_id
    for m_obj in [model.base_model, getattr(model.base_model, 'model', None)]:
        if m_obj:
            m_obj.img_context_token_id = img_context_id

    model.eval()
    return model, tokenizer, img_context_id

# ----------------------------------------------------------------------
# JSON parsing of model output
# ----------------------------------------------------------------------
def parse_prediction(text):
    """Returns (label, reasoning, parse_ok)."""
    cleaned = text.strip()
    try:
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()
        obj = json.loads(cleaned)
        label = str(obj.get("label", "none")).strip().lower()
        reasoning = str(obj.get("reasoning", ""))
        if label not in LABELS:
            label = "none"
        return label, reasoning, True
    except Exception:
        return "none", text, False

def parse_reference(text):
    try:
        obj = json.loads(text)
        label = str(obj.get("label", "none")).strip().lower()
        reasoning = str(obj.get("reasoning", ""))
        return label, reasoning
    except Exception:
        return "none", text

# ----------------------------------------------------------------------
# CIDEr (self-contained, standard TF-IDF n-gram implementation --
# no pycocoevalcap dependency needed)
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

        # document frequency: number of references (documents) each n-gram appears in
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
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

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
        pred_tok = nltk.word_tokenize(pred_r.lower())
        ref_tok = nltk.word_tokenize(true_r.lower())
        bleus.append(sentence_bleu([ref_tok], pred_tok, smoothing_function=smooth))
        try:
            meteors.append(meteor_score([ref_tok], pred_tok))
        except Exception:
            meteors.append(0.0)
        rouges.append(scorer.score(true_r, pred_r)['rougeL'].fmeasure)

    cider_score = CiderScorer().compute(pred_reasons, true_reasons)

    hall_rates = []
    for pred_r, true_r in zip(pred_reasons, true_reasons):
        p_lower, r_lower = pred_r.lower(), true_r.lower()
        hall, tot = 0, 0
        for elem in ROAD_SAFETY_ELEMENTS:
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
        "bleu4_mean": round(float(np.mean(bleus)), 4),
        "meteor_mean": round(float(np.mean(meteors)), 4),
        "rougeL_mean": round(float(np.mean(rouges)), 4),
        "cider": round(float(cider_score), 4),
        "json_parse_failure_rate": round(float(json_parse_failure_rate), 4),
        "hallucination_rate": round(float(np.mean(hall_rates)), 4),
    }

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained InternVL2 LoRA adapter -- no training.")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--adapter_path", default=ADAPTER_PATH)
    parser.add_argument("--val_json", default=VAL_JSON)
    parser.add_argument("--media_root", default=MEDIA_ROOT)
    parser.add_argument("--output_report", default=OUTPUT_REPORT)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading base model from: {args.model_id}")
    print(f"Loading adapter from:    {args.adapter_path}")
    model, tokenizer, img_context_id = load_model_and_tokenizer(args.model_id, args.adapter_path)
    print(f"img_context_token_id resolved to: {img_context_id}")

    with open(args.val_json, "r") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} validation samples from {args.val_json}")

    generation_config = dict(max_new_tokens=128, do_sample=False)

    pred_labels, true_labels = [], []
    pred_reasons, true_reasons = [], []
    parse_ok_flags = []
    n_generation_errors = 0

    with torch.no_grad():
        for i, sample in enumerate(samples):
            user_msg_raw = sample["conversations"][0]["value"]
            ref_text = sample["conversations"][1]["value"]
            image_paths = sample["image"]

            pixel_values = load_pixel_values(image_paths, args.media_root).to(device).to(model.dtype)

            try:
                response = model.chat(tokenizer, pixel_values, user_msg_raw, generation_config)
            except Exception as gen_err:
                print(f"  Warning: generation failed for sample {i} ({gen_err}); recording empty response.")
                response = ""
                n_generation_errors += 1

            pred_label, pred_reason, parse_ok = parse_prediction(response)
            true_label, true_reason = parse_reference(ref_text)

            pred_labels.append(pred_label)
            true_labels.append(true_label)
            pred_reasons.append(pred_reason)
            true_reasons.append(true_reason)
            parse_ok_flags.append(parse_ok)

            if (i + 1) % 10 == 0 or (i + 1) == len(samples):
                print(f"Generated {i + 1}/{len(samples)} outputs...")

    if n_generation_errors == len(samples):
        print("*** ALL generations failed -- aborting, metrics would be meaningless. ***")
        return

    metrics = compute_all_metrics(pred_labels, true_labels, pred_reasons, true_reasons, parse_ok_flags)

    report = {
        "n_events_evaluated": len(samples),
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

if __name__ == "__main__":
    main()