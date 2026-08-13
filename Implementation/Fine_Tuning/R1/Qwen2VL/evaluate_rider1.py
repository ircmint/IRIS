"""
evaluate_rider1.py - Evaluate the fine-tuned Qwen2-VL-2B LoRA adapter on
Rider1_NJ's held-out validation split (Custom_Data 4-rider dataset).

Mirrors evaluate.py's structure/metrics exactly, adapted to Rider1_NJ's
CONFIG (see qwen2_rider1.py for the matching training script and full
rationale for the label-derivation and telemetry-summary changes).

Run after qwen2_rider1.py has produced OUTPUT_DIR/best_adapter and
OUTPUT_DIR/taxonomy.json.
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

import os
import json
import re

import torch
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider

for pkg in ["wordnet", "omw-1.4", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg != "punkt" and pkg != "punkt_tab" else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

# ============================================================
# CONFIG - must match qwen2_rider1.py so the val split is identical
# ============================================================
MODEL_PATH = os.path.join(SCRATCH_ROOT, "models/Qwen2-VL-2B-Instruct/")
OUTPUT_DIR = os.path.join(SCRATCH_ROOT, "qwen2vl_2b_rider1_lora_out")
ADAPTER_DIR = os.path.join(OUTPUT_DIR, "best_adapter")

VIDEO_DIR = os.path.join(SCRATCH_ROOT, "custom_data/Rider1_NJ")
VIDEO_FILENAME = "Rider1_NJ_720p.mp4"
CSV_PATH = os.path.join(SCRATCH_ROOT, "custom_data/Rider1_NJ/gold_candidates.csv")

FRAMES_PER_EVENT = 4
RESIZE_MAX_SIDE = 448
MIN_PIXELS = 64 * 28 * 28
MAX_PIXELS = 448 * 448
MIN_CLIP_DURATION = 1.0
VAL_SPLIT = 0.2
SEED = 42

MAX_NEW_TOKENS = 200
RESULTS_JSON_PATH = os.path.join(OUTPUT_DIR, "eval_predictions.json")


# ============================================================
# Same label/clip-bounds logic as qwen2_rider1.py (kept identical on
# purpose - the val split and ground-truth labels must match training)
# ============================================================
def get_label(row: pd.Series) -> str:
    decision = str(row.get("decision", "")).strip().lower()
    if decision == "confirm":
        cat = str(row.get("category", "")).strip().lower()
        return cat if cat else "other"
    return "not_evasive"


def get_reasoning(row: pd.Series) -> str:
    val = row.get("notes")
    if pd.notna(val) and str(val).strip():
        return str(val).strip()
    return "No additional notes provided."


def get_clip_bounds(row: pd.Series):
    start = row.get("adjusted_start")
    end = row.get("adjusted_end")
    if pd.isna(start):
        start = row.get("start_time")
    if pd.isna(end):
        end = row.get("end_time")
    start, end = float(start), float(end)
    if end - start < MIN_CLIP_DURATION:
        center = (start + end) / 2.0
        half = MIN_CLIP_DURATION / 2.0
        start, end = max(center - half, 0.0), center + half
    return start, end


def build_telemetry_summary(row: pd.Series) -> str:
    return (
        f"Telemetry: event duration={row.get('duration', 'NA')}s, "
        f"n_merged_raw_triggers={row.get('n_merged', 'NA')}, "
        f"upstream trigger category={row.get('category', 'NA')}, "
        f"upstream IMU decision={row.get('decision', 'NA')}."
    )


def build_schema_instructions(taxonomy) -> str:
    return (
        "You are analyzing a short dashcam video clip from a two-wheeler for "
        "evasive-maneuver classification. Given the frames and the telemetry "
        "summary below, respond with ONLY a JSON object of this exact shape:\n"
        '{"label": "<one of: ' + ", ".join(taxonomy) + '>", '
        '"confidence": <float 0-1>, "reasoning": "<short justification grounded '
        'only in what is visible/measured>"}\n'
        "Do not include any text outside the JSON object."
    )


import cv2
import numpy as np


def sample_frames(video_path: str, start: float, end: float, n_frames: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_f = max(int(start * fps), 0)
    end_f = max(int(end * fps), start_f + 1)
    idxs = np.linspace(start_f, end_f, n_frames, dtype=int)

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img.thumbnail((RESIZE_MAX_SIDE, RESIZE_MAX_SIDE), Image.LANCZOS)
            frames.append(img)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path} [{start}s-{end}s]")
    while len(frames) < n_frames:
        frames.append(frames[-1])
    return frames


def extract_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def main():
    with open(os.path.join(OUTPUT_DIR, "taxonomy.json")) as f:
        taxonomy = json.load(f)
    print("Taxonomy:", taxonomy)

    df = pd.read_csv(CSV_PATH)
    df["_label"] = df.apply(get_label, axis=1)
    counts = df["_label"].value_counts()
    rare = counts[counts < 2].index.tolist()
    if rare:
        df = df[~df["_label"].isin(rare)].reset_index(drop=True)

    try:
        _, val_df = train_test_split(
            df, test_size=VAL_SPLIT, random_state=SEED, stratify=df["_label"]
        )
    except ValueError:
        _, val_df = train_test_split(df, test_size=VAL_SPLIT, random_state=SEED)
    val_df = val_df.reset_index(drop=True)
    print(f"Val events: {len(val_df)}")

    video_path = os.path.join(VIDEO_DIR, VIDEO_FILENAME)

    print("Loading processor + base model + adapter...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    schema_instructions = build_schema_instructions(taxonomy)

    gt_labels, pred_labels = [], []
    gt_reasonings, pred_reasonings = [], []
    n_generation_errors = 0
    n_json_parse_failures = 0
    n_hallucinated_label = 0
    all_results = []

    for i, row in val_df.iterrows():
        try:
            start, end = get_clip_bounds(row)
            frames = sample_frames(video_path, start, end, FRAMES_PER_EVENT)
            prompt_text_body = schema_instructions + "\n" + build_telemetry_summary(row)

            messages = [{
                "role": "user",
                "content": [{"type": "image", "image": f} for f in frames]
                + [{"type": "text", "text": prompt_text_body}],
            }]
            prompt_chat_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[prompt_chat_text], images=image_inputs, videos=video_inputs,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                gen_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            gen_ids_trimmed = gen_ids[:, inputs["input_ids"].shape[1]:]
            output_text = processor.batch_decode(
                gen_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]

        except Exception as e:
            print(f"[event {i}] GENERATION ERROR: {e}")
            n_generation_errors += 1
            continue

        gt_label = get_label(row)
        gt_reasoning = get_reasoning(row)

        parsed = extract_json(output_text)
        if parsed is None or "label" not in parsed:
            n_json_parse_failures += 1
            pred_label = "other"
            pred_reasoning = ""
            all_results.append({
                "event_idx": int(i), "gt_label": gt_label, "gt_reasoning": gt_reasoning,
                "raw_output": output_text, "parsed": False,
            })
        else:
            pred_label = parsed.get("label", "other")
            pred_reasoning = str(parsed.get("reasoning", ""))
            if pred_label not in taxonomy:
                n_hallucinated_label += 1
                pred_label = "other"
            all_results.append({
                "event_idx": int(i), "gt_label": gt_label, "gt_reasoning": gt_reasoning,
                "pred_label": parsed.get("label", ""), "pred_reasoning": pred_reasoning,
                "raw_output": output_text, "parsed": True,
            })

        gt_labels.append(gt_label)
        pred_labels.append(pred_label)
        gt_reasonings.append(gt_reasoning)
        pred_reasonings.append(pred_reasoning)

    n_events_evaluated = len(gt_labels)
    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    # ---------------- Classification metrics ----------------
    accuracy = accuracy_score(gt_labels, pred_labels) if n_events_evaluated else float("nan")
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        gt_labels, pred_labels, average="macro", zero_division=0
    ) if n_events_evaluated else (float("nan"),) * 3
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        gt_labels, pred_labels, average="weighted", zero_division=0
    ) if n_events_evaluated else (float("nan"),) * 3

    # ---------------- Generation-quality metrics ----------------
    smoothing = SmoothingFunction().method1
    bleu4_scores, meteor_scores, rougeL_scores = [], [], []
    r_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    cider_gts, cider_res = {}, {}

    for idx, (gt, pred) in enumerate(zip(gt_reasonings, pred_reasonings)):
        gt_tokens = nltk.word_tokenize(gt.lower())
        pred_tokens = nltk.word_tokenize(pred.lower()) if pred else [""]

        bleu4_scores.append(sentence_bleu(
            [gt_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing,
        ))
        try:
            meteor_scores.append(meteor_score([gt_tokens], pred_tokens))
        except Exception:
            meteor_scores.append(0.0)
        rougeL_scores.append(r_scorer.score(gt, pred)["rougeL"].fmeasure)

        cider_gts[str(idx)] = [gt]
        cider_res[str(idx)] = [pred if pred.strip() else "."]

    bleu4_mean = sum(bleu4_scores) / len(bleu4_scores) if bleu4_scores else float("nan")
    meteor_mean = sum(meteor_scores) / len(meteor_scores) if meteor_scores else float("nan")
    rougeL_mean = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else float("nan")

    if cider_gts:
        cider_scorer = Cider()
        cider_mean, _ = cider_scorer.compute_score(cider_gts, cider_res)
    else:
        cider_mean = float("nan")

    # ---------------- Robustness metrics ----------------
    denom = n_events_evaluated if n_events_evaluated else 1
    json_parse_failure_rate = n_json_parse_failures / denom
    hallucination_rate = n_hallucinated_label / denom

    # ---------------- Report ----------------
    print("\n===== EVALUATION SUMMARY (Rider1_NJ) =====")
    print(f"n_events_evaluated: {n_events_evaluated}")
    print(f"n_generation_errors: {n_generation_errors}")
    print(f"accuracy: {accuracy:.4f}")
    print(f"precision_macro: {precision_macro:.4f}")
    print(f"recall_macro: {recall_macro:.4f}")
    print(f"f1_macro: {f1_macro:.4f}")
    print(f"precision_weighted: {precision_weighted:.4f}")
    print(f"recall_weighted: {recall_weighted:.4f}")
    print(f"f1_weighted: {f1_weighted:.4f}")
    print(f"bleu4_mean: {bleu4_mean:.4f}")
    print(f"meteor_mean: {meteor_mean:.4f}")
    print(f"rougeL_mean: {rougeL_mean:.4f}")
    print(f"cider_mean: {cider_mean:.4f}")
    print(f"json_parse_failure_rate: {json_parse_failure_rate:.4f}")
    print(f"hallucination_rate: {hallucination_rate:.4f}")
    print(f"\nPer-event predictions saved to: {RESULTS_JSON_PATH}")

    summary = {
        "n_events_evaluated": n_events_evaluated,
        "n_generation_errors": n_generation_errors,
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "bleu4_mean": bleu4_mean,
        "meteor_mean": meteor_mean,
        "rougeL_mean": rougeL_mean,
        "cider_mean": cider_mean,
        "json_parse_failure_rate": json_parse_failure_rate,
        "hallucination_rate": hallucination_rate,
    }
    with open(os.path.join(OUTPUT_DIR, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
