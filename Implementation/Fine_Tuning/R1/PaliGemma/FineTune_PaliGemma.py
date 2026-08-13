"""
FineTune_PaliGemma.py
Fine-tunes PaliGemma-3B on the IMU-triggered evasive-event dataset.

PaliGemma takes exactly ONE image per example (no native video/multi-frame
input), so 4 sampled frames are tiled into a single 224x224 montage
(2x2 grid of 112px tiles) to preserve the temporal motion cue.

Vision tower + projector are frozen; LoRA adapts only the Gemma language
model. Frame reads are resilient to corrupted GoPro HEVC data: decord's
background decoder thread dies permanently on some malformed footage, so
on failure we discard the reader and open a fresh one, retrying a bounded
number of times before skipping just that one example.
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
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from decord import VideoReader, cpu
from decord._ffi.base import DECORDError
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

from transformers import (
    PaliGemmaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ----------------------------- CONFIG ---------------------------------- #
MODEL_PATH   = os.path.join(SCRATCH_ROOT, "models/paligemma-3b-pt-224")
VIDEO_DIR    = os.path.join(HOME_ROOT, "IRASTE")
VIDEO_FILENAME = "GX019940.MP4"
CSV_PATH     = os.path.join(HOME_ROOT, "IRASTE/DAY_2/candidate_events_updated.csv")
OUTPUT_DIR   = os.path.join(SCRATCH_ROOT, "runs/paligemma_evasive")

FRAMES_PER_EVENT = 4          # tiled into a 2x2 grid -> 1 image
TILE_SIDE        = 112        # 2 tiles per side * 112 = 224 (model's native input size)
PAD_SECONDS      = 0.5        # symmetric pad for near-zero-duration events

MAX_EPOCHS           = 10
EARLY_STOP_PATIENCE  = 3      # epochs with no val macro-F1 improvement
LR                   = 2e-4
LORA_R               = 16
LORA_ALPHA           = 32
LORA_DROPOUT         = 0.05
SEED                 = 42
VAL_FRACTION         = 0.2
MAX_READER_RETRIES   = 2      # reopen the video this many times before giving up on one event

# Matches only inside a submodule path containing "language_model", never
# inside the (identically-named) SigLIP vision tower attention layers.
# Not anchored to the start of the path since transformers has nested
# PaliGemma's submodules differently across versions (top-level vs under
# a wrapping ".model" - see the vision-tower detection above).
LORA_TARGET_REGEX = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
# ------------------------------------------------------------------------ #

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


class FrameReadError(Exception):
    """Raised when a video segment cannot be decoded even after reopening
    the reader MAX_READER_RETRIES times. Callers should skip the example."""


# --------------------------- Resilient video reading ---------------------------- #
_reader_cache = {}


def get_reader(video_path):
    if video_path not in _reader_cache:
        _reader_cache[video_path] = VideoReader(video_path, ctx=cpu(0))
    return _reader_cache[video_path]


def reset_reader(video_path):
    """decord's background decode thread can die permanently on corrupted
    HEVC data (fatal 'Check failed: run_.load()'); the only recovery is a
    brand new VideoReader instance, which spins up a fresh thread."""
    _reader_cache[video_path] = VideoReader(video_path, ctx=cpu(0))
    return _reader_cache[video_path]


def get_fps(video_path):
    return get_reader(video_path).get_avg_fps()


def read_frames_safe(video_path, frame_idxs):
    last_err = None
    for attempt in range(MAX_READER_RETRIES + 1):
        vr = get_reader(video_path)
        try:
            return vr.get_batch(frame_idxs).asnumpy()
        except DECORDError as e:
            last_err = e
            print(f"[decord] read failed (attempt {attempt + 1}/{MAX_READER_RETRIES + 1}) "
                  f"for frames {frame_idxs}: {e}. Reopening reader.")
            reset_reader(video_path)
    raise FrameReadError(
        f"Could not read frames {frame_idxs} after {MAX_READER_RETRIES + 1} attempts: {last_err}"
    )


# --------------------------- Label derivation ---------------------------- #
def derive_label_from_notes(row):
    notes = str(row.get("notes", "")).lower()
    decision = str(row.get("decision", "")).lower()

    not_evasive_kw = ["riding straight", "no visible change", "no change in heading",
                       "camera vibration", "sensor noise", "constant speed"]
    if decision == "reject" or any(k in notes for k in not_evasive_kw):
        return "not_evasive"
    if any(k in notes for k in ["swerv", "avoid"]):
        return "swerve"
    if any(k in notes for k in ["sharp turn"]):
        return "sharp_turn"
    if any(k in notes for k in ["accelerat"]):
        return "acceleration"
    if any(k in notes for k in ["lane", "pass"]):
        return "lane_change"
    return "other"


def build_clip_bounds(row):
    start, end = float(row["start_time"]), float(row["end_time"])
    if end - start < 0.05:
        start = max(0.0, start - PAD_SECONDS)
        end = end + PAD_SECONDS
    return start, end


# ----------------------------- Frame sampling ---------------------------- #
def sample_montage(video_path, fps, start, end):
    """Sample FRAMES_PER_EVENT frames evenly across [start, end] and tile
    them into a single 224x224 image (2x2 grid of 112px tiles) for
    PaliGemma's single-image input. Raises FrameReadError if the segment
    cannot be decoded even after reopening the reader."""
    vr = get_reader(video_path)
    n_total = len(vr)
    idxs = np.linspace(start, end, FRAMES_PER_EVENT)
    frame_idxs = [min(n_total - 1, max(0, int(t * fps))) for t in idxs]
    frames = read_frames_safe(video_path, frame_idxs)

    tiles = []
    for f in frames:
        img = Image.fromarray(f).convert("RGB")
        img.thumbnail((TILE_SIDE, TILE_SIDE))
        canvas = Image.new("RGB", (TILE_SIDE, TILE_SIDE), (0, 0, 0))
        canvas.paste(img, (0, 0))
        tiles.append(canvas)

    if FRAMES_PER_EVENT == 4:
        montage = Image.new("RGB", (TILE_SIDE * 2, TILE_SIDE * 2))
        montage.paste(tiles[0], (0, 0))
        montage.paste(tiles[1], (TILE_SIDE, 0))
        montage.paste(tiles[2], (0, TILE_SIDE))
        montage.paste(tiles[3], (TILE_SIDE, TILE_SIDE))
    else:
        montage = Image.new("RGB", (TILE_SIDE * FRAMES_PER_EVENT, TILE_SIDE))
        for i, t in enumerate(tiles):
            montage.paste(t, (TILE_SIDE * i, 0))
    return montage


# ----------------------------- Prompt building ---------------------------- #
def telemetry_summary(row):
    return (
        f"peak_jerk={row['peak_abs_z_jerk']:.2f}, "
        f"peak_lateral_accel={row['peak_abs_z_az']:.2f}, "
        f"duration={row['duration']:.2f}s, "
        f"n_samples={int(row['n_samples'])}, "
        f"confidence={row['peak_confidence']:.2f}, "
        f"imu_decision={row['decision']}"
    )


PROMPT_PREFIX = (
    "<image>classify the two-wheeler maneuver in this 4-frame sequence. "
    "telemetry: {telemetry}. "
    "categories: lane_change, swerve, sharp_turn, acceleration, not_evasive, other. answer:"
)


# ----------------------------- Dataset build ---------------------------- #
def build_examples(csv_path, video_path):
    df = pd.read_csv(csv_path)
    df["_label"] = df.apply(derive_label_from_notes, axis=1)
    print("Total usable events:", len(df))
    print("Label distribution:\n", df["_label"].value_counts())

    counts = df["_label"].value_counts()
    rare = counts[counts < 2].index.tolist()
    if rare:
        print("Dropping ultra-rare classes (<2 examples):", rare)
        df = df[~df["_label"].isin(rare)].reset_index(drop=True)

    taxonomy = sorted(df["_label"].unique().tolist())
    print("Taxonomy:", taxonomy)

    fps = get_fps(video_path)

    examples = []
    for _, row in df.iterrows():
        start, end = build_clip_bounds(row)
        examples.append({
            "start": start, "end": end,
            "prompt": PROMPT_PREFIX.format(telemetry=telemetry_summary(row)),
            "label": row["_label"],
        })
    return examples, taxonomy, fps


# ----------------------------- Collate / forward ---------------------------- #
def prepare_batch(processor, video_path, fps, example, device, dtype):
    montage = sample_montage(video_path, fps, example["start"], example["end"])
    inputs = processor(
        text=example["prompt"],
        images=montage,
        suffix=example["label"],
        return_tensors="pt",
    )
    return {k: v.to(device=device, dtype=dtype if v.dtype.is_floating_point else v.dtype)
            for k, v in inputs.items()}


def is_oom_error(e: Exception) -> bool:
    """Catches both torch.cuda.OutOfMemoryError and plain RuntimeError('out of memory')."""
    return "out of memory" in str(e).lower()


def oversample_train(examples):
    """Duplicate minority-class examples (with replacement) up to the
    majority class count. Without this, cross-entropy on a lane_change-
    dominated (174/208) training set collapses greedy generation to always
    predicting lane_change, since that minimizes average loss even though
    it never predicts the other three classes correctly."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for e in examples:
        buckets[e["label"]].append(e)
    max_count = max(len(v) for v in buckets.values())
    balanced = []
    for label, items in buckets.items():
        reps, remainder = divmod(max_count, len(items))
        balanced.extend(items * reps)
        balanced.extend(random.sample(items, remainder))
    random.shuffle(balanced)
    return balanced


# --------------------------------- Main --------------------------------- #
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    print("Loading model in 4-bit (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_PATH, quantization_config=bnb_config, device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model)

    # --- Freeze vision tower + projector ---
    # transformers has changed PaliGemma's module layout across versions
    # (sometimes top-level: model.vision_tower / model.multi_modal_projector,
    # sometimes nested: model.model.vision_tower / model.model.multi_modal_projector).
    # Locate both by submodule class name instead of hardcoding the path, so
    # this keeps working regardless of which layout the installed version uses.
    vision_tower, projector = None, None
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if vision_tower is None and "VisionModel" in cls_name:
            vision_tower = (name, module)
        if projector is None and "MultiModalProjector" in cls_name:
            projector = (name, module)

    if vision_tower is None or projector is None:
        found = [(n, type(m).__name__) for n, m in model.named_children()]
        raise RuntimeError(
            f"Could not locate vision tower / projector submodules automatically. "
            f"Top-level children: {found}. Inspect these and hardcode the correct "
            f"attribute path (e.g. model.model.vision_tower)."
        )

    print(f"Freezing vision tower at '{vision_tower[0]}' ({type(vision_tower[1]).__name__})")
    print(f"Freezing projector at '{projector[0]}' ({type(projector[1]).__name__})")
    for p in vision_tower[1].parameters():
        p.requires_grad = False
    for p in projector[1].parameters():
        p.requires_grad = False

    # --- LoRA-adapt only the Gemma language model ---
    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_REGEX,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Sanity check: confirm no LoRA params landed inside the vision tower
    leaked = [n for n, p in model.named_parameters()
              if p.requires_grad and "vision_tower" in n]
    if leaked:
        raise RuntimeError(f"LoRA leaked into vision tower: {leaked}")

    video_path = os.path.join(VIDEO_DIR, VIDEO_FILENAME)
    examples, taxonomy, fps = build_examples(CSV_PATH, video_path)

    labels = [e["label"] for e in examples]
    train_ex, val_ex = train_test_split(
        examples, test_size=VAL_FRACTION, random_state=SEED, stratify=labels
    )
    print(f"Train events: {len(train_ex)} | Val events: {len(val_ex)}")

    from collections import Counter
    print("Train label distribution BEFORE oversampling:", Counter(e["label"] for e in train_ex))
    train_ex = oversample_train(train_ex)
    print("Train label distribution AFTER oversampling:", Counter(e["label"] for e in train_ex))
    print(f"Train events after oversampling: {len(train_ex)}")

    with open(os.path.join(OUTPUT_DIR, "taxonomy.json"), "w") as f:
        json.dump(taxonomy, f)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )
    total_steps = MAX_EPOCHS * len(train_ex)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.03 * total_steps), num_training_steps=total_steps
    )

    best_f1, patience_left = -1.0, EARLY_STOP_PATIENCE
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        random.shuffle(train_ex)
        running_loss, n_ok, n_skipped = 0.0, 0, 0
        for step, ex in enumerate(train_ex, 1):
            try:
                batch = prepare_batch(processor, video_path, fps, ex, device, torch.bfloat16)
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                running_loss += loss.item()
                n_ok += 1
            except FrameReadError as e:
                print(f"[epoch {epoch}] step {step} SKIPPED (unreadable video segment): {e}")
                n_skipped += 1
                continue
            except RuntimeError as e:
                if is_oom_error(e):
                    print(f"[epoch {epoch}] step {step} SKIPPED due to OOM: {e}")
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    n_skipped += 1
                    continue
                raise
        print(f"[epoch {epoch}] mean train loss: {running_loss / max(1, n_ok):.4f} "
              f"({n_ok} ok, {n_skipped} skipped)")

        # --------------------------- validation --------------------------- #
        model.eval()
        preds, gold = [], []
        with torch.no_grad():
            for ex in val_ex:
                try:
                    montage = sample_montage(video_path, fps, ex["start"], ex["end"])
                except FrameReadError as e:
                    print(f"[val] SKIPPED (unreadable video segment): {e}")
                    continue
                inputs = processor(text=ex["prompt"], images=montage, return_tensors="pt").to(
                    device, torch.bfloat16
                )
                gen = model.generate(**inputs, max_new_tokens=8)
                pred_text = processor.decode(
                    gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                ).strip().lower()
                pred = pred_text if pred_text in taxonomy else "other"
                preds.append(pred)
                gold.append(ex["label"])

        acc = accuracy_score(gold, preds)
        f1_macro = f1_score(gold, preds, average="macro", labels=taxonomy, zero_division=0)
        print(f"[epoch {epoch}] val_accuracy={acc:.4f} val_f1_macro={f1_macro:.4f} "
              f"(evaluated {len(gold)}/{len(val_ex)})")
        from collections import Counter
        print(f"[epoch {epoch}] val prediction distribution: {Counter(preds)}")

        if f1_macro > best_f1:
            best_f1 = f1_macro
            patience_left = EARLY_STOP_PATIENCE
            model.save_pretrained(os.path.join(OUTPUT_DIR, "best_adapter"))
            processor.save_pretrained(os.path.join(OUTPUT_DIR, "best_adapter"))
            print(f"  -> new best (f1_macro={best_f1:.4f}), adapter saved")
        else:
            patience_left -= 1
            print(f"  -> no improvement, patience_left={patience_left}")
            if patience_left <= 0:
                print("Early stopping.")
                break

    print("Done. Best val f1_macro:", best_f1)


if __name__ == "__main__":
    main()