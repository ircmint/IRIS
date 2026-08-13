"""
qwen2_rider1_train.py — QLoRA fine-tuning of Qwen2-VL-2B on Rider1_NJ pilot data.

Trains a Qwen2-VL vision-language model to classify evasive driving events
(hard braking, lane changes, acceleration, deceleration) from dashcam clips
paired with gold-annotated telemetry windows.

Dataset: Custom_Data/Rider1_NJ — 130 events (104 train / 26 val after 80/20 split)
Model:   Qwen2-VL-2B-Instruct with 4-bit nf4 QLoRA (r=16, alpha=32)
Output:  LoRA adapter saved to OUTPUT_DIR at best validation F1

Run on Ada HPC (gnode027, RTX 2080 Ti):
    sbatch run_qwen2_rider1.sbatch
    # or directly:
    CUDA_VISIBLE_DEVICES=0 python qwen2_rider1_train.py 2>&1 | tee train_r1.log
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
import traceback
import gc

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import cv2
import numpy as np
from PIL import Image

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG - Rider1_NJ variant of qwen2.py (pilot GX019940 script).
# Adapted for the 4-rider "Custom_Data" dataset, Rider1_NJ only.
# Data confirmed present on gnode027 (ssd_scratch is node-local scratch,
# only visible from within an active job on that node) as of this run:
#   $SCRATCH_ROOT/custom_data/Rider1_NJ/Rider1_NJ_720p.mp4
#   $SCRATCH_ROOT/custom_data/Rider1_NJ/gold_candidates.csv
# NOTE: gold_candidates.csv columns for this dataset are actually:
#   group_id,category,start_time,end_time,duration,n_merged,raw_indices,decision,notes
# (NOT the clip_id/adjusted_start/... schema originally assumed - verified
# by directly reading the file header on gnode027, since the node crashed
# and recovered earlier and paths/schemas were not to be trusted blindly.)
# ============================================================
MODEL_PATH = os.path.join(SCRATCH_ROOT, "models/Qwen2-VL-2B-Instruct/")

VIDEO_DIR = os.path.join(SCRATCH_ROOT, "custom_data/Rider1_NJ")
VIDEO_FILENAME = "Rider1_NJ_720p.mp4"
CSV_PATH = os.path.join(SCRATCH_ROOT, "custom_data/Rider1_NJ/gold_candidates.csv")

OUTPUT_DIR = os.path.join(SCRATCH_ROOT, "qwen2vl_2b_rider1_lora_out")

FRAMES_PER_EVENT = 4
RESIZE_MAX_SIDE = 448
MIN_PIXELS = 64 * 28 * 28
MAX_PIXELS = 448 * 448
MIN_CLIP_DURATION = 1.0
VAL_SPLIT = 0.2
SEED = 42

CACHE_DIR = os.path.join(OUTPUT_DIR, f"tensor_cache_2b_r{RESIZE_MAX_SIDE}_f{FRAMES_PER_EVENT}")

USE_4BIT = True

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 1e-4
# Scope reduction vs pilot (MAX_EPOCHS=5): Rider1_NJ only has 130 events
# (104 train / 26 val) inside a firm 4-hour wall-clock window that must
# also leave time for evaluate_rider1.py to run to completion. Capped at
# 3 epochs with early stopping so the run reaches a real, complete
# evaluation rather than eating the whole window on training alone.
MAX_EPOCHS = 3
EARLY_STOP_PATIENCE = 2
WARMUP_RATIO = 0.03

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Label derivation for Rider1_NJ.
#
# Rider1_NJ's gold_candidates.csv has two real, hand-reviewed signal
# columns instead of a single evasive_type/notes-keyword field:
#   - decision: confirm/reject (was this candidate event actually a real
#     evasive maneuver, per human review of the merged IMU trigger)
#   - category: acceleration/lane_change/zigzag/deceleration/braking
#     (assigned to every candidate regardless of decision - it describes
#     what kind of motion signature triggered the candidate, not whether
#     it was confirmed as a genuine hazard)
#
# To stay structurally consistent with the pilot's taxonomy (which had a
# "not_evasive" catch-all bucket plus a handful of real maneuver labels,
# see qwen2.py derive_label_from_notes), the Rider1_NJ label combines the
# two exactly analogously:
#   decision == "reject"  -> "not_evasive"
#   decision == "confirm" -> the reviewed category (deceleration/
#                             lane_change/acceleration/zigzag/braking)
# This uses the actual hand-reviewed decision+category fields directly,
# which is more reliable ground truth than re-deriving from free-text
# notes the way the pilot script had to (Rider1_NJ has both signals
# available; the pilot's candidate_events_updated.csv did not).
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
    # Rider1_NJ's CSV has no adjusted_start/adjusted_end columns (unlike
    # the pilot CSV) - row.get() returns None for missing keys, which
    # falls through to start_time/end_time below, same as qwen2.py.
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
    # Rider1_NJ's CSV lacks peak_confidence/peak_abs_z_jerk/peak_abs_z_az/
    # n_samples (those are pilot-CSV-only columns) - use the fields that
    # actually exist here: duration, n_merged (how many raw IMU triggers
    # were merged into this candidate window), the upstream trigger
    # category tag, and the human decision.
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


def build_target_json(row: pd.Series) -> str:
    return json.dumps({
        "label": get_label(row),
        "confidence": 1.0,
        "reasoning": get_reasoning(row),
    })


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


class EvasiveEventDataset(Dataset):
    def __init__(self, df: pd.DataFrame, video_path: str, processor, taxonomy,
                 cache_dir: str, split_name: str):
        self.df = df.reset_index(drop=True)
        self.video_path = video_path
        self.processor = processor
        self.schema_instructions = build_schema_instructions(taxonomy)
        self.cache_dir = cache_dir
        self.split_name = split_name
        os.makedirs(cache_dir, exist_ok=True)
        self._mem_cache = {}

    def __len__(self):
        return len(self.df)

    def _compute(self, row):
        start, end = get_clip_bounds(row)
        frames = sample_frames(self.video_path, start, end, FRAMES_PER_EVENT)

        prompt_text_body = self.schema_instructions + "\n" + build_telemetry_summary(row)
        target_json = build_target_json(row)

        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": f} for f in frames]
            + [{"type": "text", "text": prompt_text_body}],
        }]

        prompt_chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_chat_text = prompt_chat_text + target_json
        image_inputs, video_inputs = process_vision_info(messages)

        prompt_only = self.processor(
            text=[prompt_chat_text], images=image_inputs, videos=video_inputs,
            return_tensors="pt",
        )
        full = self.processor(
            text=[full_chat_text], images=image_inputs, videos=video_inputs,
            return_tensors="pt",
        )

        prompt_len = prompt_only["input_ids"].shape[1]
        labels = full["input_ids"].clone()
        labels[:, :prompt_len] = -100
        full["labels"] = labels
        return full

    def __getitem__(self, i):
        if i in self._mem_cache:
            return self._mem_cache[i]

        cache_path = os.path.join(self.cache_dir, f"{self.split_name}_{i}.pt")
        if os.path.exists(cache_path):
            item = torch.load(cache_path)
            self._mem_cache[i] = item
            return item

        row = self.df.iloc[i]
        item = self._compute(row)
        torch.save(item, cache_path)
        self._mem_cache[i] = item
        return item


def collate_fn(batch):
    return batch[0]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
    )

    print("Loading model in 4-bit (QLoRA)..." if USE_4BIT else "Loading model in fp16...")
    quant_config = None
    if USE_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    if USE_4BIT:
        model = prepare_model_for_kbit_training(model)
        print(f"GPU memory after quantized load: "
              f"{torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    for param in model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---------------- Data ----------------
    df = pd.read_csv(CSV_PATH)
    df["_label"] = df.apply(get_label, axis=1)
    print(f"Total usable events: {len(df)}")
    print("Label distribution:\n", df["_label"].value_counts())

    counts = df["_label"].value_counts()
    rare = counts[counts < 2].index.tolist()
    if rare:
        print(f"Dropping ultra-rare classes (<2 examples): {rare}")
        df = df[~df["_label"].isin(rare)].reset_index(drop=True)

    taxonomy = sorted(df["_label"].unique().tolist())
    print("Taxonomy:", taxonomy)
    with open(os.path.join(OUTPUT_DIR, "taxonomy.json"), "w") as f:
        json.dump(taxonomy, f, indent=2)

    try:
        train_df, val_df = train_test_split(
            df, test_size=VAL_SPLIT, random_state=SEED, stratify=df["_label"]
        )
    except ValueError:
        print("Stratified split failed - using random split.")
        train_df, val_df = train_test_split(df, test_size=VAL_SPLIT, random_state=SEED)

    train_df, val_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True)
    print(f"Train events: {len(train_df)} | Val events: {len(val_df)}")
    print("Train label distribution:\n", train_df["_label"].value_counts())

    video_path = os.path.join(VIDEO_DIR, VIDEO_FILENAME)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found at {video_path} - check VIDEO_DIR/VIDEO_FILENAME.")

    train_ds = EvasiveEventDataset(train_df, video_path, processor, taxonomy, CACHE_DIR, "train")
    val_ds = EvasiveEventDataset(val_df, video_path, processor, taxonomy, CACHE_DIR, "val")

    label_counts = train_df["_label"].value_counts()
    class_weight = {lbl: 1.0 / cnt for lbl, cnt in label_counts.items()}
    sample_weights = train_df["_label"].map(class_weight).values
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # ---------------- Optimizer / schedule ----------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)

    steps_per_epoch = max(len(train_loader) // GRAD_ACCUM_STEPS, 1)
    total_steps = steps_per_epoch * MAX_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(int(total_steps * WARMUP_RATIO), 1),
        num_training_steps=total_steps,
    )

    # ---------------- Training loop with early stopping ----------------
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_ckpt_dir = os.path.join(OUTPUT_DIR, "best_adapter")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss, n_ok = 0.0, 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            try:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss / GRAD_ACCUM_STEPS
                loss.backward()
                running_loss += loss.item() * GRAD_ACCUM_STEPS
                n_ok += 1
                del outputs, loss, batch
            except Exception as e:
                print(f"[epoch {epoch}] step {step} SKIPPED due to error: {e}")
                traceback.print_exc()
                optimizer.zero_grad()
                gc.collect()
                torch.cuda.empty_cache()
                continue

            if step % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 20 == 0:
                print(f"[epoch {epoch}] step {step}/{len(train_loader)} "
                      f"running_loss={running_loss / max(n_ok, 1):.4f}")

        train_loss = running_loss / max(n_ok, 1)

        model.eval()
        val_loss_total, n_val_ok = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                try:
                    batch = {k: v.to(model.device) for k, v in batch.items()}
                    outputs = model(**batch)
                    val_loss_total += outputs.loss.item()
                    n_val_ok += 1
                    del outputs, batch
                except Exception as e:
                    print(f"[epoch {epoch}] val step SKIPPED due to error: {e}")
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
        val_loss = val_loss_total / max(n_val_ok, 1)

        print(f"== Epoch {epoch}/{MAX_EPOCHS} - train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} (train_ok={n_ok}/{len(train_loader)}, "
              f"val_ok={n_val_ok}/{len(val_loader)}) ==")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            model.save_pretrained(best_ckpt_dir)
            processor.save_pretrained(best_ckpt_dir)
            with open(os.path.join(best_ckpt_dir, "taxonomy.json"), "w") as f:
                json.dump(taxonomy, f, indent=2)
            print(f"  -> new best (val_loss={val_loss:.4f}), adapter saved to {best_ckpt_dir}")
        else:
            epochs_no_improve += 1
            print(f"  -> no improvement ({epochs_no_improve}/{EARLY_STOP_PATIENCE})")
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}. Best val_loss={best_val_loss:.4f}")
                break

    print(f"Training complete. Best adapter at: {best_ckpt_dir}")


if __name__ == "__main__":
    main()
