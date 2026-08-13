
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
import sys
import re
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
    PaliGemmaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG - PaliGemma-3B QLoRA, parameterized per-rider version of
# paligemma_rider1.py. Same Rider1_NJ data/label/split methodology,
# same architectural notes (single-image SigLIP input, LoRA restricted
# to language_model self_attn via regex target_modules) apply here.
# ============================================================
if len(sys.argv) < 2:
    raise SystemExit("Usage: python paligemma_riderX.py <RiderName e.g. Rider2_AZ>")
RIDER = sys.argv[1]

MODEL_PATH = os.path.join(SCRATCH_ROOT, "models/paligemma-3b-pt-224")

VIDEO_DIR = fos.path.join(SCRATCH_ROOT, "custom_data/{RIDER}")
VIDEO_FILENAME = f"{RIDER}_720p.mp4"
CSV_PATH = fos.path.join(SCRATCH_ROOT, "custom_data/{RIDER}/gold_candidates.csv")

OUTPUT_DIR = fos.path.join(SCRATCH_ROOT, "paligemma3b_{RIDER.lower()}_lora_out")

RESIZE_MAX_SIDE = 448
MIN_CLIP_DURATION = 1.0
VAL_SPLIT = 0.2
SEED = 42

CACHE_DIR = os.path.join(OUTPUT_DIR, "tensor_cache_paligemma3b_r448_f1")

USE_4BIT = True

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = r".*language_model.*self_attn\.(q_proj|k_proj|v_proj|o_proj)$"

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 1e-4
MAX_EPOCHS = 7
EARLY_STOP_PATIENCE = 4
WARMUP_RATIO = 0.03

random.seed(SEED)
torch.manual_seed(SEED)


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
        "You are analyzing a single representative dashcam frame from a "
        "two-wheeler for evasive-maneuver classification. Given the image and "
        "the telemetry summary below, respond with ONLY a JSON object of this "
        "exact shape:\n"
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


def sample_mid_frame(video_path: str, start: float, end: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_f = max(int(start * fps), 0)
    end_f = max(int(end * fps), start_f + 1)
    mid_f = (start_f + end_f) // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid_f))
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_f))
        ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"No frame extracted from {video_path} [{start}s-{end}s]")

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img.thumbnail((RESIZE_MAX_SIDE, RESIZE_MAX_SIDE), Image.LANCZOS)
    return img


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
        frame = sample_mid_frame(self.video_path, start, end)

        prompt_text = self.schema_instructions + "\n" + build_telemetry_summary(row)
        target_json = build_target_json(row)

        full = self.processor(
            images=frame, text=prompt_text, suffix=target_json,
            return_tensors="pt", padding="longest",
        )
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

    print(f"=== PaliGemma-3B QLoRA fine-tune for {RIDER} ===")
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    print("Loading model in 4-bit (QLoRA)..." if USE_4BIT else "Loading model in fp16...")
    quant_config = None
    if USE_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = PaliGemmaForConditionalGeneration.from_pretrained(
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

    trainable_module_names = sorted({
        n.rsplit(".lora_", 1)[0]
        for n, p in model.named_parameters()
        if p.requires_grad and ".lora_" in n
    })
    vision_trainable = [n for n in trainable_module_names if "vision_tower" in n]
    lm_trainable = [n for n in trainable_module_names if "language_model" in n]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VERIFIED: {len(trainable_module_names)} modules received LoRA adapters "
          f"({len(lm_trainable)} in language_model, {len(vision_trainable)} in vision_tower).")
    print(f"VERIFIED: trainable_params={trainable_params} / total_params={total_params} "
          f"({100 * trainable_params / total_params:.4f}%)")
    with open(os.path.join(OUTPUT_DIR, "trainable_modules_verified.json"), "w") as f:
        json.dump({
            "trainable_module_names": trainable_module_names,
            "n_language_model_modules": len(lm_trainable),
            "n_vision_tower_modules": len(vision_trainable),
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_pct": 100 * trainable_params / total_params,
        }, f, indent=2)

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

    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params_list, lr=LEARNING_RATE)

    steps_per_epoch = max(len(train_loader) // GRAD_ACCUM_STEPS, 1)
    total_steps = steps_per_epoch * MAX_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(int(total_steps * WARMUP_RATIO), 1),
        num_training_steps=total_steps,
    )

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
                torch.nn.utils.clip_grad_norm_(trainable_params_list, max_norm=1.0)
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

    print(f"Training complete for {RIDER}. Best adapter at: {best_ckpt_dir}")


if __name__ == "__main__":
    main()
