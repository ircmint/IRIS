
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
# CONFIG - PaliGemma-3B QLoRA variant of qwen2_rider1.py, same Rider1_NJ
# data/label/split methodology, for direct architecture comparison.
#
# IMPORTANT ARCHITECTURAL DIFFERENCE FROM qwen2_rider1.py: PaliGemma-3b
# is a single-image VLM (SigLIP vision tower encodes exactly one image
# per forward pass - there is no multi-frame/video input path the way
# Qwen2-VL has via qwen_vl_utils.process_vision_info). Verified against
# the actual PaliGemmaProcessor signature (images= takes one image per
# example). So instead of FRAMES_PER_EVENT=4 sampled frames, this script
# samples ONE representative frame per event (the temporal midpoint of
# the clip). This is a real architectural necessity, not an oversight -
# flagged explicitly here and in the final report.
#
# LoRA TARGET MODULES: verified via named_modules() on the actual loaded
# model (not assumed from Qwen's naming) that PaliGemma's Gemma decoder
# uses q_proj/k_proj/v_proj/o_proj under model.language_model.layers.N.
# self_attn - same leaf names as Qwen2. However the SigLIP vision tower
# ALSO has q_proj/k_proj/v_proj (named out_proj instead of o_proj there),
# so a plain leaf-name target list would leak LoRA into vision attention
# too. To keep the vision tower fully frozen (only the language-model
# decoder gets adapted, matching qwen2_rider1.py's intent of adapting only
# the text-generation path), LORA_TARGET_MODULES below is a REGEX string
# (peft treats a str target_modules as a regex matched with re.search
# against the full dotted module name) restricted to language_model's
# self_attn projections only.
# ============================================================
MODEL_PATH = os.path.join(SCRATCH_ROOT, "models/paligemma-3b-pt-224")

VIDEO_DIR = os.path.join(SCRATCH_ROOT, "custom_data/Rider1_NJ")
VIDEO_FILENAME = "Rider1_NJ_720p.mp4"
CSV_PATH = os.path.join(SCRATCH_ROOT, "custom_data/Rider1_NJ/gold_candidates.csv")

OUTPUT_DIR = os.path.join(SCRATCH_ROOT, "paligemma3b_rider1_lora_out")

RESIZE_MAX_SIDE = 448
MIN_CLIP_DURATION = 1.0
VAL_SPLIT = 0.2
SEED = 42

CACHE_DIR = os.path.join(OUTPUT_DIR, "tensor_cache_paligemma3b_r448_f1")

USE_4BIT = True

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# regex, matched with re.search against the full dotted module name -
# restricts LoRA injection to the Gemma language-model decoder's
# attention projections only; vision_tower stays fully frozen.
# peft matches a str target_modules with re.fullmatch, so the pattern
# must account for the full dotted path (e.g.
# "model.language_model.layers.0.self_attn.q_proj") - the fix vs the
# first attempt (which used re.fullmatch and failed because it didn't
# allow for the "model." prefix) is the leading ".*".
LORA_TARGET_MODULES = r".*language_model.*self_attn\.(q_proj|k_proj|v_proj|o_proj)$"

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 1e-4
# Same epoch count as the completed qwen2_rider1_7ep.py run, for a fair
# architecture-vs-architecture comparison.
MAX_EPOCHS = 7
EARLY_STOP_PATIENCE = 4
WARMUP_RATIO = 0.03

random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Label/clip-bounds/telemetry logic - identical to qwen2_rider1.py so the
# train/val split and ground truth are directly comparable across models.
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
    """Single representative frame at the temporal midpoint of the clip -
    PaliGemma's SigLIP vision tower takes exactly one image per example."""
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
        # fall back to the first frame in range if the midpoint seek fails
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

        # PaliGemmaProcessor's suffix= mechanism builds input_ids/labels for
        # us (prompt tokens masked with -100 automatically), analogous to
        # the manual prompt_len masking done in qwen2_rider1.py.
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

    # ---------------- Verify + report frozen vs trainable modules ----------
    # Requirement: report the ACTUAL verified target modules and trainable
    # param count, not an assumption. Walk the peft-wrapped model and list
    # every module that actually received a LoRA adapter, plus confirm the
    # vision tower has zero trainable params.
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
    print("Sample of trainable (LoRA-adapted) module names:")
    for n in lm_trainable[:8]:
        print(" ", n)
    with open(os.path.join(OUTPUT_DIR, "trainable_modules_verified.json"), "w") as f:
        json.dump({
            "trainable_module_names": trainable_module_names,
            "n_language_model_modules": len(lm_trainable),
            "n_vision_tower_modules": len(vision_trainable),
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_pct": 100 * trainable_params / total_params,
        }, f, indent=2)

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
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params_list, lr=LEARNING_RATE)

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

    print(f"Training complete. Best adapter at: {best_ckpt_dir}")


if __name__ == "__main__":
    main()
