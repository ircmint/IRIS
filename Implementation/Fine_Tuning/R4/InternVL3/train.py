"""
QLoRA fine-tune InternVL3-8B-hf for fixed-schema JSON evasive-event
classification. Training loop structure ported directly from the working
Qwen2-VL reference (FineTune.py): manual loop (not HF Trainer) so we can use
a WeightedRandomSampler for class imbalance and per-step error recovery,
gradient accumulation, cosine schedule, and early stopping on val loss.

*** USE_4BIT MUST STAY True unless you're on an 80GB-class GPU ***
InternVL3-8B in bf16 is ~16GB of weights alone.

Run:
    conda activate $SCRATCH_ROOT/envs/internvl
    python train.py
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
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from labels import build_taxonomy, save_taxonomy
from dataset import EvasiveEventDataset

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = os.path.join(SCRATCH_ROOT, "models/internvl3-8b-hf")
VIDEO_DIR = os.path.join(HOME_ROOT, "IRASTE")
VIDEO_FILENAME = "GX019940.MP4"
CSV_PATH = os.path.join(HOME_ROOT, "IRASTE/DAY_2/candidate_events_updated.csv")
OUTPUT_DIR = os.path.join(SCRATCH_ROOT, "runs/internvl3-8b-qlora-evasive")

NUM_FRAMES = 2   # dropped from 4 - this GPU is ~11GB and was OOMing on lm_head
                 # dequant alone; fewer frames means fewer vision tokens feeding
                 # the language model, which reduces hidden_state/activation size
                 # throughout the whole forward pass, not just at the vision tower
MIN_CLIP_DURATION = 1.0
CACHE_DIR = os.path.join(OUTPUT_DIR, f"tensor_cache_f{NUM_FRAMES}")

USE_4BIT = True   # REQUIRED unless you have an 80GB-class GPU - do not set False

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.08
# IMPORTANT: the vision tower uses the SAME q_proj/k_proj/v_proj/o_proj naming
# as the language model's attention layers (confirmed via named_modules() -
# model.vision_tower.encoder.layer.N.attention.q_proj exists alongside
# model.language_model.layers.N.self_attn.q_proj). A plain suffix-matched
# target_modules list would silently LoRA-tune the vision tower too, which
# directly contradicts "freeze the vision encoder." Using a regex string
# instead of a list forces peft to match on the FULL module path.
LORA_TARGET_MODULES = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"

BATCH_SIZE = 1                # keep at 1 - dataset returns pre-tokenized single items
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-4
MAX_EPOCHS = 5
EARLY_STOP_PATIENCE = 3
WARMUP_RATIO = 0.03
VAL_SPLIT = 0.15
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


def to_device_batch(batch, device):
    """pixel_values is already [num_frames, C, H, W] - the model expects this
    shape directly, no batch dim. input_ids/attention_mask/labels are 1D
    [seq_len] from __getitem__ and need a batch dim of 1 added."""
    out = {"pixel_values": batch["pixel_values"].to(device)}
    for k in ("input_ids", "attention_mask", "labels"):
        v = batch[k]
        if v.dim() == 1:
            v = v.unsqueeze(0)
        out[k] = v.to(device)
    return out


def collate_fn(batch):
    return batch[0]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("Loading model in 4-bit (QLoRA)..." if USE_4BIT else "Loading model in bf16...")
    quant_config = None
    if USE_4BIT:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            # lm_head is skipped too, not just vision - InternVL3-8B's vocab is
            # ~150K, so lm_head is a huge matrix. Quantizing it means
            # bitsandbytes has to dequantize it into a fresh ~1GB temp buffer
            # on EVERY forward pass - that transient allocation, not the LoRA
            # layers, is what's OOMing on this GPU. Keeping it unquantized
            # costs ~1GB of static VRAM but removes the repeated spike.
            llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head"],
        )

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        quantization_config=quant_config,
        dtype=torch.float16,
        device_map="auto",
    )

    if USE_4BIT:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        print(f"GPU memory after quantized load: "
              f"{torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")

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

    # safety net: fail loudly if the vision tower somehow got LoRA'd anyway,
    # rather than silently training on a broken assumption
    leaked = [n for n, p in model.named_parameters()
              if p.requires_grad and ("vision_tower" in n or "multi_modal_projector" in n)]
    if leaked:
        raise RuntimeError(
            f"LoRA leaked into the vision tower/projector ({len(leaked)} params, "
            f"e.g. {leaked[:3]}) - fix LORA_TARGET_MODULES before training."
        )

    # ---------------- Data ----------------
    df = pd.read_csv(CSV_PATH)
    df, taxonomy = build_taxonomy(df)
    print(f"Total usable events: {len(df)}")
    print("Label distribution:\n", df["_label"].value_counts())
    print("Taxonomy:", taxonomy)
    save_taxonomy(taxonomy, os.path.join(OUTPUT_DIR, "taxonomy.json"))

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
    train_ds = EvasiveEventDataset(
        train_df, video_path, processor, taxonomy, CACHE_DIR, "train",
        num_frames=NUM_FRAMES, min_clip_duration=MIN_CLIP_DURATION,
    )
    val_ds = EvasiveEventDataset(
        val_df, video_path, processor, taxonomy, CACHE_DIR, "val",
        num_frames=NUM_FRAMES, min_clip_duration=MIN_CLIP_DURATION,
    )

    # class-imbalance fix - without this the majority label dominates every
    # epoch and minority-class F1 stays near zero (this is what happened to
    # the PaliGemma run: predictions collapsed onto one class)
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
                batch = to_device_batch(batch, model.device)
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
                # the traceback holds refs to every local var in every frame
                # up to the failure point - including all intermediate
                # activation tensors computed before the OOM. Without
                # clearing batch and the exception itself, empty_cache()
                # can't reclaim that memory, which is why every subsequent
                # step was failing identically.
                if "batch" in dir():
                    del batch
                del e
                gc.collect()
                torch.cuda.empty_cache()
                continue

            if step % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                # every example has a different token length (notes text
                # varies per event), so the CUDA allocator carves a
                # differently-sized block every step. Over many steps this
                # fragments free memory into pieces too small to satisfy new
                # allocations, even when total free memory looks sufficient -
                # a periodic defrag here is cheap insurance against that.
                torch.cuda.empty_cache()

            if step % 20 == 0:
                print(f"[epoch {epoch}] step {step}/{len(train_loader)} "
                      f"running_loss={running_loss / max(n_ok, 1):.4f}")

        train_loss = running_loss / max(n_ok, 1)

        model.eval()
        val_loss_total, n_val_ok = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                try:
                    batch = to_device_batch(batch, model.device)
                    outputs = model(**batch)
                    val_loss_total += outputs.loss.item()
                    n_val_ok += 1
                    del outputs, batch
                except Exception as e:
                    print(f"[epoch {epoch}] val step SKIPPED due to error: {e}")
                    if "batch" in dir():
                        del batch
                    del e
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
            save_taxonomy(taxonomy, os.path.join(best_ckpt_dir, "taxonomy.json"))
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