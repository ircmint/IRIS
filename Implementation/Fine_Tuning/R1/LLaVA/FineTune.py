"""
QLoRA fine-tuning of LLaVA-NeXT-Video for fixed-schema JSON evasive-event
classification: {"label": ..., "confidence": ..., "reasoning": ...}

Vision tower is kept FROZEN (perception is not the failure mode here) -
only the language model (+ optionally the multi-modal projector) gets LoRA
adapters. This is instruction-following / schema-calibration fine-tuning,
not perception fine-tuning.
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
import gc
import json
import random
import hashlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import cv2

# ============================== CONFIG ==============================
MODEL_ID = "llava-hf/LLaVA-NeXT-Video-7B-hf"

BASE_DIR = SCRATCH_ROOT
MODEL_CACHE_DIR = f"{BASE_DIR}/models"
CSV_PATH       = os.path.join(HOME_ROOT, "IRASTE/DAY_2/candidate_events_updated.csv")
VIDEO_DIR      = os.path.join(HOME_ROOT, "IRASTE")
VIDEO_FILENAME = "GX019940.MP4"
OUTPUT_DIR = f"{BASE_DIR}/llava_next_video_out"
BEST_ADAPTER_DIR = f"{OUTPUT_DIR}/best_adapter"

USE_4BIT = True  # standing decision: QLoRA everywhere, never plain LoRA on a 7B model

FRAMES_PER_EVENT = 8          # LLaVA-NeXT-Video expects a real frame sequence
RESIZE_MAX_SIDE = 448         # long side cap in pixels, learned the hard way on Qwen2-VL
MIN_EVENT_DURATION = 1.0      # seconds, padding applied symmetrically if event shorter
# 8 frames alone expanded the prompt to 1307 tokens in practice (video placeholder
# expansion, not text) - budget generously above that for prompt + JSON response.
MAX_SEQ_LENGTH = 2048

LORA_R = 16
LORA_ALPHA = 32               # 2x rank, standard scaling convention
LORA_DROPOUT = 0.05
# Matched by FULL module path (regex), not bare layer name. vision_tower's
# attention blocks are ALSO named q_proj/k_proj/v_proj internally, so a
# bare-name list like ["q_proj", ...] matches inside the vision tower too
# and silently breaks the "frozen vision encoder" requirement. Anchoring
# to ".language_model." makes that impossible regardless of naming overlap
# elsewhere in the model.
LORA_TARGET_MODULES = r".*\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"

EPOCHS = 10
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
EARLY_STOPPING_PATIENCE = 3
VAL_FRACTION = 0.15
SEED = 42

# cache dir encodes its own config -> a stale cache can never silently get reused.
# PREPROCESS_VERSION bumps whenever __getitem__'s tensor format/processing logic
# changes (not just frame count/resolution), so a code fix always gets fresh tensors.
PREPROCESS_VERSION = 5  # v5: removed peak_confidence from the visible prompt - it was also the JSON target's confidence value, letting the model learn to copy it instead of estimate it, which was the real cause of the near-instant loss collapse
_cache_sig = hashlib.md5(
    f"{FRAMES_PER_EVENT}_{RESIZE_MAX_SIDE}_{MIN_EVENT_DURATION}_{PREPROCESS_VERSION}".encode()
).hexdigest()[:8]
TENSOR_CACHE_DIR = f"{OUTPUT_DIR}/tensor_cache_{_cache_sig}"

LABELS = ["not_evasive", "swerve", "acceleration", "lane_change", "other"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

SYSTEM_PROMPT = (
    "You are analyzing a short dashcam video clip from a two-wheeler. "
    "Classify the event into exactly one of: not_evasive, swerve, acceleration, "
    "lane_change, other. Respond with ONLY a JSON object of the form "
    '{"label": <one of the classes>, "confidence": <float 0-1>, "reasoning": <short justification>}. '
    "No other text."
)

os.environ.setdefault("HF_HOME", f"{MODEL_CACHE_DIR}/hf_cache")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TENSOR_CACHE_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================== TAXONOMY ==============================
def classify_from_notes(notes: str) -> str:
    """Keyword-rule mapping from free-text notes to the 4(+1)-class taxonomy.
    Adopted from the Qwen2-VL reference script against the same CSV."""
    if not isinstance(notes, str) or not notes.strip():
        return "other"
    t = notes.lower()
    if any(k in t for k in ["no evasive", "not evasive", "false positive", "normal riding", "no maneuver"]):
        return "not_evasive"
    if any(k in t for k in ["swerve", "swerved", "swerving", "dodge", "dodged"]):
        return "swerve"
    if any(k in t for k in ["brake", "braking", "accelerat", "sudden stop", "speed up", "throttle"]):
        return "acceleration"
    if any(k in t for k in ["lane change", "lane shift", "changed lane", "merge"]):
        return "lane_change"
    return "other"


# ============================== DATA CLEANING ==============================
def clean_events_df(df: pd.DataFrame) -> pd.DataFrame:
    """Some rows in the 10-day-collected CSV don't have adjusted_start/end
    computed (NaN). Fall back to the raw start_time/end_time for those, and
    drop anything still unusable so a single bad row can't crash training
    partway through an epoch."""
    df = df.copy()
    if "start_time" in df.columns:
        df["adjusted_start"] = df["adjusted_start"].fillna(df["start_time"])
    if "end_time" in df.columns:
        df["adjusted_end"] = df["adjusted_end"].fillna(df["end_time"])

    before = len(df)
    df = df.dropna(subset=["adjusted_start", "adjusted_end"])
    df = df[df["adjusted_end"] > df["adjusted_start"]]
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped}/{before} rows with unusable start/end timing (NaN or non-positive duration)")
    return df.reset_index(drop=True)


# ============================== TELEMETRY PROMPT ==============================
def build_telemetry_summary(row: pd.Series) -> str:
    # NOTE: peak_confidence is deliberately NOT included here even though it's
    # a real column. It's used as the JSON target's "confidence" field - if it
    # were also shown verbatim in the prompt, the model could just copy that
    # number back out instead of learning to estimate confidence, which was
    # silently collapsing the loss (the model doesn't need video/reasoning to
    # nail a value it can already see in its own context).
    return (
        f"Telemetry summary: duration={row['duration']:.2f}s, "
        f"peak_abs_z_jerk={row['peak_abs_z_jerk']:.3f}, "
        f"peak_abs_z_az={row['peak_abs_z_az']:.3f}."
    )


# ============================== FRAME SAMPLING ==============================
def sample_frames(video_path: str, start: float, end: float, n_frames: int, max_side: int) -> list:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_f = max(0, int(start * fps))
    end_f = int(end * fps)
    if end_f <= start_f:
        end_f = start_f + 1
    idxs = np.linspace(start_f, end_f, n_frames, dtype=int)

    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            frame = frames[-1] if frames else np.zeros((max_side, max_side, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            scale = max_side / max(h, w)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        frames.append(frame)
    cap.release()
    return frames


# ============================== DATASET ==============================
class EvasiveEventDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor: LlavaNextVideoProcessor):
        self.df = df.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        cache_path = os.path.join(TENSOR_CACHE_DIR, f"{row['clip_id']}.pt")
        if os.path.exists(cache_path):
            return torch.load(cache_path)

        start = row["adjusted_start"]
        end = row["adjusted_end"]
        if end - start < MIN_EVENT_DURATION:
            pad = (MIN_EVENT_DURATION - (end - start)) / 2
            start, end = max(0, start - pad), end + pad

        video_path = os.path.join(VIDEO_DIR, f"{row['clip_id']}.mp4")
        frames = sample_frames(video_path, start, end, FRAMES_PER_EVENT, RESIZE_MAX_SIDE)

        label = classify_from_notes(row.get("notes", ""))
        telemetry = build_telemetry_summary(row)

        system_msg = {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
        user_msg = {"role": "user", "content": [
            {"type": "video"},
            {"type": "text", "text": telemetry},
        ]}

        # Reasoning target MUST come from the actual per-event notes, not a
        # fixed template. A template repeated for every sample of a given
        # class (e.g. "Detected {label} pattern...") gives the model only
        # 4-5 distinct strings to memorize across the whole dataset - loss
        # collapses to ~0 in the first epoch because there's nothing to
        # actually learn, and the resulting adapter can't produce real
        # per-event reasoning at inference time. Fall back to a generic
        # (but non-identical, telemetry-grounded) sentence only when notes
        # are genuinely empty, so at minimum reasoning varies per sample.
        notes_text = row.get("notes", "")
        if isinstance(notes_text, str) and notes_text.strip():
            reasoning_text = notes_text.strip()
        else:
            reasoning_text = (
                f"Detected a {label} event with peak jerk z-score "
                f"{row['peak_abs_z_jerk']:.2f} and peak acceleration z-score {row['peak_abs_z_az']:.2f}."
            )
        confidence_val = float(row["peak_confidence"]) if pd.notna(row.get("peak_confidence")) else 0.9
        target = json.dumps({"label": label, "confidence": round(confidence_val, 2), "reasoning": reasoning_text})
        assistant_msg = {"role": "assistant", "content": [{"type": "text", "text": target}]}

        # Tokenize prompt-only and prompt+response as TWO SEPARATE calls, each
        # with the video attached. The processor expands the <video> tag into
        # one placeholder token per vision patch (hundreds+ tokens depending on
        # frame count/resolution) - that expansion must happen consistently in
        # both calls, which is why we don't just tokenize the target string on
        # its own: a standalone target tensor has no relation to the actual
        # (expanded) input sequence length and produces a shape mismatch at
        # the loss computation (logits seq_len != labels seq_len).
        prompt_text = self.processor.apply_chat_template(
            [system_msg, user_msg], add_generation_prompt=True, tokenize=False
        )
        full_text = self.processor.apply_chat_template(
            [system_msg, user_msg, assistant_msg], add_generation_prompt=False, tokenize=False
        )

        prompt_inputs = self.processor(
            text=prompt_text, videos=[frames], return_tensors="pt",
            truncation=True, max_length=MAX_SEQ_LENGTH,
        )
        full_inputs = self.processor(
            text=full_text, videos=[frames], return_tensors="pt",
            truncation=True, max_length=MAX_SEQ_LENGTH,
        )

        prompt_len = prompt_inputs["input_ids"].shape[1]
        input_ids = full_inputs["input_ids"][0]
        attention_mask = full_inputs["attention_mask"][0]

        labels = input_ids.clone()
        labels[:min(prompt_len, labels.size(0))] = -100  # mask prompt+video tokens, keep only the response

        if prompt_len >= input_ids.size(0):
            print(f"WARNING: clip {row['clip_id']} - prompt consumed the entire {MAX_SEQ_LENGTH}-token budget, "
                  f"response was truncated away entirely (all labels masked). Consider raising MAX_SEQ_LENGTH.")

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values_videos": full_inputs["pixel_values_videos"][0],
            "labels": labels,
            "label_str": label,
        }
        torch.save(item, cache_path)
        return item


def collate_fn(batch, pad_token_id):
    out = {}
    for key in ["input_ids", "attention_mask", "labels"]:
        seqs = [b[key] for b in batch]
        maxlen = max(s.size(0) for s in seqs)
        pad_val = pad_token_id if key != "labels" else -100
        padded = torch.full((len(seqs), maxlen), pad_val, dtype=seqs[0].dtype)
        for i, s in enumerate(seqs):
            padded[i, : s.size(0)] = s
        out[key] = padded
    out["pixel_values_videos"] = torch.stack([b["pixel_values_videos"] for b in batch])
    return out


# ============================== MODEL ==============================
def load_model_and_processor():
    processor = LlavaNextVideoProcessor.from_pretrained(MODEL_ID, cache_dir=MODEL_CACHE_DIR)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=USE_4BIT,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir=MODEL_CACHE_DIR,
        quantization_config=bnb_config if USE_4BIT else None,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )

    print(f"GPU memory after quantized load: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")

    # Freeze the vision tower explicitly - perception is not what we're fixing
    for name, param in model.named_parameters():
        if "vision_tower" in name:
            param.requires_grad_(False)

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Belt-and-suspenders: confirm no vision_tower params ended up trainable via LoRA
    trainable_vision = [n for n, p in model.named_parameters() if p.requires_grad and "vision_tower" in n]
    assert not trainable_vision, f"Vision tower params leaked into trainable set: {trainable_vision}"

    model.print_trainable_parameters()
    return model, processor


# ============================== TRAIN LOOP ==============================
def evaluate_val_loss(model, val_loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "label_str"}
            out = model(**batch)
            total_loss += out.loss.item()
            n += 1
    model.train()
    return total_loss / max(n, 1)


def main():
    df = pd.read_csv(CSV_PATH)
    df = clean_events_df(df)
    df["label"] = df["notes"].apply(classify_from_notes)

    val_parts = [g.sample(frac=VAL_FRACTION, random_state=SEED) for _, g in df.groupby("label")]
    val_df = pd.concat(val_parts)
    train_df = df.drop(val_df.index)

    model, processor = load_model_and_processor()
    device = next(model.parameters()).device
    pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    train_ds = EvasiveEventDataset(train_df, processor)
    val_ds = EvasiveEventDataset(val_df, processor)

    class_counts = train_df["label"].value_counts().to_dict()
    weights = train_df["label"].map(lambda l: 1.0 / class_counts[l]).values
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE
    )

    best_val_loss = float("inf")
    patience_left = EARLY_STOPPING_PATIENCE
    step = 0
    model.train()

    for epoch in range(EPOCHS):
        for batch in train_loader:
            try:
                batch_dev = {k: v.to(device) for k, v in batch.items() if k != "label_str"}
                out = model(**batch_dev)
                loss = out.loss / GRAD_ACCUM_STEPS
                loss.backward()
                step += 1
                if step % GRAD_ACCUM_STEPS == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step} loss {out.loss.item():.4f}")
            except torch.cuda.OutOfMemoryError:
                print(f"OOM at step {step}, skipping batch and clearing cache")
                optimizer.zero_grad(set_to_none=True)
                del batch_dev
                gc.collect()
                torch.cuda.empty_cache()
                continue

        val_loss = evaluate_val_loss(model, val_loader, device)
        print(f"epoch {epoch} val_loss {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_left = EARLY_STOPPING_PATIENCE
            model.save_pretrained(BEST_ADAPTER_DIR)
            processor.save_pretrained(BEST_ADAPTER_DIR)
            print(f"New best val_loss {val_loss:.4f}, adapter saved to {BEST_ADAPTER_DIR}")
        else:
            patience_left -= 1
            print(f"No improvement, patience_left={patience_left}")
            if patience_left <= 0:
                print("Early stopping triggered")
                break

    print("Training complete.")


if __name__ == "__main__":
    main()