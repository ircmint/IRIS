"""
QLoRA fine-tune Qwen2.5-VL-3B-Instruct on the motor dataset (40 gold events,
34 usable after dropping uncertain rows — all used for training, no held-out val).

Config matches Custom_Data rider runs:
  r=16 / alpha=32 / dropout=0.05 / language model attention+MLP only
  3B model fits in bf16 (~7GB) on RTX 2080 Ti — no 4-bit needed.

Run:
    source $SCRATCH_ROOT/envs/qwen2vl/bin/activate
    cd $SCRATCH_ROOT
    PYTHONUNBUFFERED=1 HF_HOME=$SCRATCH_ROOT/hf_cache \
        CUDA_VISIBLE_DEVICES=1 python motor_qwen25_train.py 2>&1 | tee motor_qwen25_train.log
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

import os, json, gc, traceback, random
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from PIL import Image
from decord import VideoReader, cpu

# ============================================================
# PATHS
# ============================================================
MODEL_PATH  = os.path.join(SCRATCH_ROOT, "Qwen2.5-VL-3B-Instruct")
DATA_ROOT   = SCRATCH_ROOT
CLIPS_DIR   = os.path.join(DATA_ROOT, "upload_clips")
CSV_PATH    = os.path.join(DATA_ROOT, "gold_candidates.csv")
OUTPUT_DIR  = os.path.join(SCRATCH_ROOT, "runs/motor_qwen25_qlora")
CACHE_DIR   = os.path.join(OUTPUT_DIR, "tensor_cache")

# ============================================================
# CONFIG — match Custom_Data rider runs
# ============================================================
NUM_FRAMES       = 4       # 3B model is small enough for 4 frames
RESIZE_MAX       = 448
LORA_R           = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05
LORA_TARGET      = r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
BATCH_SIZE       = 1
GRAD_ACCUM_STEPS = 8
LEARNING_RATE    = 2e-4
MAX_EPOCHS       = 7
WARMUP_RATIO     = 0.03
SEED             = 42

random.seed(SEED)
torch.manual_seed(SEED)


# ---- Label helpers ----
def get_label(row):
    return "evasive" if str(row.get("decision", "")).strip().lower() == "confirm" else "not_evasive"

def get_reasoning(row):
    v = row.get("notes")
    return str(v).strip() if pd.notna(v) and str(v).strip() else "No additional notes provided."

def get_clip_path(row):
    cid = row["clip_id"]
    start = row.get("adjusted_start") if pd.notna(row.get("adjusted_start")) else row.get("start_time")
    end   = row.get("adjusted_end")   if pd.notna(row.get("adjusted_end"))   else row.get("end_time")
    start, end = float(start), float(end)
    if end - start < 1.0:
        c = (start + end) / 2.0
        start, end = max(c - 0.5, 0.0), c + 0.5
    return os.path.join(CLIPS_DIR, f"{cid}__{start:.3f}_{end:.3f}.mp4")

def build_taxonomy(df):
    df = df[df["decision"].str.lower().isin(["confirm", "reject"])].copy().reset_index(drop=True)
    df["_label"] = df.apply(get_label, axis=1)
    return df, sorted(df["_label"].unique().tolist())

SCHEMA_TMPL = (
    "You are analyzing a short dashcam clip from a two-wheeler for evasive-maneuver "
    "classification. Given the frames and telemetry below, respond ONLY with a JSON object:\n"
    '{{"label": "<one of: {labels}>", "confidence": <float 0-1>, '
    '"reasoning": "<short justification grounded in what is visible/measured>"}}\n'
    "No text outside the JSON."
)

def build_prompt_messages(row, taxonomy):
    labels = ", ".join(taxonomy)
    schema = SCHEMA_TMPL.format(labels=labels)
    telemetry = (
        f"Telemetry: peak jerk={row.get('peak_abs_z_jerk','NA')}, "
        f"peak lateral accel={row.get('peak_abs_z_az','NA')}, "
        f"duration={row.get('duration','NA')}s, "
        f"IMU decision={row.get('decision','NA')}."
    )
    content = [{"type": "video", "video": get_clip_path(row), "nframes": NUM_FRAMES, "max_pixels": RESIZE_MAX*RESIZE_MAX}]
    content.append({"type": "text", "text": f"{schema}\n{telemetry}"})
    return [{"role": "user", "content": content}]

def build_target(row, taxonomy):
    return json.dumps({"label": get_label(row), "confidence": 1.0, "reasoning": get_reasoning(row)})


class MotorQwenDataset(Dataset):
    def __init__(self, df, processor, taxonomy, cache_dir, split_name):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.taxonomy = taxonomy
        self.cache_dir = cache_dir
        self.split_name = split_name
        os.makedirs(cache_dir, exist_ok=True)
        self._mem = {}

    def __len__(self): return len(self.df)

    def _compute(self, row):
        messages = build_prompt_messages(row, self.taxonomy)
        prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        prompt_enc = self.processor(
            text=[prompt_text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        target = build_target(row, self.taxonomy)
        full_text = prompt_text + target + self.processor.tokenizer.eos_token
        full_enc = self.processor(
            text=[full_text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        )

        input_ids = full_enc["input_ids"][0]
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        item = {
            "input_ids": input_ids,
            "attention_mask": full_enc["attention_mask"][0],
            "labels": labels,
        }
        if "pixel_values_videos" in full_enc:
            item["pixel_values_videos"] = full_enc["pixel_values_videos"]
        if "pixel_values" in full_enc:
            item["pixel_values"] = full_enc["pixel_values"]
        if "image_grid_thw" in full_enc:
            item["image_grid_thw"] = full_enc["image_grid_thw"]
        if "video_grid_thw" in full_enc:
            item["video_grid_thw"] = full_enc["video_grid_thw"]
        return item

    def __getitem__(self, i):
        if i in self._mem: return self._mem[i]
        cache_path = os.path.join(self.cache_dir, f"{self.split_name}_{i}.pt")
        if os.path.exists(cache_path):
            item = torch.load(cache_path)
            self._mem[i] = item
            return item
        row = self.df.iloc[i]
        item = self._compute(row)
        torch.save(item, cache_path)
        self._mem[i] = item
        return item


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

def collate_fn(batch): return batch[0]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    print("Loading model in bf16 (3B fits without quantization)...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True
    )
    model.config.use_cache = False
    print(f"GPU after load: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    for p in model.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- Data ----
    df = pd.read_csv(CSV_PATH)
    df, taxonomy = build_taxonomy(df)
    print(f"Training on {len(df)} events. Labels: {taxonomy}")
    with open(os.path.join(OUTPUT_DIR, "taxonomy.json"), "w") as f:
        json.dump(taxonomy, f, indent=2)

    train_ds = MotorQwenDataset(df, processor, taxonomy, CACHE_DIR, "train")

    label_counts = df["_label"].value_counts()
    class_weight = {lbl: 1.0/cnt for lbl, cnt in label_counts.items()}
    sample_weights = df["_label"].map(class_weight).values
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn)

    # ---- Optimizer ----
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE)
    steps_per_epoch = max(len(train_loader) // GRAD_ACCUM_STEPS, 1)
    total_steps = steps_per_epoch * MAX_EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=max(int(total_steps*WARMUP_RATIO), 1),
        num_training_steps=total_steps,
    )

    best_ckpt = os.path.join(OUTPUT_DIR, "best_adapter")
    best_loss = float("inf")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss, n_ok = 0.0, 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            try:
                batch = to_device(batch, model.device)
                outputs = model(**batch)
                loss = outputs.loss / GRAD_ACCUM_STEPS
                loss.backward()
                running_loss += loss.item() * GRAD_ACCUM_STEPS
                n_ok += 1
                del outputs, loss, batch
            except Exception as e:
                print(f"[epoch {epoch}] step {step} SKIPPED: {e}")
                traceback.print_exc()
                optimizer.zero_grad()
                gc.collect(); torch.cuda.empty_cache()
                continue

            if step % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                torch.cuda.empty_cache()

            if step % 10 == 0:
                print(f"[epoch {epoch}] step {step}/{len(train_loader)} loss={running_loss/max(n_ok,1):.4f}", flush=True)

        train_loss = running_loss / max(n_ok, 1)
        print(f"== Epoch {epoch}/{MAX_EPOCHS} train_loss={train_loss:.4f} ==", flush=True)

        model.save_pretrained(os.path.join(OUTPUT_DIR, f"checkpoint_epoch{epoch}"))
        processor.save_pretrained(os.path.join(OUTPUT_DIR, f"checkpoint_epoch{epoch}"))
        if train_loss < best_loss:
            best_loss = train_loss
            model.save_pretrained(best_ckpt)
            processor.save_pretrained(best_ckpt)
            with open(os.path.join(best_ckpt, "taxonomy.json"), "w") as f:
                json.dump(taxonomy, f, indent=2)
            print(f"  -> new best loss={train_loss:.4f}, saved to {best_ckpt}", flush=True)

    print(f"\nDone. Best adapter: {best_ckpt}", flush=True)

if __name__ == "__main__":
    main()
