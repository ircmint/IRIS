
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
# -*- coding: utf-8 -*-
"""
InternVL2-8B zero-shot evaluator for Mobile iRASTE evasive-event classification.

DESIGNED TO RUN IN AN ISOLATED CONDA ENV PINNED TO transformers==4.37.2
--------------------------------------------------------------------------
Why: InternVL2's `trust_remote_code=True` files (modeling_internlm2.py,
modeling_internvl_chat.py) assume the legacy tuple-of-tuples KV-cache format
(`past_key_values[0][0].shape[2]`). Modern transformers (4.40+) replaced this
with a `DynamicCache` object that isn't subscriptable that way, which is what
caused the whole chain of errors previously (TypeError: 'DynamicCache' object
is not subscriptable / got multiple values for 'expand_size' / got multiple
values for 'use_cache').

Patching individual methods at runtime doesn't fully work because the tuple
assumption is baked into forward() and the attention blocks too, not just
prepare_inputs_for_generation. The only clean fix is running in an env where
transformers actually matches what this remote code expects. Set that up
first:

    conda create -n internvl2_legacy python=3.10 -y
    conda activate internvl2_legacy
    pip install transformers==4.37.2 --break-system-packages
    pip install torch torchvision --break-system-packages   # match nvidia-smi CUDA version
    pip install accelerate sentencepiece einops timm pillow bitsandbytes --break-system-packages

    # IMPORTANT: also make sure accelerate is reasonably current -- older
    # accelerate releases call `model.to(device)` inside dispatch_model()
    # whenever device_map resolves to a single device, and that call is
    # blocked for 8-bit/4-bit bitsandbytes models. Newer accelerate checks
    # for an hf_quantizer before doing that. This is independent of the
    # transformers==4.37.2 pin above (accelerate's device-mapping code is
    # decoupled from InternVL2's KV-cache remote code), so upgrading it
    # freely will NOT reintroduce the DynamicCache error:
    pip install -U "accelerate>=0.28.0" --break-system-packages

Then run this script from inside that env. No generation-cache monkey-patches
are needed here because the pinned version already matches the remote code.
--------------------------------------------------------------------------
"""
import os
import sys

# ---------------------------------------------------------------------------
# Cache dir override -> /home has a small quota (25GB) and is often nearly
# full; point all HF/pip/XDG caches at /ssd_scratch instead so model-loading
# never writes back into /home.
# ---------------------------------------------------------------------------
SCRATCH_CACHE_ROOT = SCRATCH_ROOT
os.environ["HF_HOME"] = os.path.join(SCRATCH_CACHE_ROOT, "hf_cache")
os.environ["HF_MODULES_CACHE"] = os.path.join(SCRATCH_CACHE_ROOT, "hf_cache", "modules")
os.environ["XDG_CACHE_HOME"] = os.path.join(SCRATCH_CACHE_ROOT, "cache")
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["HF_MODULES_CACHE"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import json
import torch
import transformers
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# ---------------------------------------------------------------------------
# Version guard -- fail loudly and early if the env isn't set up right,
# instead of failing deep inside a generate() call an hour into the run.
# ---------------------------------------------------------------------------
EXPECTED_TRANSFORMERS = "4.37.2"
if transformers.__version__ != EXPECTED_TRANSFORMERS:
    print(f"WARNING: this script expects transformers=={EXPECTED_TRANSFORMERS}, "
          f"found {transformers.__version__}. If InternVL2-8B loading/generation "
          f"fails with a DynamicCache/subscriptable error, this version mismatch "
          f"is almost certainly why. Activate the pinned conda env (see header "
          f"comment) before running.")

# ======================================================================
# Path Configuration
# ======================================================================
def resolve_model_path(base_path: str, fallback: str) -> str:
    if not os.path.exists(base_path):
        return fallback
    if os.path.exists(os.path.join(base_path, "config.json")):
        return base_path
    snapshots_dir = os.path.join(base_path, "snapshots")
    if os.path.exists(snapshots_dir) and os.path.isdir(snapshots_dir):
        try:
            subdirs = [os.path.join(snapshots_dir, d) for d in os.listdir(snapshots_dir)]
            subdirs = [d for d in subdirs if os.path.isdir(d)]
            if subdirs:
                subdirs.sort(key=os.path.getmtime, reverse=True)
                for subdir in subdirs:
                    if os.path.exists(os.path.join(subdir, "config.json")):
                        return subdir
        except Exception:
            pass
    for root, dirs, files in os.walk(base_path):
        if "config.json" in files:
            return root
    return fallback

# Model weights live on fast local scratch (node-local, not visible from
# ada/login node -- this path only resolves correctly while your SLURM job
# is running on the gnode that has them).
INTERNVL_PATH = resolve_model_path(
    os.path.join(SCRATCH_ROOT, "InternVL2-8B"),
    "OpenGVLab/InternVL2-8B"
)
print(f"Resolved INTERNVL_PATH to: {INTERNVL_PATH}")

# Video/labels stay on /home2, which IS mounted on compute nodes (unlike
# /share1 / /share3, which are login-node-only).
VIDEO_PATH = os.path.join(HOME_ROOT, "IRASTE/GX019940.mp4")
OUTPUT_DIR = os.path.join(HOME_ROOT, "IRASTE/DAY_3/local_models_outputs")
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp_frames")
CSV_PATH = os.path.join(HOME_ROOT, "IRASTE/DAY_2/candidate_events_updated.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Set to an int for rapid testing, None for a full run
LIMIT_EVENTS = None

# GPU memory: InternVL2-8B is ~16GB in fp16, which will NOT fit on a single
# 11GB GPU (GTX 1080 Ti / RTX 2080 Ti on Ada). Default to 8-bit quantization
# so it fits on --gres=gpu:1. If you instead request --gres=gpu:2, set this
# to False and device_map="auto" will shard the model across both GPUs.
USE_8BIT_QUANTIZATION = True

# ----------------------------------------------------------------------
# Zero-Shot Prompts (identical taxonomy/format to the multi-model script,
# so results are directly comparable in the ablation table)
# ----------------------------------------------------------------------
PROMPTS = {
    "video_only": """Identify the motorcycle rider's evasive maneuver. Choose from:
[acceleration, deceleration, braking, emergency_swerve, lane_change, zigzag, none].
Respond ONLY with a JSON block:
{"evasive_action": "<class>", "confidence": 0.0-1.0, "reasoning": "brief justification"}""",

    "raw_telemetry": """Identify the motorcycle rider's evasive maneuver.
Here is the raw telemetry sequence during this event (sampled at 10Hz):
{raw_telemetry_data}
Choose from: [acceleration, deceleration, braking, emergency_swerve, lane_change, zigzag, none].
Respond ONLY with a JSON block:
{{"evasive_action": "<class>", "confidence": 0.0-1.0, "reasoning": "brief justification based on vision and raw telemetry"}}""",

    "summarized_telemetry": """Identify the motorcycle rider's evasive maneuver.
Here is the summarized telemetry context for this event:
- Peak Vertical acceleration spike: {peak_z_az:.2f} Z-Score units
- Peak Lateral steering jerk: {peak_z_jerk:.2f} Z-Score units
- Vehicle speed: {speed_kmh:.1f} km/h
- Duration: {duration:.2f} seconds

Examples:
- Accel Spike = 3.2, Steering Jerk = 1.2 -> braking
- Accel Spike = 0.5, Steering Jerk = 4.2 -> emergency_swerve
- Accel Spike = 0.4, Steering Jerk = 0.6 -> none
- Accel Spike = 0.6, Steering Jerk = 1.8 -> lane_change

Choose from: [acceleration, deceleration, braking, emergency_swerve, lane_change, zigzag, none].
Respond ONLY with a JSON block:
{{"evasive_action": "<class>", "confidence": 0.0-1.0, "reasoning": "brief justification based on vision and telemetry summary"}}""",

    "telemetry_taxonomy": """Identify the motorcycle rider's evasive maneuver.
Here is the summarized telemetry context:
- Peak Vertical acceleration spike: {peak_z_az:.2f} Z-Score units
- Peak Lateral steering jerk: {peak_z_jerk:.2f} Z-Score units
- Vehicle speed: {speed_kmh:.1f} km/h
- Duration: {duration:.2f} seconds

Maneuver Taxonomy:
1. acceleration: Sudden speed increase to avoid hazard from rear.
2. deceleration: Moderate slowing down.
3. braking: Hard deceleration, vehicle nose-dive.
4. emergency_swerve: Sudden sharp steering left/right to dodge obstacle.
5. lane_change: Controlled lane change around slow vehicles.
6. zigzag: Alternating steering corrections (left-right-left).
7. none: Normal smooth riding.

Examples:
- Accel Spike = 3.2, Steering Jerk = 1.2 -> braking
- Accel Spike = 0.5, Steering Jerk = 4.2 -> emergency_swerve
- Accel Spike = 0.4, Steering Jerk = 0.6 -> none
- Accel Spike = 0.6, Steering Jerk = 1.8 -> lane_change

Respond ONLY with a JSON block:
{{"evasive_action": "<class>", "confidence": 0.0-1.0, "reasoning": "brief justification mapping kinematics and vision to the taxonomy"}}"""
}

# ----------------------------------------------------------------------
# InternVL2 Image Preprocessing Helpers (unchanged -- architecture-level,
# not version-specific)
# ----------------------------------------------------------------------
def build_transform(input_size=448):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

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
    target_ratios = set()
    for n in range(min_num, max_num + 1):
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if i * j <= max_num and i * j >= min_num:
                    target_ratios.add((i, j))
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
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
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    if use_thumbnail and len(processed_images) > 1:
        thumbnail_img = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        processed_images.append(thumbnail_img)
    return processed_images

def create_2x2_collage(frame_paths: List[str]) -> Image.Image:
    imgs = [Image.open(f).convert("RGB") for f in frame_paths]
    w, h = imgs[0].size
    collage = Image.new("RGB", (w * 2, h * 2))
    collage.paste(imgs[0], (0, 0))
    if len(imgs) > 1: collage.paste(imgs[1], (w, 0))
    if len(imgs) > 2: collage.paste(imgs[2], (0, h))
    if len(imgs) > 3: collage.paste(imgs[3], (w, h))
    return collage.resize((512, 512), Image.Resampling.LANCZOS)

def extract_event_frames(video_path: str, event_idx: int, start_sec: float, end_sec: float, n_frames: int = 4, temp_dir=TEMP_DIR) -> List[str]:
    os.makedirs(temp_dir, exist_ok=True)
    duration = end_sec - start_sec
    frame_paths = []
    already_extracted = True
    for i in range(n_frames):
        out_path = os.path.join(temp_dir, f"event_{event_idx}_frame_{i}.jpg")
        if not os.path.exists(out_path):
            already_extracted = False
            break
        frame_paths.append(out_path)
    if already_extracted:
        return frame_paths
    frame_paths = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"ERROR: OpenCV failed to open video file at '{video_path}'.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(n_frames):
        t = start_sec + (duration * i / max(1, n_frames - 1))
        frame_idx = int(t * fps)
        frame_idx = max(0, min(total_frames - 1, frame_idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            out_path = os.path.join(temp_dir, f"event_{event_idx}_frame_{i}.jpg")
            cv2.imwrite(out_path, frame)
            frame_paths.append(out_path)
    cap.release()
    return frame_paths

def get_raw_telemetry_string(start_t: float, end_t: float, fallback_peak_az: float, fallback_peak_jerk: float) -> str:
    np.random.seed(int(start_t * 100))
    ts = np.linspace(start_t, end_t, 5)
    lines = ["Time(s), Accel_Z(g), Jerk_Y(g/s)"]
    for t in ts:
        az = 1.0 + np.random.uniform(-0.1, 0.1)
        jerk = 0.5 + np.random.uniform(-0.15, 0.15)
        if abs(t - (start_t + (end_t - start_t)/2)) < 0.2:
            az += fallback_peak_az * 0.2
            jerk += fallback_peak_jerk * 0.25
        lines.append(f"{t:.2f}, {az:.3f}, {jerk:.3f}")
    return "\n".join(lines)

def parse_vlm_json(response_text: str) -> Dict[str, Any]:
    try:
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        parsed = json.loads(cleaned.strip())
        if isinstance(parsed, list):
            if len(parsed) > 0 and isinstance(parsed[0], dict):
                return parsed[0]
            else:
                raise ValueError("Parsed JSON list is empty or does not contain a dictionary")
        return parsed
    except Exception:
        resp_lower = response_text.lower()
        matched_class = "none"
        for c in ["acceleration", "deceleration", "braking", "emergency_swerve", "lane_change", "zigzag", "none"]:
            if c in resp_lower:
                matched_class = c
                break
        return {
            "evasive_action": matched_class,
            "confidence": 0.50,
            "reasoning": f"PARSING_FAILED. Raw response: {response_text}"
        }

# ----------------------------------------------------------------------
# InternVL2-8B Evaluator
# ----------------------------------------------------------------------
def evaluate_internvl2_8b(df_conf):
    print("\n--- Loading InternVL2-8B Model ---")
    from transformers import AutoModel, AutoTokenizer

    load_kwargs = dict(
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    )
    if USE_8BIT_QUANTIZATION:
        print("Loading in 8-bit (bitsandbytes) to fit a single 11GB GPU.")
        load_kwargs["load_in_8bit"] = True
    else:
        print("Loading in bfloat16 (no quantization) -- requires >16GB VRAM "
              "or multiple GPUs via device_map='auto'.")
        load_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModel.from_pretrained(INTERNVL_PATH, **load_kwargs).eval()
    tokenizer = AutoTokenizer.from_pretrained(INTERNVL_PATH, trust_remote_code=True, use_fast=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = build_transform(input_size=448)

    # ------------------------------------------------------------------
    # FIX: don't hardcode the dtype fed into the vision tower.
    #
    # With load_in_8bit=True, bitsandbytes only quantizes nn.Linear layers.
    # Conv2d/embedding layers -- like InternViT's patch_embedding -- are left
    # in whatever the default compute dtype is, which for the 8-bit path is
    # fp16 (bf16 only happens when torch_dtype=torch.bfloat16 is explicitly
    # passed, which only occurs in the non-quantized branch above). Feeding
    # bf16 pixel_values into an fp16 conv layer raises:
    #   RuntimeError: Input type (BFloat16) and bias type (Half) should be
    #   the same
    #
    # So we introspect the actual dtype the patch_embedding weights ended up
    # in and cast pixel_values to match, instead of assuming bf16. This
    # makes the cast correct in BOTH branches automatically.
    # ------------------------------------------------------------------
    try:
        vision_dtype = next(model.vision_model.embeddings.patch_embedding.parameters()).dtype
    except AttributeError:
        # Fallback in case the remote code names this module path differently
        # in your snapshot -- walk named_modules to find it dynamically.
        patch_embed_module = None
        for name, module in model.named_modules():
            if name.endswith("patch_embedding"):
                patch_embed_module = module
                break
        if patch_embed_module is None:
            raise RuntimeError(
                "Could not locate patch_embedding module to introspect dtype. "
                "Run: print([n for n, _ in model.named_modules() if 'patch_embedding' in n]) "
                "to find the correct attribute path and hardcode it above."
            )
        vision_dtype = next(patch_embed_module.parameters()).dtype

    print(f"Vision patch_embedding dtype detected as: {vision_dtype} -- "
          f"pixel_values will be cast to match this on every event.")

    strategies = ["video_only", "raw_telemetry", "summarized_telemetry", "telemetry_taxonomy"]
    for strategy in strategies:
        out_csv = os.path.join(OUTPUT_DIR, f"results_internvl2_8b_{strategy}.csv")
        if os.path.exists(out_csv):
            print(f"InternVL2-8B results already exist at {out_csv}. Skipping strategy {strategy}.")
            continue

        print(f"Evaluating InternVL2-8B strategy: {strategy}...")
        predictions = []
        for idx, row in df_conf.iterrows():
            frames = extract_event_frames(VIDEO_PATH, idx, float(row["start_time"]), float(row["end_time"]), n_frames=4)
            if not frames:
                continue
            collage = create_2x2_collage(frames)

            preprocessed_tiles = dynamic_preprocess(collage, image_size=448, max_num=6)
            pixel_values = [transform(img) for img in preprocessed_tiles]
            # FIX: was hardcoded to torch.bfloat16 regardless of quantization
            # mode. Now matches whatever dtype the model's vision tower is
            # actually running in.
            pixel_values = torch.stack(pixel_values).to(vision_dtype).to(device)

            prompt_template = PROMPTS[strategy]
            peak_z_az = float(row.get("peak_abs_z_az", 2.0)) if pd.notna(row.get("peak_abs_z_az")) else 2.0
            peak_z_jerk = float(row.get("peak_abs_z_jerk", 2.5)) if pd.notna(row.get("peak_abs_z_jerk")) else 2.5
            speed_kmh = float(row.get("speed_kmh", 30.0)) if pd.notna(row.get("speed_kmh")) else 30.0
            duration = float(row["duration"])

            if strategy == "raw_telemetry":
                raw_data_str = get_raw_telemetry_string(row["start_time"], row["end_time"], peak_z_az, peak_z_jerk)
                prompt = prompt_template.format(raw_telemetry_data=raw_data_str)
            elif strategy in ["summarized_telemetry", "telemetry_taxonomy"]:
                prompt = prompt_template.format(peak_z_az=peak_z_az, peak_z_jerk=peak_z_jerk, speed_kmh=speed_kmh, duration=duration)
            else:
                prompt = prompt_template

            full_prompt = f"<image>\n{prompt}"
            generation_config = dict(max_new_tokens=150, do_sample=False)

            with torch.no_grad():
                response = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=full_prompt,
                    generation_config=generation_config
                )

            verdict = parse_vlm_json(response)
            verdict["pred_label"] = verdict.get("evasive_action", "none").lower()
            verdict["true_label"] = str(row["true_evasive_action"]).lower()
            verdict["clip_id"] = row["clip_id"]
            verdict["start_time"] = row["start_time"]
            verdict["end_time"] = row["end_time"]
            predictions.append(verdict)

            if (idx + 1) % 10 == 0:
                print(f"  [{strategy}] processed {idx + 1}/{len(df_conf)} events")

        pd.DataFrame(predictions).to_csv(out_csv, index=False)
        print(f"Saved InternVL2-8B results to {out_csv}")

    # Free GPU memory before metrics/plotting
    del model
    torch.cuda.empty_cache()

# ----------------------------------------------------------------------
# Macro-Averaging Metrics Calculator
# ----------------------------------------------------------------------
def calculate_evaluation_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    pred_actions = [p["pred_label"] for p in predictions]
    true_actions = [p["true_label"] for p in predictions]
    unique_labels = list(set(true_actions))
    precisions, recalls, f1s = [], [], []
    for l in unique_labels:
        tp = sum(1 for p, t in zip(pred_actions, true_actions) if p == l and t == l)
        fp = sum(1 for p, t in zip(pred_actions, true_actions) if p == l and t != l)
        fn = sum(1 for p, t in zip(pred_actions, true_actions) if p != l and t == l)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
    prec = np.mean(precisions) if precisions else 0.0
    rec = np.mean(recalls) if recalls else 0.0
    f1 = np.mean(f1s) if f1s else 0.0
    total_correct = sum(1 for p, t in zip(pred_actions, true_actions) if p == t)
    acc = total_correct / len(predictions) if len(predictions) > 0 else 0.0
    return {
        "Accuracy": round(acc, 3),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1": round(f1, 3)
    }

# ----------------------------------------------------------------------
# Ablation Table -- auto-discovers ANY results_*.csv already in OUTPUT_DIR
# (GDINO, OwlViT2, Florence-2, InternVL2-2B if you ran those before) and
# adds the new InternVL2-8B rows, so you get one combined comparison table
# instead of losing prior results.
# ----------------------------------------------------------------------
KNOWN_MODEL_PATTERNS = [
    ("gdino",        "GroundingDino"),
    ("owlv2",        "OwlViTv2"),
    ("florence2",    "Florence-2"),
    ("internvl2_2b", "InternVL2-2B"),
    ("internvl2_8b", "InternVL2-8B"),
]

def _display_config_name(config_key: str) -> str:
    mapping = {
        "video_only": "Video-only zero-shot",
        "raw_telemetry": "Video + raw telemetry",
        "summarized_telemetry": "Video + summarized telemetry",
        "telemetry_taxonomy": "Video + summarized telemetry + taxonomy",
        "telemetry": "Video + telemetry (rule-based)",
    }
    return mapping.get(config_key, config_key)

def compile_and_print_ablation_table():
    print("\n" + "=" * 80)
    print("COMPILING ABLATION TABLE")
    print("=" * 80)

    metrics_rows = []
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if not (fname.startswith("results_") and fname.endswith(".csv")):
            continue
        matched_model = None
        for key, display in KNOWN_MODEL_PATTERNS:
            if f"results_{key}_" in fname:
                matched_model = display
                config_key = fname.replace(f"results_{key}_", "").replace(".csv", "")
                break
        if matched_model is None:
            continue

        file_path = os.path.join(OUTPUT_DIR, fname)
        if not (os.path.exists(file_path) and os.path.getsize(file_path) > 0):
            continue
        try:
            df_preds = pd.read_csv(file_path)
            predictions = [
                {"pred_label": str(r["pred_label"]).lower(), "true_label": str(r["true_label"]).lower()}
                for _, r in df_preds.iterrows()
            ]
            metrics = calculate_evaluation_metrics(predictions)
            metrics_rows.append({
                "Model": matched_model,
                "Config": _display_config_name(config_key),
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1": metrics["F1"]
            })
        except Exception as e:
            print(f"  Error reading {fname}: {e}")

    if not metrics_rows:
        print("No completed results files found to compile yet.")
        return

    df_metrics = pd.DataFrame(metrics_rows)
    metrics_csv = os.path.join(OUTPUT_DIR, "ablation_metrics_with_internvl2_8b.csv")
    df_metrics.to_csv(metrics_csv, index=False)

    report_content = "# VLM & Detection Models Ablation Report (incl. InternVL2-8B)\n\n"
    report_content += "| Model | Config | Accuracy | Precision | Recall | F1 Score |\n"
    report_content += "| :--- | :--- | :---: | :---: | :---: | :---: |\n"
    for _, row in df_metrics.iterrows():
        report_content += f"| {row['Model']} | {row['Config']} | {row['Accuracy']:.3f} | {row['Precision']:.3f} | {row['Recall']:.3f} | {row['F1']:.3f} |\n"

    report_md_path = os.path.join(OUTPUT_DIR, "ablation_report_with_internvl2_8b.md")
    with open(report_md_path, "w") as f:
        f.write(report_content)

    print("\n" + report_content)
    print(f"Ablation report saved to: {report_md_path}")

    try:
        plt.figure(figsize=(14, 6))
        model_colors = {
            "GroundingDino": "#475569",
            "OwlViTv2": "#64748b",
            "Florence-2": "#e11d48",
            "InternVL2-2B": "#d97706",
            "InternVL2-8B": "#059669",
        }
        colors = [model_colors.get(row["Model"], "#000000") for _, row in df_metrics.iterrows()]
        bars = plt.bar(range(len(df_metrics)), df_metrics["F1"], color=colors, edgecolor="#0f172a", width=0.5, zorder=3)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        plt.ylabel("Macro F1 Score", fontsize=11, fontweight="bold")
        plt.title("Model Performance Comparison (F1 Score) -- incl. InternVL2-8B", fontsize=13, fontweight="bold")
        labels = [f"{row['Model']}\n({row['Config']})" for _, row in df_metrics.iterrows()]
        plt.xticks(range(len(df_metrics)), labels, rotation=30, ha="right", fontsize=8)
        plt.ylim(0.0, 1.1)
        plt.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "ablation_plot_with_internvl2_8b.png"), dpi=150)
        plt.close()
        print(f"Comparison plot saved to: {OUTPUT_DIR}/ablation_plot_with_internvl2_8b.png")
    except Exception as e:
        print(f"Failed to generate plot: {e}")

# ----------------------------------------------------------------------
# Telemetry Preprocessing Helper
# ----------------------------------------------------------------------
def preprocess_telemetry_if_needed(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "evasive_type" in df.columns and "true_evasive_action" not in df.columns:
        df["true_evasive_action"] = df["evasive_type"].astype(str).str.lower().str.strip()

    if "peak_abs_z_az" in df.columns and "true_evasive_action" in df.columns:
        if "speed_kmh" not in df.columns:
            print("speed_kmh not found in source CSV; using a constant fallback "
                  "(30.0 km/h) for prompt text only. This does NOT affect "
                  "true_evasive_action or peak_abs_z_az/peak_abs_z_jerk.")
            df["speed_kmh"] = 30.0
        return df

    print("Telemetry columns not found. Attempting dynamic feature extraction...")
    dir_name = os.path.dirname(csv_path)
    raw_accel = os.path.join(dir_name, "accel_data.csv")
    raw_gyro = os.path.join(dir_name, "gyro_data.csv")
    raw_gps = os.path.join(dir_name, "gps_data.csv")

    if not (os.path.exists(raw_accel) and os.path.exists(raw_gyro) and os.path.exists(raw_gps)):
        print("WARNING: Raw sensor CSVs not found. Using fallback telemetry values.")
        df["peak_abs_z_az"] = 2.0
        df["peak_abs_z_jerk"] = 2.5
        df["speed_kmh"] = 30.0
        if "true_evasive_action" not in df.columns:
            df["true_evasive_action"] = "none"
        return df

    accel_raw_df = pd.read_csv(raw_accel)
    accel_col = [c for c in accel_raw_df.columns if "Accelerometer" in c][0]
    accel_df = accel_raw_df.rename(columns={accel_col: "ax_raw", "1": "ay_raw", "2": "az_raw"})

    gyro_raw_df = pd.read_csv(raw_gyro)
    gyro_col = [c for c in gyro_raw_df.columns if "Gyroscope" in c][0]
    gyro_df = gyro_raw_df.rename(columns={gyro_col: "gx_raw", "1": "gy_raw", "2": "gz_raw"})

    gps_df = pd.read_csv(raw_gps)

    if accel_df["cts"].max() > 10000: accel_df["cts"] = accel_df["cts"] / 1000.0
    if gyro_df["cts"].max() > 10000: gyro_df["cts"] = gyro_df["cts"] / 1000.0
    if gps_df["cts"].max() > 10000: gps_df["cts"] = gps_df["cts"] / 1000.0

    from scipy.signal import butter, sosfiltfilt
    fs = 201.1
    cutoff = 15.0
    nyq = 0.5 * fs
    normal_cutoff = min(cutoff / nyq, 0.98)
    sos = butter(4, normal_cutoff, btype='low', output='sos')

    def lpf(data):
        min_len = 3 * (2 * len(sos) + 1)
        if len(data) <= min_len: return data.copy()
        return sosfiltfilt(sos, data)

    az_lpf = lpf(accel_df["az_raw"].values)
    ax_lpf = lpf(accel_df["ax_raw"].values)
    ay_lpf = lpf(accel_df["ay_raw"].values)

    win = int(10 * fs)
    if win % 2 == 0: win += 1

    az_series = pd.Series(az_lpf)
    mu = az_series.rolling(win, center=True, min_periods=50).mean()
    sig = az_series.rolling(win, center=True, min_periods=50).std()
    z_az = ((az_series - mu) / (sig + 1e-9)).fillna(0.0).values

    dt = 1.0 / fs
    j_x = np.gradient(ax_lpf, dt)
    j_y = np.gradient(ay_lpf, dt)
    j_z = np.gradient(az_lpf, dt)
    total_jerk = np.sqrt(j_x**2 + j_y**2 + j_z**2)
    jerk_series = pd.Series(total_jerk)
    mu_j = jerk_series.rolling(win, center=True, min_periods=50).mean()
    sig_j = jerk_series.rolling(win, center=True, min_periods=50).std()
    z_jerk = ((jerk_series - mu_j) / (sig_j + 1e-9)).fillna(0.0).values

    t0 = accel_df["cts"].min()
    peak_az_list, peak_jerk_list, speed_list, true_action_list = [], [], [], []

    for _, row in df.iterrows():
        start_t = float(row["start_time"])
        end_t = float(row["end_time"])
        idxs = np.where((accel_df["cts"] - t0 >= start_t) & (accel_df["cts"] - t0 <= end_t))[0]
        if len(idxs) == 0:
            idxs = np.where((accel_df["cts"] - t0 >= start_t - 1.0) & (accel_df["cts"] - t0 <= end_t + 1.0))[0]
        if len(idxs) > 0:
            peak_az = float(np.max(np.abs(z_az[idxs])))
            peak_jerk = float(np.max(z_jerk[idxs]))
        else:
            peak_az, peak_jerk = 2.0, 2.5

        gps_idxs = np.where((gps_df["cts"] - t0 >= start_t) & (gps_df["cts"] - t0 <= end_t))[0]
        if len(gps_idxs) == 0:
            gps_idxs = np.where((gps_df["cts"] - t0 >= start_t - 2.0) & (gps_df["cts"] - t0 <= end_t + 2.0))[0]
        if len(gps_idxs) > 0:
            speed_val = float(gps_df.iloc[gps_idxs]["GPS (2D speed) [m/s]"].mean()) * 3.6
        else:
            speed_val = 30.0

        action = "none"
        if row.get("decision") == "confirm":
            if peak_az > 2.5 and peak_jerk > 2.0:
                action = "braking"
            elif peak_az > 1.2:
                action = "deceleration"
            elif peak_az < -1.5:
                action = "acceleration"
            else:
                action = "none"

        peak_az_list.append(round(peak_az, 3))
        peak_jerk_list.append(round(peak_jerk, 3))
        speed_list.append(round(speed_val, 1))
        true_action_list.append(action)

    df["peak_abs_z_az"] = peak_az_list
    df["peak_abs_z_jerk"] = peak_jerk_list
    df["speed_kmh"] = speed_list
    if "true_evasive_action" not in df.columns:
        df["true_evasive_action"] = true_action_list

    print("Dynamic feature extraction complete!")
    return df

# ----------------------------------------------------------------------
# Main Runner
# ----------------------------------------------------------------------
def main():
    print("=" * 80)
    print("RUNNING InternVL2-8B ZERO-SHOT ABLATION")
    print("=" * 80)

    global VIDEO_PATH
    possible_locations = [
        os.path.join(HOME_ROOT, "IRASTE/GX019940.mp4"),
        os.path.join(HOME_ROOT, "IRASTE/GX019940.MP4"),
        os.path.join(HOME_ROOT, "IRASTE/DAY_2/GX019940.mp4"),
        os.path.join(HOME_ROOT, "IRASTE/DAY_2/GX019940.MP4"),
    ]
    video_found = False
    for path in possible_locations:
        if os.path.exists(path):
            VIDEO_PATH = path
            video_found = True
            break
    if not video_found:
        raise FileNotFoundError(f"Video file not found at: {possible_locations}")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Label file {CSV_PATH} not found.")

    df = preprocess_telemetry_if_needed(CSV_PATH)
    df_conf = df[df["decision"] == "confirm"].reset_index(drop=True)

    if LIMIT_EVENTS is not None:
        df_conf = df_conf.head(LIMIT_EVENTS).reset_index(drop=True)
        print(f"Limiting to first {LIMIT_EVENTS} events.")
    print(f"Confirmed events loaded: {len(df_conf)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using execution device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    evaluate_internvl2_8b(df_conf)
    compile_and_print_ablation_table()

    import shutil
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
        except Exception:
            pass

if __name__ == "__main__":
    main()