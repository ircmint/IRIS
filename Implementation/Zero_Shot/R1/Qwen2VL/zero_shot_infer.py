"""
zero_shot_infer.py - Zero-shot evasive-riding-event detection with VLMs.

Design decision (documented, not hidden): the model is asked to judge each of
the 40 hand-reviewed candidate windows per rider (Custom_Data/RiderX/gold_candidates.csv)
rather than scanning the full video at arbitrary strides. These 40 windows are
exactly the set that has real human ground truth (decision=confirm/reject), so
scoring against anything else would have no gold to compare to. The candidate
windows themselves come from an upstream IMU-peak-detection heuristic (not the
VLM) - this is a standard two-stage design (propose windows cheaply, classify
with the expensive model) and is applied identically across all 3 conditions,
so it does not advantage any one condition.

The gold `decision` / `notes` columns are NEVER passed into the prompt -
they are read only for scoring, in a separate script.

Conditions (selected via --condition):
  video_only            : frames only, no telemetry text at all.
  raw_telemetry         : frames + a small sampled slice of raw
                           Accelerometer.csv / Gyroscope.csv rows spanning the
                           event window (not the full file - down-sampled to
                           at most RAW_SAMPLE_ROWS rows so the prompt stays a
                           reasonable size).
  summarized_telemetry  : frames + the upstream-computed summary stats for
                           this candidate window (peak jerk, peak lateral
                           accel, duration, n_samples, peak_confidence) -
                           i.e. the same numeric hint an analyst would see,
                           without raw waveform rows.

Usage:
  python zero_shot_infer.py --rider Rider1_NJ --model qwen2-2b --condition video_only \
      --model_path $SCRATCH_ROOT/models/Qwen2-VL-2B-Instruct \
      --data_root $SCRATCH_ROOT/custom_data \
      --out_dir $SCRATCH_ROOT/custom_data/predictions

Output: predictions/<rider>__<model>__<condition>.json - one record per
candidate event with the model's raw output + parsed is_evasive_event/
confidence/reasoning, plus the window's start/end (for later IoU scoring
against gold_candidates.csv in a separate scoring script).
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
import re
import json
import argparse

import cv2
import numpy as np
import pandas as pd
from PIL import Image

RESIZE_MAX_SIDE = 448
FRAMES_PER_EVENT = 4
RAW_SAMPLE_ROWS = 12          # cap on raw telemetry rows shown per axis
MAX_NEW_TOKENS = 200
MIN_CLIP_DURATION = 1.0


def sample_frames(video_path, start, end, n_frames=FRAMES_PER_EVENT):
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


def get_clip_bounds(row):
    start, end = float(row["start_time"]), float(row["end_time"])
    if end - start < MIN_CLIP_DURATION:
        center = (start + end) / 2.0
        half = MIN_CLIP_DURATION / 2.0
        start, end = max(center - half, 0.0), center + half
    return start, end


def load_raw_slice(csv_path, start, end, max_rows=RAW_SAMPLE_ROWS):
    """Down-sampled raw rows spanning [start,end] (+/- 0.5s pad), evenly
    picked so we never dump the whole file into the prompt."""
    df = pd.read_csv(csv_path)
    time_col = None
    for c in df.columns:
        if "time" in c.lower() or "seconds" in c.lower():
            time_col = c
            break
    if time_col is None:
        time_col = df.columns[0]
    mask = (df[time_col] >= start - 0.5) & (df[time_col] <= end + 0.5)
    sub = df[mask]
    if len(sub) == 0:
        return "(no raw samples found in this window)"
    if len(sub) > max_rows:
        idx = np.linspace(0, len(sub) - 1, max_rows).astype(int)
        sub = sub.iloc[idx]
    return sub.to_csv(index=False)


COT_ANCHOR_EXAMPLES = (
    "Here are two short worked examples showing the expected reasoning "
    "format (these are illustrative anchors only, not this clip):\n\n"
    "Example A (evasive): Frame-by-frame: frame 1 shows the rider centered "
    "in the lane at steady speed; frame 2 shows a pedestrian stepping off "
    "the curb directly ahead and the rider's body leaning sharply left with "
    "the brake light of a nearby vehicle visible; frame 3 shows the rider "
    "now shifted into the adjacent lane, still leaning; frame 4 shows the "
    "rider straightening back up past the pedestrian. Motion delta: yes, a "
    "sudden lean + lane shift within about 1 second, directly triggered by "
    "the pedestrian entering the path -- very little reaction time. -> "
    "{\"is_evasive_event\": true, \"confidence\": 0.9, \"reasoning\": "
    "\"Sharp left lean and lane shift triggered by a pedestrian stepping "
    "into the path, minimal reaction time.\"}\n\n"
    "Example B (normal): Frame-by-frame: all 4 frames show the rider "
    "riding in a straight line, centered in the same lane, consistent "
    "speed and lean angle, no brake lights, traffic (if any) moving in the "
    "same direction at a steady distance. Motion delta: no sudden lean, "
    "brake, or lane change in any frame; nothing to react to. -> "
    "{\"is_evasive_event\": false, \"confidence\": 0.85, \"reasoning\": "
    "\"Consistent lane position and speed across all frames, no braking or "
    "swerve, no hazard requiring reaction.\"}\n\n"
)


def build_prompt(condition, row, accel_csv, gyro_csv):
    task = (
        "You are analyzing a short first-person dashcam video clip from a "
        "two-wheeler rider in mixed Indian traffic. Judge whether this clip "
        "shows an EVASIVE RIDING EVENT: a sudden swerve, hard brake/lane "
        "squeeze, close near-miss with another vehicle/pedestrian, or "
        "abrupt lane change made to avoid a hazard. Ordinary riding through "
        "traffic (even dense/congested traffic) with no sudden avoidance "
        "maneuver is NOT evasive.\n\n"
    )
    if condition == "video_only_cot":
        task += (
            "You are shown 4 frames sampled in chronological order across "
            "the clip. Reason step by step BEFORE giving your verdict:\n"
            "1. Describe what you observe in sequence across the frames: "
            "the rider's position/lane, lean angle, any brake lights, and "
            "proximity to other vehicles, pedestrians, or obstacles, and "
            "how these change frame to frame.\n"
            "2. Identify whether there is a SUDDEN motion delta between "
            "frames -- an abrupt lean, hard brake, or abrupt lane change "
            "-- as opposed to a gradual/steady change.\n"
            "3. If there is a sudden motion delta, identify the immediate "
            "trigger visible in the frames (a specific vehicle, pedestrian, "
            "or obstacle the rider appears to be reacting to) and roughly "
            "how much reaction time/distance the rider had before it -- "
            "very little (evasive) vs. plenty of time/no real hazard "
            "(not evasive).\n\n"
            + COT_ANCHOR_EXAMPLES
            + "Now do the same step-by-step analysis for the actual clip "
            "shown to you, then give your verdict.\n\n"
        )
    if condition == "raw_telemetry":
        start, end = get_clip_bounds(row)
        accel_slice = load_raw_slice(accel_csv, start, end)
        gyro_slice = load_raw_slice(gyro_csv, start, end)
        task += (
            "Raw accelerometer samples (m/s^2) spanning the clip window:\n"
            f"{accel_slice}\n"
            "Raw gyroscope samples (rad/s) spanning the clip window:\n"
            f"{gyro_slice}\n\n"
        )
    elif condition == "summarized_telemetry":
        task += (
            "Pre-computed motion-sensor summary for this clip (from an "
            "upstream IMU peak-detection stage, provided as a hint - not a "
            "ground-truth label):\n"
            f"  peak jerk (z-axis) = {row.get('peak_abs_z_jerk', 'NA')}\n"
            f"  peak lateral acceleration (z-axis) = {row.get('peak_abs_z_az', 'NA')}\n"
            f"  event duration = {row.get('duration', 'NA')}s\n"
            f"  n_samples in window = {row.get('n_samples', 'NA')}\n"
            f"  upstream candidate confidence = {row.get('peak_confidence', 'NA')}\n\n"
        )
    # condition == "video_only": no telemetry text added.

    if condition == "video_only_cot":
        task += (
            'Respond with ONLY a JSON object of this exact shape:\n'
            '{"is_evasive_event": <true/false>, "confidence": <float 0-1>, '
            '"reasoning": "<your step-by-step frame-by-frame analysis from '
            'steps 1-3 above, condensed, ending in the motion-delta/trigger '
            'conclusion that justifies your verdict>"}\n'
            "Put your full step-by-step reasoning inside the \"reasoning\" "
            "field itself. Do not include any text outside the JSON object."
        )
    else:
        task += (
            'Respond with ONLY a JSON object of this exact shape:\n'
            '{"is_evasive_event": <true/false>, "confidence": <float 0-1>, '
            '"reasoning": "<short justification grounded only in what is visible/measured>"}\n'
            "Do not include any text outside the JSON object."
        )
    return task


def extract_json(text):
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


def load_model(model_key, model_path):
    import torch
    if model_key in ("qwen2-2b",):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=64 * 28 * 28, max_pixels=448 * 448
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        model.eval()
        return "qwen2vl", model, processor
    elif model_key in ("qwen25-3b",):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=64 * 28 * 28, max_pixels=448 * 448
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        model.eval()
        return "qwen2vl", model, processor
    elif model_key in ("paligemma-3b",):
        from transformers import PaliGemmaForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_path)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={"": 0}
        )
        model.eval()
        return "paligemma", model, processor
    elif model_key in ("internvl2-8b",):
        # Reuses the loading pattern from ~/IRASTE/DAY_4/FineTune.py (the
        # earlier fine-tuning script for this same checkpoint): 4-bit
        # quantized load via bitsandbytes, trust_remote_code=True (InternVL2
        # ships custom modeling code, not a standard transformers class).
        # 8B params in fp16 would need ~16GB, more than this 11GB GPU has,
        # so 4-bit is not optional here, it's required to fit at all.
        from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModel.from_pretrained(
            model_path, quantization_config=quantization_config,
            trust_remote_code=True, device_map="auto",
        )
        model.eval()

        # NOTE: a "defensive" monkey-patch (forcing use_cache=False in
        # generate() + a custom prepare_inputs_for_generation, copied
        # verbatim from a DIFFERENT InternVL2 checkpoint's debug session
        # at ~/IRASTE/DAY_3/run_server_internvl_only.py) used to live here.
        # Verified via direct debugging (2026-07-25) that this patch was
        # itself the bug for this checkpoint/transformers combo: every
        # single event failed with
        #   "The size of tensor a (1257) must match the size of tensor b
        #    (2512) at non-singleton dimension 3"
        # With the patch removed entirely, model.chat() works correctly
        # out of the box (confirmed: clean load, real generation, valid
        # JSON output on a real Rider1_NJ event). Removed rather than
        # further patched -- the DynamicCache bug it was defending against
        # was never actually observed on this checkpoint/transformers
        # 4.37.2 combination.
        return "internvl2", model, tokenizer
    else:
        raise ValueError(f"Unsupported model_key for this script: {model_key}")


def run_paligemma_generation(model, processor, frames, condition, row, accel_csv, gyro_csv):
    """paligemma-3b-pt-224 is a base (non-chat) VLM: single image in, short
    text out. It cannot take 4 separate frames as a chat model can, so we
    build one 2x2 montage image (same visual content, one image) and use a
    plain prefix-completion prompt (PaliGemma's expected format 'answer en\n<q>').
    The base checkpoint doesn't reliably follow multi-sentence instructions or
    emit valid JSON — this is a real, documented limitation of the pt-224
    checkpoint being used, not a script bug.

    BUGFIX (found during Rider1_NJ run): this function used to hardcode the
    same short_prompt regardless of `condition`, so raw_telemetry and
    summarized_telemetry silently produced byte-identical output to
    video_only (verified: 130/130 identical raw_output strings on
    Rider1_NJ). Now builds a condition-appropriate telemetry hint, kept short
    since the base checkpoint has very limited instruction-following for
    long text, and appends it to the yes/no question so the condition
    actually reaches the model."""
    import torch
    from PIL import Image
    w, h = frames[0].size
    montage = Image.new("RGB", (w * 2, h * 2))
    positions = [(0, 0), (w, 0), (0, h), (w, h)]
    for f, pos in zip(frames, positions):
        montage.paste(f.resize((w, h)), pos)

    telemetry_note = ""
    if condition == "raw_telemetry":
        start, end = get_clip_bounds(row)
        accel_slice = load_raw_slice(accel_csv, start, end, max_rows=4)
        gyro_slice = load_raw_slice(gyro_csv, start, end, max_rows=4)
        telemetry_note = (
            f" Accelerometer samples: {accel_slice.strip()[:180]} "
            f"Gyroscope samples: {gyro_slice.strip()[:180]}."
        )
    elif condition == "summarized_telemetry":
        telemetry_note = (
            f" Motion sensor summary: peak jerk="
            f"{row.get('peak_abs_z_jerk', 'NA')}, peak lateral accel="
            f"{row.get('peak_abs_z_az', 'NA')}, duration={row.get('duration', 'NA')}s, "
            f"upstream candidate confidence={row.get('peak_confidence', 'NA')}."
        )

    if condition == "video_only_cot":
        # EXPERIMENT (paligemma-3b-pt-224 is a base, non-instruction-tuned
        # VQA checkpoint -- multi-step CoT + few-shot is a real test of
        # whether structured prompting can rein in this model's observed
        # near-constant "yes" collapse (recall~1.0, precision~0.10-0.20).
        # Kept deliberately short per anchor example since the base
        # checkpoint has weak long-instruction-following (documented
        # elsewhere in this file) -- a long CoT prompt may simply be
        # truncated/ignored by this checkpoint, which is itself part of
        # what this experiment is testing, not assumed to work.
        short_prompt = (
            "answer en Look at this 2x2 grid of 4 frames in reading order "
            "(top-left, top-right, bottom-left, bottom-right) from a "
            "two-wheeler dashcam clip. First silently compare the frames: "
            "does the rider's lane position or lean angle change suddenly "
            "between frames, and if so, is there a specific vehicle, "
            "pedestrian, or obstacle visible that the rider is reacting to "
            "with very little time? Example: steady lane position across "
            "all 4 frames, no hazard = answer no. Example: sharp lean plus "
            "a car or pedestrian suddenly in the path = answer yes." +
            telemetry_note +
            " Is this an evasive maneuver? Answer yes or no and briefly why."
        )
        max_tokens_this_call = 400
    else:
        short_prompt = (
            "answer en Is this two-wheeler rider performing a sudden evasive "
            "maneuver (swerve, hard brake, near-miss avoidance)?" + telemetry_note +
            " Answer yes or no and briefly why."
        )
        max_tokens_this_call = MAX_NEW_TOKENS

    inputs = processor(text=short_prompt, images=montage, return_tensors="pt").to(
        model.device, model.dtype
    )
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=max_tokens_this_call, do_sample=False)
    gen_ids_trimmed = gen_ids[:, inputs["input_ids"].shape[1]:]
    output_text = processor.batch_decode(gen_ids_trimmed, skip_special_tokens=True)[0]
    return output_text


INTERNVL_IMAGENET_MEAN = (0.485, 0.456, 0.406)
INTERNVL_IMAGENET_STD = (0.229, 0.224, 0.225)


def _internvl_build_transform(input_size):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=INTERNVL_IMAGENET_MEAN, std=INTERNVL_IMAGENET_STD),
    ])


def run_internvl_generation(model, tokenizer, frames, prompt_text):
    """InternVL2-8B zero-shot, real chat-API inference (not the LoRA
    fine-tuning path in FineTune.py -- this reuses only its image
    preprocessing: build_transform/dynamic_preprocess, verbatim logic,
    single tile per frame (max_num=1, no thumbnail) to keep 4 frames a
    manageable pixel_values tensor on an 11GB GPU under 4-bit quantization.
    model.chat() does its own <image> token expansion internally (per
    InternVL2's documented API), so we just prefix each image with an
    'Image-N:' marker the way their multi-image chat examples do."""
    import torch
    transform = _internvl_build_transform(448)
    pixel_values_list = []
    num_patches_list = []
    for f in frames:
        tiles = [f]  # max_num=1 -> dynamic_preprocess would just return [f]-equivalent; skip tiling entirely for a single 448-resized frame
        tiles_t = torch.stack([transform(t) for t in tiles])
        pixel_values_list.append(tiles_t)
        num_patches_list.append(tiles_t.shape[0])
    pixel_values = torch.cat(pixel_values_list, dim=0).to(torch.float16).to(model.device)

    image_prefix = "".join(f"Image-{i+1}: <image>\n" for i in range(len(frames)))
    question = image_prefix + prompt_text
    generation_config = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    response = model.chat(
        tokenizer, pixel_values, question, generation_config,
        num_patches_list=num_patches_list,
    )
    return response


def run_qwen_generation(model, processor, frames, prompt_text, max_new_tokens=MAX_NEW_TOKENS):
    from qwen_vl_utils import process_vision_info
    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": f} for f in frames]
        + [{"type": "text", "text": prompt_text}],
    }]
    chat_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat_text], images=image_inputs, videos=video_inputs, return_tensors="pt",
    ).to(model.device)
    import torch
    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_ids_trimmed = gen_ids[:, inputs["input_ids"].shape[1]:]
    output_text = processor.batch_decode(
        gen_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    return output_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rider", required=True)
    ap.add_argument("--model", required=True,
                     choices=["qwen2-2b", "qwen25-3b", "paligemma-3b", "internvl2-8b"])
    ap.add_argument("--condition", required=True,
                     choices=["video_only", "raw_telemetry", "summarized_telemetry", "video_only_cot"])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_root", default=os.path.join(SCRATCH_ROOT, "custom_data"))
    ap.add_argument("--out_dir", default=os.path.join(SCRATCH_ROOT, "custom_data/predictions"))
    ap.add_argument("--video_suffix", default="_720p.mp4",
                     help="filename suffix for the (re-encoded) video to load")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rider_dir = os.path.join(args.data_root, args.rider)
    gold_path = os.path.join(rider_dir, "gold_candidates.csv")
    video_path = os.path.join(rider_dir, args.rider + args.video_suffix)
    accel_csv = os.path.join(rider_dir, "Accelerometer.csv")
    gyro_csv = os.path.join(rider_dir, "Gyroscope.csv")

    if not os.path.exists(gold_path):
        raise FileNotFoundError(gold_path)
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    df = pd.read_csv(gold_path)
    print(f"Loaded {len(df)} candidate windows for {args.rider}")

    kind, model, processor = load_model(args.model, args.model_path)

    results = []
    for i, row in df.iterrows():
        start, end = get_clip_bounds(row)
        try:
            frames = sample_frames(video_path, start, end)
            prompt_text = build_prompt(args.condition, row, accel_csv, gyro_csv)
            if kind == "qwen2vl":
                gen_tokens = 400 if args.condition == "video_only_cot" else MAX_NEW_TOKENS
                output_text = run_qwen_generation(model, processor, frames, prompt_text, max_new_tokens=gen_tokens)
            elif kind == "paligemma":
                output_text = run_paligemma_generation(model, processor, frames, args.condition, row, accel_csv, gyro_csv)
            elif kind == "internvl2":
                output_text = run_internvl_generation(model, processor, frames, prompt_text)
            else:
                raise ValueError(f"unknown model kind {kind}")
        except Exception as e:
            print(f"[event {i}] ERROR: {e}")
            results.append({
                "event_idx": int(i), "start_time": start, "end_time": end,
                "error": str(e),
            })
            continue

        parsed = extract_json(output_text)
        rec = {
            "event_idx": int(i),
            "start_time": start,
            "end_time": end,
            "raw_output": output_text,
        }
        if parsed is not None and "is_evasive_event" in parsed:
            rec["parsed"] = True
            rec["parse_method"] = "json"
            rec["pred_is_evasive"] = bool(parsed.get("is_evasive_event"))
            rec["pred_confidence"] = parsed.get("confidence", None)
            rec["pred_reasoning"] = parsed.get("reasoning", "")
        elif kind == "paligemma":
            # base pt-224 checkpoint does not reliably emit JSON; fall back
            # to a documented, honest yes/no text heuristic on its raw
            # completion instead of silently marking everything unparsed.
            low = output_text.strip().lower()
            said_yes = low.startswith("yes") or " yes" in low[:20]
            said_no = low.startswith("no") or " no" in low[:20]
            if said_yes or said_no:
                rec["parsed"] = True
                rec["parse_method"] = "yesno_heuristic"
                rec["pred_is_evasive"] = bool(said_yes and not said_no)
                rec["pred_confidence"] = None
                rec["pred_reasoning"] = output_text.strip()
            else:
                rec["parsed"] = False
                rec["parse_method"] = "none"
                rec["pred_is_evasive"] = None
                rec["pred_confidence"] = None
                rec["pred_reasoning"] = ""
        else:
            rec["parsed"] = False
            rec["parse_method"] = "none"
            rec["pred_is_evasive"] = None
            rec["pred_confidence"] = None
            rec["pred_reasoning"] = ""
        results.append(rec)
        print(f"[{i+1}/{len(df)}] start={start:.2f} pred={rec['pred_is_evasive']} parsed={rec['parsed']}")

    out_path = os.path.join(args.out_dir, f"{args.rider}__{args.model}__{args.condition}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
