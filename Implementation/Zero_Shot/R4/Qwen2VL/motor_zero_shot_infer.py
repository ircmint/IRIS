"""
Zero-shot evasive-event VLM inference for the MOTOR (GoPro-native IMU) dataset.
Adapted, not rewritten from scratch, from:
  G:\\Driver_Behaviour\\pipeline\\Custom_Data\\Results\\code\\zero_shot_infer.py
Differences from the reference (documented, not hidden):
  - No "rider" concept; candidates are grouped by clip_id (GX010422 etc.)
    across a single combined gold_candidates.csv covering all 4 videos.
  - Video files are the raw GoPro .MP4 (no 720p re-encode was done for this
    dataset given time constraints) -- cv2 can read the HEVC MP4 directly.
  - Telemetry CSVs are the ACCL/GYRO files produced by gopro-telemetry, with
    columns cts_ms(ms)/ax,ay,az or gx,gy,gz, NOT the phyphox time_s/ax,ay,az
    schema, so load_raw_slice's time-column detection and unit label in the
    prompt are adjusted accordingly (cts is milliseconds, so converted to
    seconds before windowing).
  - summarized_telemetry uses this dataset's own upstream columns
    (peak_abs_z_jerk, peak_abs_z_az = peak |z_amag| from merge_filter_threshold.py)
    which are conceptually the same jerk/lateral-accel z-score summary as the
    reference pipeline, just computed by adaptive_threshold() in this repo.
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
RAW_SAMPLE_ROWS = 12
MAX_NEW_TOKENS = 200
MIN_CLIP_DURATION = 1.0


def sample_frames(video_path, start, end, n_frames=FRAMES_PER_EVENT):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
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


CLIP_PAD = 0.5  # matches extract_candidate_clips.py's PAD


def get_pretrimmed_bounds(row):
    """When --clips_dir is used, each video file is already trimmed to
    [orig_start - PAD, orig_end + PAD] (after MIN_CLIP_DURATION expansion was
    applied at extraction time), so frame sampling must use bounds relative
    to that trimmed file, not the absolute original video timeline."""
    start, end = get_clip_bounds(row)
    return CLIP_PAD, CLIP_PAD + (end - start)


def load_raw_slice(csv_path, start, end, max_rows=RAW_SAMPLE_ROWS):
    """This dataset's ACCL/GYRO csvs use cts (ms) as the time column, not
    seconds -- convert before windowing, unlike the phyphox reference."""
    df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="ignore")
    time_col = df.columns[0]  # 'cts'
    t_sec = df[time_col].astype(float) / 1000.0
    mask = (t_sec >= start - 0.5) & (t_sec <= end + 0.5)
    sub = df[mask]
    if len(sub) == 0:
        return "(no raw samples found in this window)"
    if len(sub) > max_rows:
        idx = np.linspace(0, len(sub) - 1, max_rows).astype(int)
        sub = sub.iloc[idx]
    return sub.to_csv(index=False)


def build_prompt(condition, row, accel_csv, gyro_csv):
    task = (
        "You are analyzing a short first-person dashcam video clip from a "
        "two-wheeler (motorcycle) rider in mixed Indian traffic. Judge "
        "whether this clip shows an EVASIVE RIDING EVENT: a sudden swerve, "
        "hard brake/lane squeeze, close near-miss with another vehicle/"
        "pedestrian/cyclist, or abrupt lane change made to avoid a hazard. "
        "Ordinary riding through traffic (even dense/congested traffic) with "
        "no sudden avoidance maneuver, and ordinary road-surface bumps/"
        "potholes with no steering reaction, are NOT evasive.\n\n"
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
            "upstream IMU adaptive-thresholding stage, provided as a hint - "
            "not a ground-truth label):\n"
            f"  peak |z-score| jerk = {row.get('peak_abs_z_jerk', 'NA')}\n"
            f"  peak |z-score| accel magnitude = {row.get('peak_abs_z_az', 'NA')}\n"
            f"  event duration = {row.get('duration', 'NA')}s\n"
            f"  n_samples in window = {row.get('n_samples', 'NA')}\n"
            f"  upstream candidate confidence = {row.get('peak_confidence', 'NA')}\n\n"
        )
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
    if model_key == "qwen2-2b":
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=64 * 28 * 28, max_pixels=448 * 448
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        model.eval()
        return "qwen2vl", model, processor
    elif model_key == "qwen25-3b":
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=64 * 28 * 28, max_pixels=448 * 448
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        model.eval()
        return "qwen2vl", model, processor
    else:
        raise ValueError(f"Unsupported model_key for this script: {model_key}")


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
    ap.add_argument("--model", required=True, choices=["qwen2-2b", "qwen25-3b"])
    ap.add_argument("--condition", required=True,
                     choices=["video_only", "raw_telemetry", "summarized_telemetry"])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_root", default=os.path.join(SCRATCH_ROOT, "motor_data"))
    ap.add_argument("--out_dir", default=os.path.join(SCRATCH_ROOT, "motor_data/predictions"))
    ap.add_argument("--gold_csv", default=None,
                     help="defaults to <data_root>/gold_candidates.csv")
    ap.add_argument("--clips_dir", default=None,
                     help="if set, use pre-trimmed per-candidate mp4 files named "
                          "<clip_id>__<start:.3f>_<end:.3f>.mp4 in this dir instead of "
                          "seeking into the full raw video (avoids transferring huge raw MP4s)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    gold_path = args.gold_csv or os.path.join(args.data_root, "gold_candidates.csv")
    df = pd.read_csv(gold_path)
    print(f"Loaded {len(df)} candidate windows total across clips: {sorted(df['clip_id'].unique())}")

    kind, model, processor = load_model(args.model, args.model_path)

    results = []
    for i, row in df.iterrows():
        clip_id = row["clip_id"]
        accel_csv = os.path.join(args.data_root, f"{clip_id}_HERO10 Black-ACCL.csv")
        gyro_csv = os.path.join(args.data_root, f"{clip_id}_HERO10 Black-GYRO.csv")

        orig_start, orig_end = get_clip_bounds(row)
        if args.clips_dir:
            video_path = os.path.join(
                args.clips_dir, f"{clip_id}__{orig_start:.3f}_{orig_end:.3f}.mp4")
            start, end = get_pretrimmed_bounds(row)
        else:
            video_path = os.path.join(args.data_root, f"{clip_id}.MP4")
            start, end = orig_start, orig_end
        try:
            frames = sample_frames(video_path, start, end)
            prompt_text = build_prompt(args.condition, row, accel_csv, gyro_csv)
            output_text = run_qwen_generation(model, processor, frames, prompt_text)
        except Exception as e:
            print(f"[event {i}] ERROR: {e}")
            results.append({
                "event_idx": int(i), "clip_id": clip_id,
                "start_time": orig_start, "end_time": orig_end,
                "error": str(e),
            })
            continue

        parsed = extract_json(output_text)
        rec = {
            "event_idx": int(i), "clip_id": clip_id,
            "start_time": orig_start, "end_time": orig_end,
            "gold_decision": row.get("decision", ""),
            "raw_output": output_text,
        }
        if parsed is not None and "is_evasive_event" in parsed:
            rec["parsed"] = True
            rec["pred_is_evasive"] = bool(parsed.get("is_evasive_event"))
            rec["pred_confidence"] = parsed.get("confidence", None)
            rec["pred_reasoning"] = parsed.get("reasoning", "")
        else:
            rec["parsed"] = False
            rec["pred_is_evasive"] = None
            rec["pred_confidence"] = None
            rec["pred_reasoning"] = ""
        results.append(rec)
        print(f"[{i+1}/{len(df)}] clip={clip_id} start={orig_start:.2f} gold={rec['gold_decision']} pred={rec['pred_is_evasive']} parsed={rec['parsed']}")

    out_path = os.path.join(args.out_dir, f"motor__{args.model}__{args.condition}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
