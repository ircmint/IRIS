"""
Full 40-clip x 3-condition PaliGemma-3B-pt-224 run.
NOTE: control test showed this model collapses to "yes" on the
evasive-maneuver yes/no question regardless of ground truth (verified
non-degenerate on generic questions -- see pg_diag*.py). This run is being
executed anyway on the coordinator's explicit request for a real,
scored-even-if-degenerate number; the resulting metrics must be reported
alongside that caveat, not presented as a working detector.
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
import os, sys, json, argparse
import torch
from PIL import Image
import pandas as pd

sys.path.insert(0, '.')
from motor_zero_shot_infer import (
    get_clip_bounds, get_pretrimmed_bounds, sample_frames, load_raw_slice
)
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

MODEL_PATH = os.path.join(SCRATCH_ROOT, "models/paligemma-3b-pt-224")
DATA_ROOT = os.path.join(SCRATCH_ROOT, "motor_data")
CLIPS_DIR = os.path.join(DATA_ROOT, 'upload_clips')
OUT_DIR = os.path.join(DATA_ROOT, 'predictions')


def montage(frames):
    w, h = frames[0].size
    m = Image.new('RGB', (w * 2, h * 2))
    positions = [(0, 0), (w, 0), (0, h), (w, h)]
    for f, pos in zip(frames, positions):
        m.paste(f.resize((w, h)), pos)
    return m


def build_prefix_prompt(condition, row, accel_csv, gyro_csv):
    q = ("Is this two-wheeler rider performing a sudden evasive maneuver "
         "(swerve, hard brake, near-miss avoidance)? Answer yes or no.")
    prefix = ""
    if condition == "raw_telemetry":
        start, end = get_clip_bounds(row)
        accel_slice = load_raw_slice(accel_csv, start, end, max_rows=6)
        gyro_slice = load_raw_slice(gyro_csv, start, end, max_rows=6)
        prefix = (f"Accel(m/s^2): {accel_slice.strip()[:300]} "
                  f"Gyro(rad/s): {gyro_slice.strip()[:300]} ")
    elif condition == "summarized_telemetry":
        prefix = (f"Motion sensor hint: peak jerk z-score="
                  f"{row.get('peak_abs_z_jerk', 'NA')}, peak accel z-score="
                  f"{row.get('peak_abs_z_az', 'NA')}. ")
    return f"answer en {prefix}{q}"


def run(model, processor, img, prompt, max_new=20):
    inputs = processor(text=prompt, images=img, return_tensors='pt').to(model.device, model.dtype)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return processor.batch_decode(gen[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]


def parse_yesno(text):
    t = text.strip().lower()
    if t.startswith('yes') or ' yes' in t[:10]:
        return True
    if t.startswith('no') or ' no' in t[:10]:
        return False
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True,
                     choices=["video_only", "raw_telemetry", "summarized_telemetry"])
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(os.path.join(DATA_ROOT, "gold_candidates.csv"))
    print(f"Loaded {len(df)} candidates", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map={'': 0})
    model.eval()

    results = []
    for i, row in df.iterrows():
        clip_id = row["clip_id"]
        accel_csv = os.path.join(DATA_ROOT, f"{clip_id}_HERO10 Black-ACCL.csv")
        gyro_csv = os.path.join(DATA_ROOT, f"{clip_id}_HERO10 Black-GYRO.csv")
        orig_start, orig_end = get_clip_bounds(row)
        video_path = os.path.join(CLIPS_DIR, f"{clip_id}__{orig_start:.3f}_{orig_end:.3f}.mp4")
        start, end = get_pretrimmed_bounds(row)
        try:
            frames = sample_frames(video_path, start, end)
            img = montage(frames)
            prompt = build_prefix_prompt(args.condition, row, accel_csv, gyro_csv)
            output_text = run(model, processor, img, prompt)
        except Exception as e:
            print(f"[event {i}] ERROR: {e}", flush=True)
            results.append({
                "event_idx": int(i), "clip_id": clip_id,
                "start_time": orig_start, "end_time": orig_end,
                "error": str(e),
            })
            continue

        pred = parse_yesno(output_text)
        rec = {
            "event_idx": int(i), "clip_id": clip_id,
            "start_time": orig_start, "end_time": orig_end,
            "gold_decision": row.get("decision", ""),
            "raw_output": output_text,
            "parsed": pred is not None,
            "pred_is_evasive": pred,
        }
        results.append(rec)
        print(f"[{i+1}/{len(df)}] clip={clip_id} gold={rec['gold_decision']} pred={pred} raw={output_text!r}", flush=True)

    out_path = os.path.join(OUT_DIR, f"motor__paligemma-3b__{args.condition}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
