# Zero-Shot — R1 / Qwen2VL

Zero-shot inference using Qwen2-VL on Rider1_NJ pilot data.

## Files

| File | Purpose |
|------|---------|
| `zero_shot_infer.py` | Main zero-shot inference script — loads model, runs on all candidate events |
| `scoring.py` | Score predictions against gold labels (precision, recall, F1) |
| `prep_pilot_telemetry.py` | Prepare structured telemetry JSON from merged CSV for prompt injection |
| `extract_pilot_qualitative.py` | Extract qualitative descriptions from model outputs for manual review |

## Usage

```bash
source /ssd_scratch/abhishek.vedula/envs/qwen2vl/bin/activate
export HF_HOME=/ssd_scratch/abhishek.vedula/hf_cache

# Run zero-shot inference
python zero_shot_infer.py

# Score results
python scoring.py --pred results_qwen2vl_video_only.csv --gold gold_labels.csv
```

## Ablation Modes

Set `TELEMETRY_MODE` in `zero_shot_infer.py`:
- `"video_only"` — visual frames only
- `"raw_telemetry"` — raw ax/ay/az appended to prompt
- `"summarized_telemetry"` — NL summary of IMU peaks
- `"telemetry_taxonomy"` — structured event feature JSON

## Model

**Qwen2-VL-2B-Instruct** (default) or **Qwen2-VL-7B-Instruct**

Model path on Ada: `/ssd_scratch/abhishek.vedula/hf_cache/hub/models--Qwen--Qwen2-VL-2B-Instruct/`
