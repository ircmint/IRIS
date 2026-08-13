# Zero-Shot R2 / Qwen2VL

Cross-rider zero-shot inference on Rider2_AZ using Qwen2-VL.

## Files

| File | Purpose |
|------|---------|
| `motor_zero_shot_infer.py` | Zero-shot inference — adapted from pilot script, handles GoPro-native IMU schema |
| `score_qwen.py` | Score Qwen2VL predictions against Rider2_AZ gold labels |

## Usage

```bash
# Set RIDER_NAME = "Rider2_AZ" inside the script
python motor_zero_shot_infer.py --condition video_only
python score_qwen.py --pred results_r2_video_only.json --gold gold_candidates.csv
```
