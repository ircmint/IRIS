# Zero-Shot — R2 (Rider2_AZ)

Zero-shot cross-rider evaluation. Models run on Rider2_AZ without any R2-specific training.

## Notes

- R2 uses the same model checkpoints as R1 (no retraining)
- Scripts reference `Rider2_AZ` data path on Ada: `/ssd_scratch/abhishek.vedula/Rider2_AZ/`
- Referred to as `riderX` in evaluation scripts

## Models

| Folder | Model | Script |
|--------|-------|--------|
| `Qwen2VL/` | Qwen2-VL-2B/7B | `motor_zero_shot_infer.py` |
| `InternVL/` | InternVL3-8B | `internvl3_full_run.py` |
| `PaliGemma/` | PaliGemma2-3B | `paligemma2_full_run.py` |

## Usage

```bash
# Change RIDER_NAME = "Rider2_AZ" in the script, then:
python Qwen2VL/motor_zero_shot_infer.py
```
