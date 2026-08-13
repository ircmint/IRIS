# Zero-Shot — R1 / PaliGemma

Zero-shot inference using PaliGemma 3B and PaliGemma2 on Rider1_NJ data.

## Files

| File | Purpose |
|------|---------|
| `paligemma_full_run.py` | Full zero-shot run using PaliGemma-3B |
| `paligemma2_full_run.py` | Full zero-shot run using PaliGemma2-3B (improved version) |
| `paligemma_control_test.py` | Control test — sanity check model loading and basic inference |

## Usage

```bash
source /ssd_scratch/abhishek.vedula/envs/paligemma/bin/activate

# PaliGemma
python paligemma_full_run.py

# PaliGemma2
python paligemma2_full_run.py
```

## Notes

- PaliGemma uses a different image preprocessing pipeline than Qwen2VL/InternVL
- Requires separate conda env: `paligemma` (JAX-based) or `torch` variant
- Outputs CSV with `event_id`, `model_output`, `predicted_label`, `confidence`
