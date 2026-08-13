# Zero-Shot — R1 (Rider1_NJ)

Zero-shot inference on Rider1_NJ clips. R1 is the primary rider and was used for most ablation experiments.

## Models

| Folder | Model Family | Key Files |
|--------|-------------|-----------|
| `Qwen2VL/` | Qwen2-VL-2B/7B | `zero_shot_infer.py`, `scoring.py` |
| `InternVL/` | InternVL2/3 | `InternVL_zeroshot.py`, `ablation.py` |
| `PaliGemma/` | PaliGemma 3B, PaliGemma2 | `paligemma_full_run.py`, `paligemma2_full_run.py` |
| `Florence2/` | Florence-2 | `run_florence2.py` |
| `GroundingDINO/` | GroundingDINO | `run_grounding_dino.py` |

## Telemetry Prep

Before running zero-shot, prepare the telemetry context file:
```bash
python Qwen2VL/prep_pilot_telemetry.py
```
This generates a structured telemetry JSON used by all models for context injection.
