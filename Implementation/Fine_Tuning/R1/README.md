# Fine-Tuning — R1 (Rider1_NJ)

All fine-tuning experiments use **Rider1_NJ** gold labels (`gold_labels_v1_FROZEN.csv`) as training data.

## Model Comparison

| Model | Folder | Params | Quant | Best F1 (R1) |
|-------|--------|--------|-------|-------------|
| Qwen2-VL-7B | `Qwen2VL/` | 7B | 4-bit nf4 | — |
| InternVL2-8B | `InternVL2/` | 8B | 4-bit nf4 | — |
| InternVL3-8B | `InternVL3/` | 8B | 4-bit nf4 | **best** |
| PaliGemma-3B | `PaliGemma/` | 3B | bf16 | — |
| LLaVA | `LLaVA/` | — | — | — |
| DeepSeek-VL | `DeepSeek/` | — | — | — |

## Data

Training data: `/ssd_scratch/abhishek.vedula/Rider1_NJ/`
- `Rider1_NJ_720p.mp4` — dashcam video
- `gold_candidates.csv` — annotated events (34–40 usable rows after quality filter)

## Evaluation

After training, evaluate in-distribution (R1) with the `evaluate_rider1.py` scripts.  
Cross-rider results are in `R2/`, `R3/`, `R4/` folders.
