# Fine-Tuning — R2 (Rider2_AZ) — Cross-Rider Evaluation

Models trained on **R1 (Rider1_NJ)** are evaluated here on **Rider2_AZ** (riderX).  
No new training is done for R2 — only evaluation/inference.

## Files (all riders share riderX evaluation scripts)

| Folder | Script | Purpose |
|--------|--------|---------|
| `Qwen2VL/` | `qwen2_riderX_train.py` | [Optional] Retrain on riderX data for comparison |
| `Qwen2VL/` | `evaluate_riderX.py` | Evaluate R1-trained Qwen2VL on R2 |
| `InternVL3/` | `train.py`, `eval.py` | Optional retrain + evaluate on R2 |
| `PaliGemma/` | `train_riderX.py`, `evaluate_riderX.py` | Optional retrain + evaluate |

## Usage

```bash
# Evaluate R1-trained model on Rider2_AZ
python Qwen2VL/evaluate_riderX.py --rider Rider2_AZ
```

## Key Variable

In all riderX scripts, set `RIDER_NAME = "Rider2_AZ"` (or pass as CLI arg).
