# Fine-Tuning R2 / Qwen2VL — Cross-Rider Evaluation

Evaluate the Qwen2-VL model trained on R1 (Rider1_NJ) on **Rider2_AZ** data.

| File | Purpose |
|------|---------|
| `qwen2_riderX_train.py` | [Optional] Retrain on riderX data for comparison experiment |
| `qwen2_riderX_train_7ep.py` | Extended 7-epoch training variant |
| `evaluate_riderX.py` | Evaluate R1-checkpoint on Rider2_AZ events |

Set `RIDER_NAME = "Rider2_AZ"` in scripts.
