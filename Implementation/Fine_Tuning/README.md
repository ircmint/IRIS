# Fine-Tuning

QLoRA fine-tuning of VLMs on IRASTE pilot data. Models are trained on R1 gold labels and evaluated on R2/R3/R4 (cross-rider generalization).

## Organization

```
Fine_Tuning/
├── R1/    — Training data rider; all model fine-tunes run here
│   ├── Qwen2VL/    — Qwen2-VL-7B fine-tuned on Rider1_NJ
│   ├── InternVL2/  — InternVL2-8B fine-tuned
│   ├── InternVL3/  — InternVL3-8B fine-tuned (best overall)
│   ├── PaliGemma/  — PaliGemma 3B fine-tuned
│   ├── LLaVA/      — LLaVA fine-tuned
│   └── DeepSeek/   — DeepSeek-VL fine-tuned
├── R2/    — riderX evaluation: model trained on R1, tested on R2
├── R3/    — riderX evaluation on R3
└── R4/    — riderX evaluation on R4
```

## Common Training Config

All models share the same LoRA config for reproducibility:

| Param | Value |
|-------|-------|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | attention + MLP layers (vision tower frozen) |
| Epochs | 5–7 |
| Batch size | 4 (gradient accumulation: 4 → effective 16) |
| Learning rate | 2e-4 |

## Rider Nomenclature

In training scripts, riders are referred to as:
- `rider1` / `Rider1_NJ` → **R1** (training rider)
- `riderX` → **R2, R3, R4** (cross-rider evaluation)

## Workflow

```bash
# 1. Train on R1
sbatch run_qwen2_rider1.sbatch     # or python qwen2_rider1_train.py

# 2. Evaluate on R1 (in-distribution)
python evaluate_rider1.py

# 3. Evaluate on R2/R3/R4 (cross-rider)
python evaluate_riderX.py --rider Rider2_AZ
python evaluate_riderX.py --rider Rider3_VA
python evaluate_riderX.py --rider Rider4_UC
```

## Results

See `Results/` folder for evaluation CSVs with per-rider, per-model metrics.
