# Implementation

All code for the IRASTE pipeline, organized by method.

## Modules

| Folder | Description |
|--------|-------------|
| `Filtering_and_Thresholding/` | IMU signal processing — extract evasive event candidates from raw telemetry |
| `TAGNet/TAGNet_V1/` | Telemetry-Augmented Graph Network V1: Qwen2.5-VL-3B + QLoRA + TelemetryAdapter |
| `TAGNet/TAGNet_V2/` | TAGNet V2: InternVL3-8B + QLoRA (cross-rider generalization focus) |
| `Zero_Shot/R1–R4/` | Zero-shot VLM inference per rider, per model family |
| `Fine_Tuning/R1–R4/` | QLoRA fine-tuning per rider, per model family |

## Pipeline Dependency Order

```
Raw pdata
    │
    ▼
Filtering_and_Thresholding  →  gold_labels.csv
    │
    ├──▶  Zero_Shot/R1–R4          (no training needed)
    │
    ├──▶  Fine_Tuning/R1–R4        (train on gold_labels, evaluate on other riders)
    │
    └──▶  TAGNet/                  (train on gold_labels + telemetry vectors)
```

## Compute Environment

All training and inference runs on **Ada HPC** (IIIT Hyderabad):
- GPU nodes: gnode025, gnode027, gnode050 (RTX 2080 Ti / A100)
- Job scheduler: SLURM (`sbatch`, `srun`)
- Scratch storage: `/ssd_scratch/abhishek.vedula/`
- Home storage: `/home2/abhishek.vedula/tagnet/`
