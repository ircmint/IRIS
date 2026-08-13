# Fine-Tuning — R1 / Qwen2VL

QLoRA fine-tuning of Qwen2-VL-7B on Rider1_NJ pilot data.

## Files

| File | Purpose |
|------|---------|
| `qwen2_rider1_train.py` | Training script — Qwen2-VL-7B + QLoRA, 5 epochs |
| `qwen2_rider1_train_7ep.py` | Same but 7 epochs (extended run) |
| `evaluate_rider1.py` | Evaluate trained model on Rider1_NJ held-out set |
| `evaluate_rider1_7ep.py` | Evaluation for 7-epoch checkpoint |

## Training

```bash
source /ssd_scratch/abhishek.vedula/envs/qwen2vl/bin/activate
# Submit via SLURM
sbatch run_qwen2_rider1.sbatch

# Or directly on GPU node
CUDA_VISIBLE_DEVICES=0 python qwen2_rider1_train.py 2>&1 | tee train_r1.log
```

## Checkpoint

Saved to: `/ssd_scratch/abhishek.vedula/runs/qwen2vl_rider1_qlora/`

## Model

**Qwen2-VL-7B-Instruct** — 4-bit nf4 quantization (BitsAndBytes)
- Handles multi-image and video input natively via `qwen_vl_utils.process_vision_info`
- LoRA applied to `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
