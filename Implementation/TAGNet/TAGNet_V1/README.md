# TAGNet V1

**Backbone:** Qwen2.5-VL-3B-Instruct  
**Adapters:** QLoRA (r=16, α=32, dropout=0.05) + TelemetryAdapter + ContextAdapter  
**Adapter mode:** `embed` — telemetry/context tokens prepended to input embeddings

## Files

| File | Purpose |
|------|---------|
| `rider_pipeline.py` | **Main inference script** — supports ZS, FT, and V1 modes for R1–R4 |
| `build_rider_datasets.py` | Build per-rider JSONL dataset files for training and evaluation |
| `rider_eval_full.py` | Full evaluation: loads checkpoint, runs all events, writes JSONL results |
| `rider_metrics.py` | Compute precision/recall/F1 from JSONL results |
| `rider_nlp_metrics.py` | Compute NLP metrics (BLEU, ROUGE, BERTScore) on generated descriptions |
| `motor_qwen25_train.py` | QLoRA training script for Qwen2.5-VL-3B on motor/custom datasets |
| `motor_qwen25_eval.py` | Evaluation script after Qwen2.5 fine-tuning |

## Core TAGNet Modules (on Ada Server)

The following files define the TAGNet V1 architecture and are located at `/home2/abhishek.vedula/tagnet/` on the Ada HPC server:

| File | Role |
|------|------|
| `tagnet_vlm.py` | `TAGNetVLMConfig` dataclass + `TAGNetVLM` model class (TelemetryAdapter, ContextAdapter) |
| `train_tagnet_vlm.py` | `TagNetVLMForTraining` wrapper — training loop, loss, optimizer |
| `Dataset/build_tagnet_dataset.py` | JSONL dataset preparation from gold_labels CSV + telemetry |

## Running Inference (Ada)

```bash
# SSH to Ada gnode050 (GPU node with A100)
srun --partition=long --gres=gpu:1 --pty bash

source /ssd_scratch/abhishek.vedula/envs/qwen2vl/bin/activate
cd /ssd_scratch/abhishek.vedula

# Run V1 inference for Rider1
nohup python rider_pipeline.py --rider R1 --model V1 \
    > /ssd_scratch/abhishek.vedula/Rider1_NJ/log_v1.txt 2>&1 &

# Run for all riders
for rider in R1 R2 R3 R4; do
    python rider_pipeline.py --rider $rider --model V1
done
```

## Training (Qwen2.5-VL-3B — Motor Dataset)

```bash
source /ssd_scratch/abhishek.vedula/envs/qwen2vl/bin/activate
PYTHONUNBUFFERED=1 HF_HOME=/ssd_scratch/abhishek.vedula/hf_cache \
    CUDA_VISIBLE_DEVICES=0 python motor_qwen25_train.py 2>&1 | tee motor_qwen25_train.log
```

## Checkpoint Details

| Parameter | Value |
|-----------|-------|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | language_model attention + MLP (vision tower frozen) |
| Quantization | bf16 (no 4-bit; 3B model fits in ~7 GB) |
| Epochs | 4 (final checkpoint) |
| Checkpoint path | `/home2/abhishek.vedula/tagnet/checkpoints_v1_embed_final/epoch_4` |

## TelemetryAdapter

```python
# 2-layer MLP: telemetry_dim (6: ax,ay,az,gx,gy,gz) → hidden_dim → model embed_dim
TelemetryAdapter(in_dim=6, hidden_dim=256, out_dim=model.embed_dim)
```

## ContextAdapter

```python
# Encodes GPS context: speed_kmh, zone_one_hot, lat, lon
ContextAdapter(in_dim=6, hidden_dim=128, out_dim=model.embed_dim)
```
