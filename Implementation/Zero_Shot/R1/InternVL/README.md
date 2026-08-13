# Zero-Shot — R1 / InternVL

Zero-shot inference using InternVL2/3 on Rider1_NJ pilot data.

## Files

| File | Purpose |
|------|---------|
| `InternVL_zeroshot.py` | InternVL2/3 zero-shot inference with multi-context ablation |
| `ablation.py` | Ablation runner — tests all 4 telemetry context modes across events |
| `run_server_all_models.py` | Server-side script to run all models in sequence on Ada gnode |

## Usage

```bash
source /ssd_scratch/abhishek.vedula/envs/internvl/bin/activate
export HF_HOME=/ssd_scratch/abhishek.vedula/hf_cache

python InternVL_zeroshot.py
```

## Model

**InternVL2-8B** or **InternVL3-8B** — loaded with `trust_remote_code=True`

Base path on Ada: `/ssd_scratch/abhishek.vedula/hf_cache/models--OpenGVLab--InternVL3-8B/`
