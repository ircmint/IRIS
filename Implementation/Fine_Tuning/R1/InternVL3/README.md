# Fine-Tuning — R1 / InternVL3

QLoRA fine-tuning of InternVL3-8B on Rider1_NJ pilot data. This is the **best-performing model** overall.

## Files

| File | Purpose |
|------|---------|
| `train.py` | Training script — InternVL3-8B + QLoRA |
| `eval.py` | Evaluate on R1 and optionally all riders |
| `dataset.py` | `PilotEventDataset` — loads clips + gold labels for InternVL3 |

## Training

```bash
source /ssd_scratch/abhishek.vedula/envs/internvl/bin/activate
python train.py 2>&1 | tee internvl3_r1_train.log
```

## Architecture Note

InternVL3 uses a custom image tokenizer and requires:
1. `trust_remote_code=True` when loading
2. `img_context_token_id` set manually after loading
3. Pixel values normalized with ImageNet stats (mean/std)
4. Vision tower (`InternVisionModel`) excluded from LoRA via regex filter

## Checkpoint

Saved to: `/home2/abhishek.vedula/tagnet/InternVL3_adapter/best_adapter/`
