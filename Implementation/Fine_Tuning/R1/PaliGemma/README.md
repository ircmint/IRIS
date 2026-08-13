# Fine-Tuning — R1 / PaliGemma

Fine-tuning of PaliGemma-3B on Rider1_NJ pilot data.

## Files

| File | Purpose |
|------|---------|
| `FineTune_PaliGemma.py` | Main fine-tuning script |
| `train_rider1.py` | Rider1-specific training config (data paths, epochs) |
| `evaluate_rider1.py` | Evaluation on Rider1_NJ held-out set |

## Notes

- PaliGemma uses the **SigLIP** vision encoder (different from CLIP used by Qwen2VL)
- Training uses the `transformers` `PaliGemmaForConditionalGeneration` class
- Requires JAX or PyTorch backend (PyTorch preferred for this codebase)
- Smaller than InternVL3/Qwen2VL but fast inference
