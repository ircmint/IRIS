# Fine-Tuning — R1 / LLaVA

Fine-tuning of LLaVA on Rider1_NJ pilot data.

## Files

| File | Purpose |
|------|---------|
| `FineTune.py` | LLaVA fine-tuning script |
| `Evaluate.py` | Evaluation on R1 dataset |

## Notes

- LLaVA uses a CLIP visual encoder + LLaMA/Vicuna language model
- Fine-tuned on conversation-style QA format matching the evasive action prompts
- Less competitive than InternVL3/Qwen2VL in this domain
