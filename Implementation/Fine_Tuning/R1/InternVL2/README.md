# Fine-Tuning — R1 / InternVL2

QLoRA fine-tuning of InternVL2-8B on Rider1_NJ pilot data.

## Files

| File | Purpose |
|------|---------|
| `FineTuning.py` | Training script for InternVL2-8B |
| `evaluate.py` | Evaluate InternVL2 on pilot dataset |

## Notes

- InternVL2 uses the same architecture as InternVL3 but an older base model
- Superseded by InternVL3 which shows better zero-shot and fine-tuned performance
- Use InternVL3 for new experiments unless you need to reproduce old results
