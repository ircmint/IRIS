# Fine-Tuning — R1 / DeepSeek

Fine-tuning of DeepSeek-VL on Rider1_NJ pilot data.

## Files

| File | Purpose |
|------|---------|
| `FineTune.py` | DeepSeek-VL fine-tuning script |
| `evaluate.py` | Evaluation on pilot dataset |

## Notes

- DeepSeek-VL uses a hybrid CNN + transformer visual encoder
- Competitive on dense visual reasoning; evaluated as alternative to Qwen2VL
- Requires the `deepseek_vl` package or `transformers >= 4.40`
