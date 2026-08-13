# Zero-Shot Inference

Zero-shot VLM inference for evasive action detection — no fine-tuning required. Models are prompted directly with dashcam frames and (optionally) telemetry context.

## Organization

```
Zero_Shot/
├── R1/  (Rider1_NJ — primary rider)
│   ├── Qwen2VL/       — Qwen2-VL-2B/7B zero-shot
│   ├── InternVL/      — InternVL2/3 zero-shot + ablation
│   ├── PaliGemma/     — PaliGemma 3B and PaliGemma2 zero-shot
│   ├── Florence2/     — Florence-2 object detection
│   └── GroundingDINO/ — Grounding DINO open-vocabulary detection
├── R2/  (cross-rider evaluation, riderX scripts)
├── R3/
└── R4/
```

## Models Evaluated

| Model | Family | Size | Notes |
|-------|--------|------|-------|
| Qwen2-VL-2B | Qwen | 2B | Fast; used as baseline ZS |
| Qwen2-VL-7B | Qwen | 7B | Better comprehension |
| InternVL2-8B | OpenGVLab | 8B | Strong VQA base |
| InternVL3-8B | OpenGVLab | 8B | Best zero-shot overall |
| PaliGemma-3B | Google | 3B | Lightweight; good spatial reasoning |
| PaliGemma2-3B | Google | 3B | Improved version |
| Florence-2 | Microsoft | 0.7B | Object detection focus |
| GroundingDINO | IDEA-Research | — | Open-vocab grounding |

## Prompt Strategy

Three telemetry context levels were ablated:
1. **Video-only** — frame(s) + action label prompt, no IMU context
2. **Raw telemetry** — raw ax/ay/az values appended to prompt
3. **Summarized telemetry** — natural-language summary of IMU signals
4. **Telemetry taxonomy** — structured JSON of event features

## Running Zero-Shot (R1)

```bash
# Qwen2VL
source /ssd_scratch/abhishek.vedula/envs/qwen2vl/bin/activate
python zero_shot_infer.py --rider R1 --model qwen2vl_2b

# InternVL
source /ssd_scratch/abhishek.vedula/envs/internvl/bin/activate
python InternVL_zeroshot.py
```
