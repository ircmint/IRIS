# TAGNet — Telemetry-Augmented Graph Network

TAGNet is a custom VLM architecture that fuses visual tokens from a pre-trained vision-language model with learned telemetry embeddings, enabling grounded evasive-action classification using both dashcam video and IMU signals.

## Architecture

```
Input:
  ├── Video frame(s)          → VLM visual encoder → visual tokens
  ├── IMU telemetry vector    → TelemetryAdapter   → telemetry token (1 token)
  └── GPS/context vector      → ContextAdapter     → context token  (1 token)

Fusion:
  [telemetry_token | context_token | visual_tokens | text_tokens]
  └── QLoRA fine-tuned language model backbone → answer

Output: structured text (action label + confidence + explanation)
```

## Versions

| Version | Backbone | New Modules | Notes |
|---------|----------|-------------|-------|
| **V1** | Qwen2.5-VL-3B-Instruct | QLoRA + TelemetryAdapter + ContextAdapter | bf16, ~7GB on RTX 2080 Ti; KG retrieval via ChromaDB |
| **V2** | Qwen2-VL-2B-Instruct | QLoRA (4-bit) + AMG + CGPA + multimodal RAG | Learned modality gating; clause pointer attention; 3-fold JSON output |

## Key Differences from Standard VLM Fine-tuning

1. **Telemetry injection**: IMU signals (ax/ay/az/gx/gy/gz) are encoded by a 2-layer MLP (`TelemetryAdapter`) and prepended as soft tokens to the input sequence
2. **Context injection**: GPS-derived features (speed, zone, location) are encoded by a separate `ContextAdapter`
3. **KG-augmented prompting**: IRC-35 and IRC-67 road safety clause retrieval via ChromaDB provides regulatory context
4. **No vision tower fine-tuning**: LoRA is applied only to the language model (attention + MLP layers); the vision encoder is frozen

## Checkpoint Paths (Ada Server)

```
/home2/abhishek.vedula/tagnet/
├── checkpoints_v1_embed_final/epoch_4/   # TAGNet V1 final checkpoint
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── adapters.pt                       # telemetry_adapter + context_adapter weights
└── InternVL3_adapter/best_adapter/       # V2 LoRA adapter
```

## Usage

See `TAGNet_V1/rider_pipeline.py` for end-to-end inference:
```bash
python rider_pipeline.py --rider R1 --model V1
python rider_pipeline.py --rider R2 --model FT
python rider_pipeline.py --rider R1 --model ZS
```
