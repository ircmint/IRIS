# TAGNet V2

**Backbone:** Qwen2-VL-2B-Instruct  
**Adapters:** QLoRA (r=16, α=32, 4-bit nf4) + TelemetryAdapter + ContextAdapter + **AMG** + **CGPA**  
**New in V2 vs V1:** Adaptive Modality Gating, Clause-Grounded Pointer Attention, RAG-based IRC compliance

## Architecture

```
Input:
  ├── Video frame(s)           → VisionEncoder → visual tokens
  ├── IMU telemetry vector     → TelemetryAdapter → telemetry token
  ├── GPS / context vector     → ContextAdapter  → context token
  └── IRC clause embeddings    → ClauseEncoder   → clause tokens (retrieved via RAG)

Adaptive Modality Gate (AMG):
  Learns a gate weight g ∈ [0,1] per event:
    merged = g * telemetry_token + (1-g) * visual_summary
  Prevents telemetry from dominating on visual-rich events and vice versa.

Clause-Grounded Pointer Attention (CGPA):
  Given that a clause applies (presence == "Yes" from generated JSON),
  CGPA resolves WHICH clause using pointer attention over retrieved IRC clauses.
  Only invoked for positive events — avoids the 56:1 class imbalance in all events.

Backbone (QLoRA):
  [tel_token | ctx_token | AMG_token | visual_tokens | text_tokens]
  └── Qwen2-VL-2B language model → 3-fold JSON output:
        {"evasive_action": ..., "road_compliance": ..., "infrastructure": ...}
```

## Key Differences from V1

| Aspect | TAGNet V1 | TAGNet V2 |
|--------|-----------|-----------|
| Backbone | Qwen2.5-VL-3B | Qwen2-VL-2B (smaller but better zero-shot) |
| Modality fusion | fixed prepend | **Adaptive Modality Gate** (learned) |
| Clause citation | prompt-based | **CGPA** (pointer attention) |
| RAG | ChromaDB (KG retrieval) | ChromaDB + **plate_index** + **multimodal index** |
| Output format | free text | structured 3-fold JSON |
| New metrics | HCAS, CHR, IDAS | + CRRS, RHR, CiDEr, HCAS-silver |

## Files

### Core
| File | Purpose |
|------|---------|
| `tagnet_v2_final.py` | **Main architecture file** — all modules: AMG, CGPA, ClauseEncoder, HCAS metric |
| `train_tagnet_v2.py` | Training script with presence-gated pointer_loss + oversampling |
| `build_v2_dataset.py` | Build `v2_dataset.json` from gold labels + RAG clause retrieval |
| `draw_tagnet_v2_architecture.py` | Generates the architecture diagram PNG |
| `tagnet_v2_architecture.png` | Architecture diagram |

### RAG Pipeline
| File | Purpose |
|------|---------|
| `RAG/01_ingest.py` | Ingest IRC-35/IRC-67 PDFs → `clauses.jsonl` |
| `RAG/02_build_indexes.py` | Build ChromaDB text index from clauses |
| `RAG/03_build_multimodal_index.py` | Build multimodal index (plate images + text) |
| `RAG/04_query_multimodal.py` | Query interface for the multimodal index |
| `RAG/05_detector.py` | Road sign / plate detector for multimodal index |
| `RAG/clauses.jsonl` | Pre-built clause database (IRC-35 + IRC-67) |

### Ablations
| Folder | Description |
|--------|-------------|
| `Ablations/amg_only/` | V2 without CGPA tests AMG contribution alone |
| `Ablations/cgpa_only/` | V2 without AMG  tests CGPA contribution alone |
| `Ablations/Qwen25_3B/` | Full V2 ablations on Qwen2.5-VL-3B backbone (vs 2B) |

### Metrics
All novel V2 evaluation metrics are in `Metrics/`:

| Metric | File | Description |
|--------|------|-------------|
| HCAS | `compute_hcas.py` | Hierarchical Clause Alignment Score |
| CHR | `compute_chr.py` | Clause Hit Rate |
| IDAS | `compute_idas.py` | Infrastructure Damage Assessment Score |
| CRRS | `compute_crrs.py` | Clause Retrieval Relevance Score |
| RHR | `compute_rhr.py` | Road Hazard Recognition score |
| HAA | `compute_haa.py` | Hazard Awareness Accuracy |
| Standard | `compute_standard_metrics.py` | Precision/Recall/F1, Accuracy |
| CiDEr | `cider_scorer.py` | Caption consensus metric |

### QualQuant
Per-rider results and visualization in `QualQuant/`:
- `R1_v2_results.jsonl`, `R2_v2_results.jsonl`, ...  inference outputs per rider
- `render_panel.py` — generate qualitative panels (frame + prediction + clause citation)
- `compute_v2_metrics.py` — aggregate all metrics across riders

## Training

```bash
# Set paths first
export IRASTE_SCRATCH=/scratch/<your_username>
export IRASTE_HOME=/home/<your_username>

# Build dataset (needs ChromaDB index to exist first)
python build_v2_dataset.py


# Or smoke-test locally
python train_tagnet_v2.py --smoke_test --adapter_mode embed
```


CGPA is trained with **presence gating**: pointer_loss is only computed on rows where the gold JSON says `"presence": "Yes"`. This removes the imbalanced binary decision from CGPA's task  it only learns *which* clause applies, given that a positive was already predicted by the LM head.
