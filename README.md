# IRIS - Infrastructure-aware Multimodal Reasoning for Road Safety Auditing and Regulatory Compliance

It Is a low-cost, multimodal road-safety auditing framework designed to complement conventional manual road-safety audits. It combines road-scene video, inertial measurements, GPS, and regulatory knowledge to connect observed road and rider events with infrastructure conditions and applicable IRC provisions. The goal is to turn inexpensive mobile sensing into structured, evidence-backed audit information that can help engineers and authorities identify, verify, and prioritize road-safety interventions.

# Implementation


---

## Modules

| Folder | Description |
|--------|-------------|
| [`Preprocessing`](Preprocessing/) | IMU signal processing pipeline bandpass filter, jerk detection, candidate event extraction from raw GoPro telemetry |
| [`TAGNet/TAGNet_V1/`](TAGNet/TAGNet_V1/) | TAGNet V1: Qwen2.5-VL-3B-Instruct + QLoRA + TelemetryAdapter + ContextAdapter + KG retrieval |
| [`TAGNet/TAGNet_V2/`](TAGNet/TAGNet_V2/) | TAGNet V2: Qwen2-VL-2B-Instruct + QLoRA + Adaptive Modality Gate (AMG) + Clause-Grounded Pointer Attention (CGPA) + multimodal RAG |
| [`Zero_Shot/R1–R4/`](Zero_Shot/) | Zero-shot VLM inference, organized by rider then model family |
| [`Fine_Tuning/R1–R4/`](Fine_Tuning/) | QLoRA fine-tuning, organized by rider then model family |

---

## Pipeline

```
Raw telemetry (accel / gyro / GPS CSVs per rider)
        │
        ▼
Filtering_and_Thresholding
  ├─ step0: merge + bandpass-filter raw IMU streams
  ├─ step1: detect candidate evasive windows (z-score fusion)
  ├─ step2: human annotation → gold_labels.csv  ◄─ frozen ground truth
  └─ step3–7: ablation studies (filter params, thresholds, HITL offset)
        │
        ├──────────────────────────────────────────────────────────┐
        │                                                          │
        ▼                                                          ▼
 Zero_Shot/R1–R4/                                      Fine_Tuning/R1–R4/
  ├─ Qwen2VL     (Qwen2-VL-2B / 7B)                    ├─ Qwen2VL     train on R1
  ├─ InternVL    (InternVL3-8B)                          ├─ InternVL2/3 train on R1
  ├─ PaliGemma   (PaliGemma 3B / PaliGemma2)            ├─ PaliGemma   train on R1
  ├─ Florence2   (object detection, R1 only)             ├─ LLaVA       train on R1
  └─ GroundingDINO (open-vocab grounding, R1 only)       └─ DeepSeek    train on R1
        │                                                          │
        │                    Evaluate on R2, R3, R4 ◄─────────────┘
        │                    (cross-rider generalization)
        │
        ▼
 TAGNet/
  ├─ TAGNet_V1/  Qwen2.5-VL-3B + TelemetryAdapter + ContextAdapter
  │               train on gold_labels + raw IMU vectors
  │
  └─ TAGNet_V2/   Qwen2-VL-2B + AMG + CGPA + IRC RAG
                  train on gold_labels + telemetry + retrieved IRC clauses
                  ablations: AMG-only, CGPA-only, Qwen2.5-3B backbone
```

---

## Method Summary

### Filtering and Thresholding
Extracts evasive event candidates from raw IMU telemetry using a Butterworth bandpass filter (0.3–15 Hz) and a multi-axis z-score fusion detector. Events are reviewed by a human annotator to produce the frozen gold label set used by all downstream methods.

### Zero-Shot Inference
VLMs are prompted directly with 4 dashcam frames per candidate event window (no fine-tuning). Four telemetry context modes are ablated: video-only, raw IMU values, NL telemetry summary, and structured taxonomy JSON. Models evaluated: Qwen2-VL-2B/7B, InternVL2-8B, InternVL3-8B, PaliGemma-3B, PaliGemma2-3B, Florence-2, GroundingDINO, OWLv2.

### Fine-Tuning
QLoRA fine-tuning (rank=16, α=32, 4-bit nf4) of VLMs on Rider1_NJ gold labels (~130 events). Models are trained on R1 and evaluated on R2/R3/R4 to measure cross-rider generalization. Models fine-tuned: Qwen2-VL, InternVL2-8B, InternVL3-8B, PaliGemma-3B, LLaVA, DeepSeek-VL.

### TAGNet V1
Extends Qwen2.5-VL-3B-Instruct with two learned adapter modules:
- **TelemetryAdapter**  2-layer MLP mapping raw IMU features (ax/ay/az/gx/gy/gz) to a soft token prepended to the input sequence
- **ContextAdapter**  MLP encoding GPS context (speed, zone, location) as a second soft token

The vision tower is frozen; QLoRA is applied only to the language model (attention + MLP layers). IRC-35/67 clause retrieval via ChromaDB provides regulatory context at inference time.

### TAGNet V2
Extends Qwen2-VL-2B-Instruct with AMG and CGPA on top of V1's adapters:
- **Adaptive Modality Gate (AMG)** — learned per-event scalar g ∈ [0,1] that weights telemetry vs visual evidence; prevents IMU from dominating on visually rich events
- **Clause-Grounded Pointer Attention (CGPA)** — pointer attention over retrieved IRC clauses to resolve *which* specific clause is violated; only activated when the LM head predicts `presence="Yes"` (avoids the ~56:1 clause-positive imbalance)
- **Multimodal RAG** — ChromaDB text index + plate/sign image index over IRC-35 and IRC-67
- **3-fold JSON output**: `evasive_action`, `road_compliance`, `infrastructure` structured, parseable format used by all V2 metrics

---



## Setup

```bash
# 1. Install PyTorch with your CUDA version first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Install all other dependencies
pip install -r requirements.txt

```
