# IRC Compliance Pipeline for 2-Wheeler Video Datasets

Checks dashcam/2-wheeler video frames against **IRC:35-2015 (Road Markings)**
and **IRC:67-2022 (Road Signs)** — each treated as a completely separate
knowledge base — flags non-compliant frames, cites the exact clause, builds
a knowledge graph of the whole run, and now produces every intermediate
output needed to compute retrieval/compliance evaluation metrics
(**Precision@k, ICDI, CRRS, IDAS, CHR**).

## What's real vs. heuristic (read this first)

| Component | How it works | Status |
|---|---|---|
| IRC clause extraction | `pdfplumber` parses your actual PDFs, splits on real clause numbering (e.g. `4.6`, `11.6.2`), keeps only the sections you specified | ✅ Real, verified against your PDFs |
| Compliance→clause citation | TF-IDF retrieval scoped to the correct PDF's KB only | ✅ Real, tested — e.g. correctly cited clause **4.4.2** (warning lines) and **15.57** ("Cattle Crossing") on your sample frame |
| Lane/marking detection | Classical CV: HSV colour masks + connected-component elongation filter + Hough lines, restricted to a road-surface ROI | ⚠️ Heuristic triage, not a trained segmentation model |
| Sign detection | Classical CV: colour + shape (circle/triangle/rectangle) classification mapped to IRC:67 §3 categories | ⚠️ Heuristic; will also pick up shop hoardings near the carriageway (itself flagged under IRC:67 §4/§11 sign-siting/visibility rules) |
| Vehicles/pedestrians/animals | Pretrained YOLOv8n (COCO-80) | ✅ Real pretrained model, off-the-shelf classes only |
| `absence_confidence` | Heuristic, derived per-rule from the actual detector signal (e.g. marking coverage ratio) at the moment the rule fires | ⚠️ Heuristic, documented per-rule in `compliance_engine.py` |
| `zone_type` | Heuristic guess from `object_detector.py` counts only (no dedicated zone/scene classifier exists yet) — see `infer_zone_type()` in `compliance_engine.py` | ⚠️ Heuristic — replace with a real classifier or GPS/map-matched zone metadata when available |

**Upgrade path:** every detector is an isolated module (`marking_detector.py`,
`sign_shape_detector.py`, `object_detector.py`) with a documented seam for
swapping in a fine-tuned segmentation/detection model once you have labelled
Indian road-marking/sign data — no other code changes needed.

## Sections actually loaded (as you specified)

- **IRC:35-2015** → Sections 3, 4, 6.1, 6.2, 7, 8, 11
- **IRC:67-2022** → Sections 3, 11, 13, 14, 15, 16, 17, 24, 25, 26

(Section→page ranges were derived by scanning the real section headers in
your two PDFs — see `config.py`. If IRC:88-2010 is added later, add its
`pdf_path`/`section_pages`/`requested_sections` block to `IRC_DOCS` in
`config.py` — the rest of the pipeline needs no changes since each doc is
already processed independently.)

## Architecture

```
                         ┌─────────────────────┐
                         │  IRC35_2015.pdf      │──┐
                         │  IRC67_2022.pdf      │──┤  irc_kb_builder.py
                         └─────────────────────┘  │  (separate KB per PDF)
                                                    ▼
video/frames ──► main_pipeline.py ──► marking_detector.py ─┐
                       │              sign_shape_detector.py├─► compliance_engine.py ──► violations + cited clauses
                       │              object_detector.py    ┘        │  (+ retrieval_score, absence_confidence,
                       │                                              │   zone_type, severity, infrastructure_element)
                       │                                              ▼
                       ├──────────────────────────────────► knowledge_graph.py ──► GraphML / JSON / interactive HTML
                       └──────────────────────────────────► visualize.py ──► annotated frame images

outputs/frame_reports.csv ──┐
outputs/retrieval_results.csv├──► merge_events.py (+ events.csv from IMU pipeline) ──► outputs/event_compliance.csv
outputs/verdicts.csv ────────┘                                                              │
                                                                                              ▼
                                                                                     compute_icdi.py ──► event_icdi.csv
                                                                                              │
                                                                     ┌────────────────────────┼────────────────────────┐
                                                                     ▼                         ▼                        
                                                          compute_crrs.py ──► route_risk.csv   compute_idas.py ──► idas.csv

outputs/retrieval_results.csv ──► compute_precision.py ──► precision_at_k.csv
outputs/retrieval_results.csv + outputs/verdicts.csv ──► compute_chr.py ──► chr_report.csv
```

## Setup

```bash
pip install -r requirements.txt
```

(First run downloads the small pretrained `yolov8n.pt` COCO weights, ~6MB.)

## Usage

```bash
# 1. Put your PDFs at the paths configured in config.py.

# 2. Full video dataset:
python3 main_pipeline.py --video /path/to/ride.mp4 --sample-every 15 --video-name "ride_2026_07_22"

# 3. Single test frame:
python3 main_pipeline.py --image /path/to/frame.jpg

# 4. A folder of pre-extracted frames:
python3 main_pipeline.py --image-dir /path/to/frames_folder

# 5. Merge with your IMU-pipeline events.csv (clip_id, start_time, end_time,
#    decision, event_type, peak_confidence):
python3 merge_events.py --events events.csv \
    --frame-reports outputs/frame_reports.csv \
    --output outputs/event_compliance.csv

# 6. Run the evaluation scripts (Part 5), in dependency order:
python3 compute_icdi.py         # event_compliance.csv      -> event_icdi.csv
python3 compute_crrs.py         # event_icdi.csv            -> route_risk.csv
python3 compute_idas.py         # event_icdi.csv            -> idas.csv
python3 compute_chr.py          # retrieval_results.csv + verdicts.csv -> chr_report.csv

# 7. Precision@k needs a manual annotation pass first (fill in the blank
#    `human_relevant` column in outputs/retrieval_results.csv with 1/0), then:
python3 compute_precision.py --k 3
```

`--sample-every N` controls how many raw video frames are skipped between
checks (15 ≈ every 0.5s at 30fps). Lower it for denser checking, raise it for
faster runs on long rides.

## Outputs (in `outputs/` and `vis/`)

### Existing (extended, Part 1)

- `frame_reports.json` — full per-frame detection + compliance detail
- `frame_reports.csv` — flattened one-row-per-violation table. **New columns
  appended** (nothing removed):

  | Column | Meaning |
  |---|---|
  | `retrieval_score` | TF-IDF cosine similarity of the top-1 cited clause |
  | `retrieval_rank` | rank of the cited clause among retrieved candidates (always 1 for the citation actually used) |
  | `query_text` | exact text used for retrieval |
  | `absence_confidence` | heuristic [0–1] confidence the required infrastructure is actually absent (see per-rule logic in `compliance_engine.py`) |
  | `zone_type` | `School` / `Residential` / `Highway` / `Pedestrian Crossing` / `Unknown` — heuristic, see above |
  | `infrastructure_element` | e.g. `Lane Marking`, `Warning Sign`, `Regulatory Sign`, `Informatory Sign` |
  | `severity` | `1`=LOW, `2`=MEDIUM, `3`=HIGH, `4`=CRITICAL |

- `irc_knowledge_graph.html` / `.graphml` / `.json` — knowledge graph exports (unchanged)
- `vis/annotated_<frame_id>.jpg` — bounding boxes + verdict overlay (unchanged)

### New (Part 2–5)

| File | Produced by | Columns | Purpose |
|---|---|---|---|
| `retrieval_results.csv` | `main_pipeline.py` | `event_id, frame_id, query_text, retrieval_rank, clause_id, irc_code, similarity_score, retrieved_clause_text, human_relevant` (blank, for manual annotation) | Precision@k |
| `verdicts.csv` | `main_pipeline.py` | `event_id, frame_id, verdict, cited_clause, reasoning, retrieved_clause, retrieval_score` | CHR / RHR |
| `event_compliance.csv` | `merge_events.py` | `event_id, start_time, end_time, frame_id, timestamp, rule, clause_id, retrieval_score, absence_confidence, severity, zone_type` | Input to ICDI |
| `event_icdi.csv` | `compute_icdi.py` | `event_id, start_time, end_time, zone_type, zone_risk, num_rows, mean_absence_confidence, mean_severity, mean_retrieval_score, icdi` | Input to CRRS/IDAS |
| `route_risk.csv` | `compute_crrs.py` | `event_id, group_id, start_time, end_time, icdi, frequency_weight, weighted_icdi` (+ trailing `ROUTE_TOTAL` row) | CRRS |
| `idas.csv` | `compute_idas.py` | `event_id, zone_type, mean_absence_confidence, infra_status, infra_weight, icdi, idas_score` (+ trailing `IDAS_TOTAL` row) | IDAS |
| `precision_at_k.csv` | `compute_precision.py` | `event_id, k, relevant_in_topk, annotated_in_topk, precision_at_k` | Precision@k detail |
| `chr_report.csv` | `compute_chr.py` | `event_id, frame_id, cited_clause, hallucinated, reason` (+ trailing `CHR_TOTAL` row) | CHR |

`event_id` note: `retrieval_results.csv`/`verdicts.csv`/`frame_reports.csv`
share a **violation-level** `event_id` of the form `<frame_id>_v<index>`
(assigned in `main_pipeline.py`). `event_compliance.csv` onward uses the
**IMU-event-level** `event_id` (`<clip_id>_evt<index>`, generated in
`merge_events.py` since your `events.csv` has no id column of its own).
These are two different, non-interchangeable keys — don't join across them.

## Evaluation scripts — formulas & assumptions

- **Precision@k** (`compute_precision.py`): mean, over annotated queries, of
  (relevant clauses in top-k) / k, using the `human_relevant` column you fill
  in on `retrieval_results.csv`.
- **ICDI** (`compute_icdi.py`): `absence_confidence × severity × zone_risk ×
  retrieval_score`, computed per `event_compliance.csv` row then averaged per
  event. Zone risk: School=1.5, Pedestrian Crossing=1.4, Residential=1.2,
  Highway=1.0, Unknown=1.0.
- **CRRS** (`compute_crrs.py`): `Σ(ICDI × frequency_weight) / number_of_events`.
  Events are clustered by time proximity (`--gap`, default 30s) to detect
  recurring hazards; `frequency_weight` = size of the cluster an event
  belongs to. **This clustering/weighting rule is a modelling choice** (the
  spec didn't pin one down) — adjust `--gap` or the weighting logic in
  `compute_crrs.py` if your definition differs.
- **IDAS** (`compute_idas.py`): each event's `mean_absence_confidence` is
  bucketed into Absent (≥0.66) / Unclear (0.33–0.66) / Present (<0.33), each
  with weight 1.0 / 0.5 / 0.1 respectively; `idas_score = icdi ×
  infrastructure_weight`, averaged for the overall IDAS.
- **CHR** (`compute_chr.py`): a citation (on a `NON_COMPLIANT` verdict row) is
  hallucinated if its clause wasn't among that event's retrieved candidates
  in `retrieval_results.csv`, OR the clause_id doesn't exist in any
  `kb/*_kb.json` file at all.

## Knowledge graph schema

```
Video -[HAS_FRAME]-> Frame -[HAS_DETECTION]-> Detection
Detection -[RESULTS_IN]-> Violation -[CITES]-> Clause -[PART_OF]-> Section -[PART_OF]-> IRCDocument
Detection -[COMPLIES_WITH]-> Compliant
```

`IRC35_2015` and `IRC67_2022` remain separate `IRCDocument` nodes with their
own `Section`/`Clause` subtrees — they are never merged, per your requirement
that each PDF be treated separately.

## Files

| File | Purpose | Status |
|---|---|---|
| `config.py` | Paths, per-PDF section→page mapping, requested sections | unchanged |
| `irc_kb_builder.py` | Parses each PDF into a clause-level JSON KB | unchanged |
| `marking_detector.py` | IRC:35-relevant marking detection (classical CV) | unchanged |
| `sign_shape_detector.py` | IRC:67-relevant sign shape/condition detection (classical CV) | unchanged |
| `object_detector.py` | YOLOv8n wrapper for vehicles/pedestrians/animals | unchanged |
| `compliance_engine.py` | Rule set + TF-IDF clause citation | **extended**: retrieval_score/rank, query_text, absence_confidence, zone_type, infrastructure_element, severity |
| `knowledge_graph.py` | Builds/exports the KG | unchanged |
| `visualize.py` | Annotated frame rendering | unchanged |
| `main_pipeline.py` | CLI orchestrator | **extended**: writes `retrieval_results.csv` + `verdicts.csv`, extra `frame_reports.csv` columns |
| `merge_events.py` | **new** — merges IMU `events.csv` with `frame_reports.csv` → `event_compliance.csv` |
| `compute_precision.py` | **new** — Precision@k from `retrieval_results.csv` |
| `compute_icdi.py` | **new** — ICDI from `event_compliance.csv` |
| `compute_crrs.py` | **new** — CRRS from `event_icdi.csv` |
| `compute_idas.py` | **new** — IDAS from `event_icdi.csv` |
| `compute_chr.py` | **new** — CHR from `retrieval_results.csv` + `verdicts.csv` |

## Backward compatibility

- All existing CLI flags/behaviour of `main_pipeline.py` and
  `compliance_engine.py` are unchanged — only new keys/columns were added.
- `knowledge_graph.py`, `marking_detector.py`, `sign_shape_detector.py`, and
  `object_detector.py` were **not modified**.
- Existing `frame_reports.json`/`.csv` consumers keep working; the new CSV
  columns are appended at the end, not inserted in the middle.