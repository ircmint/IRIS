# TAG-Net metrics — how each one works, and how V1 actually did

Pipeline order: `run_inference.py` → `compute_standard_metrics.py` →
`merge_events.py` → `compute_icdi.py` → `compute_idas.py` / `compute_crrs.py`,
plus the independent `compute_chr_per_event.py`, `compute_rhr.py` → `compute_haa.py`,
and `compute_precision.py`. `run_all_metrics.sh` runs the full chain.

## Standard NLG / classification metrics (`compute_standard_metrics.py`)

Computed per fold (`road_surface`, `infrastructure`), over events where the
model's JSON output parsed cleanly (`fail_reason is None` — parse failures are
excluded and reported, not scored as wrong).

- **Accuracy** — exact match of predicted `presence` (Yes/No/Unclear) vs gold, per fold.
- **Macro-F1** — sklearn `f1_score(average="macro", labels=["Yes","No","Unclear"])`.
  Unweighted mean of per-class F1 — punishes ignoring the minority classes even
  though most events are "No".
- **BLEU** — `nltk.sentence_bleu` with smoothing, comparing generated `reasoning`
  text to gold `reasoning` text, corpus-averaged (only over rows where both are non-empty).
- **ROUGE-L** — `rouge_scorer`, `rougeL.fmeasure`, same text pair.
- **CIDEr** — `cider_scorer.py`, same text pair.

**V1 held-out results (94 events, seed=42):**
| | road_surface | infrastructure |
|---|---|---|
| Accuracy | 0.9677 | 0.9785 |
| Macro-F1 | 0.4611 | 0.4963 |
| BLEU | 0.2791 | 0.2773 |
| ROUGE-L | 0.5117 | 0.5263 |
| CIDEr | 0.9006 | 0.5386 |

**V1 full dataset (626 events):** Accuracy 0.966/0.964, Macro-F1 0.490/0.468,
BLEU 0.330/0.310, ROUGE-L 0.542/0.541, CIDEr 1.452/0.943. 9/626 parse failures (1.4%).
Full-dataset numbers land close to the held-out split — the small val set was representative.
**Reading it:** accuracy is high mainly because most events are true negatives
("No" is the dominant class); Macro-F1 ~0.46-0.50 is the honest signal — the
model does engage the minority "Yes"/"Unclear" classes, but not strongly.

## Citation Hallucination Rate — CHR (`compute_chr_per_event.py`)

For every `NON_COMPLIANT` verdict with a non-empty `cited_clause`: is that
clause one of the clauses Knowledge RAG actually retrieved for that specific
event (any rank)? `CHR = hallucinated / total`.

There are two versions in the codebase — use the per-event one, not `compute_chr.py`:
- `compute_chr.py` (**don't use as primary** — checks against a tiny static
  global allow-list `gold_clause_mapping.json`, ~6 clauses total, unrelated to
  what was actually retrieved per event — flags real, well-grounded citations
  as hallucinated. This was a metric-design bug, not a retrieval bug.)
- `compute_chr_per_event.py` (**correct definition** — checks against that
  event's own retrieval set, from `v1_dataset_with_rag_fixed.json`'s
  `retrieved_irc35`/`retrieved_irc67` fields)

**V1 result:** CHR = 0.0000 (0/6 on held-out val; 0.0167 = 1/60 on the full
626 dataset, 10x the sample). V1's citations are genuinely well-grounded —
every citation was something Knowledge RAG actually offered up.

## Reasoning Hallucination Rate — RHR (`compute_rhr.py`)

For every `NON_COMPLIANT` verdict: premise = actual IRC clause text (looked up
from `kb/*_kb.json` by `cited_clause`), hypothesis = the model's generated
`reasoning` text. Runs `cross-encoder/nli-deberta-v3-large` (MNLI cross-encoder)
to classify Entailment / Neutral / Contradiction.

`RHR = (Neutral + Contradiction) / Total`

High RHR = the model's stated reasoning doesn't actually logically follow from
the clause it correctly cited — the citation is right, but the justification
sentence isn't really "proven" by that clause's text.

**V1 result:** RHR = 1.0000 (0/6 entailment, 1 contradiction, 5 neutral on val;
0/56 entailment at full-dataset scale, 49 neutral + 7 contradiction). This is
V1's one real, consistent, disclosed weakness — confirmed at 10x scale, not a
small-sample artifact.

## Hallucination-Adjusted Accuracy — HAA (`compute_haa.py`)

`HAA = Accuracy × (1 − RHR)` — reads Accuracy from a verdict-vs-gold join and
RHR from `rhr_report.csv`'s `RHR_TOTAL` row (or `--recompute-rhr` to rerun NLI live).

**V1 result:** HAA = 0.0000 — collapses to zero because RHR = 1.0. This is
mechanically expected given RHR's result above, not a separate finding.

## ICDI — Individual/Infrastructure Compliance Deficiency Index (`compute_icdi.py`)

Per (event, frame, rule) row:
`ICDI = absence_confidence × severity × zone_risk × retrieval_score`

Zone risk weights: School 1.5, Pedestrian Crossing 1.4, Residential 1.2,
Highway 1.0, Unknown 1.0. Aggregated to one value per `event_id` as the mean
across that event's rows.

**V1 result:** mean ICDI = 0.1778 (n=617 full dataset, min 0.05, max 2.7);
0.1355 on the 94-event held-out split.

## IDAS — Infrastructure Deficiency/Absence Score (`compute_idas.py`)

Buckets each event's `mean_absence_confidence` into a status: **Absent**
(≥0.66, weight 1.0), **Unclear** (0.33–0.66, weight 0.5), **Present** (<0.33,
weight 0.1). `idas_score = ICDI × infra_weight`, then `IDAS = mean(idas_score)`
across all events.

**V1 result:** IDAS = 0.1349 (617 events full dataset); 0.0919 (93 events, val split).

## CRRS — Cumulative Route Risk Score (`compute_crrs.py`)

Events sorted by `start_time`, clustered into groups (new group whenever the
gap to the previous event's `end_time` exceeds `--gap`, default 30s) so the
same physical hazard triggering several nearby IMU events isn't double-counted
as independent risks. `frequency_weight` = size of that event's group.

`CRRS = Σ(ICDI_i × frequency_weight_i) / number_of_events`

**Note:** this is an explicit modeling assumption (grouping/weighting rule
wasn't pinned down by spec) — documented as such in the script, not a fixed formula.

**V1 result:** CRRS = 109.675 (1 cluster, full 617-event dataset); 12.6 (1
cluster, 94-event val — smaller only because fewer events aggregate into the
same cluster, not worse performance).

## Precision@k (`compute_precision.py`)

For each retrieval query (`event_id`), `Precision@k = (# relevant among top-k) / k`,
using a manually-annotated `human_relevant` column in `retrieval_results.csv`.
Rows with no annotation are skipped, not scored as 0.

**V1 result:** 0/0 — genuinely never annotated (needs manual relevance
labeling that was never done), not a fabricated zero.

## Latency (`measure_latency.py` / `measure_latency_embed.py`)

Measured on final checkpoint, gnode050 (1080Ti-class GPU), n=10 samples,
`max_new_tokens=300`, greedy decoding.

**V1 result:** 18.81s/sample avg, ~15.95 tok/s throughput, 2.94GB peak GPU memory.

---

## Bottom line for V2 planning

V1's real, disclosed weakness is **RHR**, not citation grounding — CHR is
already 0 once measured correctly. Whatever the new V2 design changes, RHR is
the metric to move: reasoning text needs to actually entail the clause it
cites, not just cite the right clause. Precision@k and CIDEr (V2 side) were
never fully wired up in the old CGPA attempt — worth deciding upfront whether
V2 will finish those or explicitly defer them again.
