#!/bin/bash
# run_all_metrics.sh -- full metrics pipeline, run on gnode025 (needs GPU + checkpoint).
set -e

echo "=== 1. Building kb/*.json and gold_clause_mapping.json ==="
python build_kb_and_mapping.py

echo "=== 2. Running inference on held-out val split ==="
python run_inference.py --checkpoint ~/tagnet/checkpoints_scoped/epoch_7

echo "=== 3. Standard metrics: Accuracy, Macro-F1, BLEU, ROUGE-L, CIDEr ==="
python compute_standard_metrics.py

echo "=== 4. Domain metrics: CHR ==="
python compute_chr.py

echo "=== 5. Domain metrics: RHR (NLI entailment, loads a model) ==="
python compute_rhr.py --device cuda

echo "=== 6. Domain metrics: HAA (uses RHR just computed) ==="
python compute_haa.py

echo "=== 7. Domain metrics: ICDI ==="
python compute_icdi.py

echo "=== 8. Domain metrics: IDAS ==="
python compute_idas.py

echo "=== 9. Domain metrics: CRRS ==="
python compute_crrs.py

echo "=== 10. Domain metrics: Precision@k (will report 0 annotated -- see caveat) ==="
python compute_precision.py --k 3

echo "=== 11. Qualitative report (visual, 3-fold boxes) ==="
python qualitative_report.py --n 6

echo "=== DONE. See outputs/ for all CSVs + standard_metrics.json + qualitative_report.png ==="
