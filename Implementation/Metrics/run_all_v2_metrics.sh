#!/bin/bash
# run_all_v2_metrics.sh -- full V2 metrics pipeline, run on gnode027 (needs GPU + checkpoint).
# CHECKPOINT_DIR must be set by the caller (points at whichever
# checkpoints_v2_* output_dir this training run used). BEST_EPOCH is read
# from best_epoch.txt, written by train_tagnet_v2.py's early-stopping logic
# -- with early stopping, the LAST epoch saved is not necessarily the BEST
# one, so this must not be hardcoded to a specific epoch number.
set -e

: "${CHECKPOINT_DIR:?CHECKPOINT_DIR must be set, e.g. $HOME_ROOT/tagnet_V2/checkpoints_v2_8ep}"
BEST_EPOCH=$(cat "${CHECKPOINT_DIR}/best_epoch.txt" 2>/dev/null || echo "epoch_0")
echo "Using best checkpoint: ${CHECKPOINT_DIR}/${BEST_EPOCH}"

echo "=== 1. Running inference on held-out val split (real generation + presence-gated CGPA) ==="
python run_inference_v2.py --checkpoint "${CHECKPOINT_DIR}/${BEST_EPOCH}"

echo "=== 2. Rebuilding CSVs from V2 predictions ==="
python rebuild_csvs_from_v2_predictions.py --predictions predictions_v2.json
cp predictions_v2.json predictions.json   # compute_standard_metrics.py hardcodes this filename

echo "=== 3. Standard metrics: Accuracy, Macro-F1, BLEU, ROUGE-L, CIDEr ==="
python compute_standard_metrics.py

echo "=== 4. Domain metrics: CHR (per-event, corrected) ==="
python compute_chr_per_event.py --dataset v2_dataset_for_chr.json

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

echo "=== 11. NEW metric: HCAS (hierarchical clause alignment) ==="
python compute_hcas.py

echo "=== DONE. See outputs/ for all CSVs + standard_metrics.json ==="
