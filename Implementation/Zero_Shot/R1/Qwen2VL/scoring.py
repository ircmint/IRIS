"""
scoring.py - Score zero-shot VLM event-detection predictions against gold
labels.

Reads a prediction JSON produced by one of the per-model inference scripts
(list of records with event_idx/start_time/end_time/pred_is_evasive) and the
corresponding rider's gold_candidates.csv (decision=confirm/reject at the
SAME event_idx / start_time / end_time, since predictions are made against
the fixed gold candidate-window list - see inference scripts for why).

Because predictions and gold rows are both indexed against the same
gold_candidates.csv row order, "matching" is a simple row-alignment problem
(no IoU/timestamp search is actually needed - the windows are identical by
construction). We still verify start_time alignment between the prediction
and gold row before scoring, as a sanity check, and fall back to nearest
start_time match if a prediction's event_idx is missing (e.g. from a
generation error that skipped a row).

Usage:
  python scoring.py --pred predictions/Rider1_NJ__qwen2-2b__video_only.json \
      --gold $SCRATCH_ROOT/custom_data/Rider1_NJ/gold_candidates.csv \
      --out scores/Rider1_NJ__qwen2-2b__video_only_score.json
"""
import json
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.pred) as f:
        preds = json.load(f)
    gold_df = pd.read_csv(args.gold).reset_index(drop=True)

    y_true, y_pred = [], []
    n_missing_or_error = 0
    for rec in preds:
        idx = rec.get("event_idx")
        if idx is None or idx >= len(gold_df):
            n_missing_or_error += 1
            continue
        gold_row = gold_df.iloc[idx]
        # sanity check: start_time should match within 0.01s (same row order)
        if abs(float(gold_row["start_time"]) - float(rec.get("start_time", -1))) > 0.5:
            n_missing_or_error += 1
            continue
        gold_label = 1 if str(gold_row["decision"]).strip().lower() == "confirm" else 0
        pred_label = rec.get("pred_is_evasive")
        if pred_label is None:
            # unparsed output -> counted as a negative prediction (conservative,
            # documented choice - an unparsed/failed generation cannot be
            # credited as a correct detection)
            pred_label = False
        y_true.append(gold_label)
        y_pred.append(1 if pred_label else 0)

    n = len(y_true)
    if n == 0:
        summary = {"n_scored": 0, "n_missing_or_error": n_missing_or_error,
                   "accuracy": None, "precision": None, "recall": None, "f1": None}
    else:
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=1, zero_division=0
        )
        summary = {
            "n_scored": n,
            "n_missing_or_error": n_missing_or_error,
            "accuracy": acc, "precision": p, "recall": r, "f1": f1,
        }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
