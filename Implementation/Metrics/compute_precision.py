"""
compute_precision.py
---------------------
Reads outputs/retrieval_results.csv and computes Precision@k using the
manually-annotated `human_relevant` column (expected values: "1" / "0";
blank rows have not yet been annotated and are skipped).

For each event_id (one retrieval query), Precision@k is:
    (# of relevant clauses among the top-k retrieved) / k

The overall Precision@k reported is the mean across all annotated events.

Usage:
    python3 compute_precision.py --k 3 \
        --input outputs/retrieval_results.csv \
        --output outputs/precision_at_k.csv
"""

import argparse
import csv
import os
from collections import defaultdict

from config import OUTPUT_DIR


def _to_int(x, default=None):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def _to_bool_relevance(x):
    """Returns True/False if annotated, None if blank/unannotated."""
    if x is None:
        return None
    x = str(x).strip()
    if x == "":
        return None
    return x in ("1", "true", "True", "yes", "Y", "y")


def load_retrieval_results(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["_rank"] = _to_int(row.get("retrieval_rank"), None)
            row["_relevant"] = _to_bool_relevance(row.get("human_relevant"))
            rows.append(row)
    return rows


def compute_precision_at_k(rows, k):
    by_event = defaultdict(list)
    for r in rows:
        by_event[r["event_id"]].append(r)

    per_event_precision = {}
    for event_id, cands in by_event.items():
        annotated = [c for c in cands if c["_relevant"] is not None
                     and c["_rank"] is not None and c["_rank"] <= k]
        if not annotated:
            continue  # nothing annotated within top-k for this event yet
        relevant_count = sum(1 for c in annotated if c["_relevant"])
        # Precision@k is defined over k, not just the annotated subset, but we
        # only have ground truth for annotated rows -> report on what's annotated
        # and note how many of the top-k slots were actually annotated.
        per_event_precision[event_id] = {
            "relevant_in_topk": relevant_count,
            "annotated_in_topk": len(annotated),
            "k": k,
            "precision_at_k": round(relevant_count / k, 4),
        }
    return per_event_precision


def write_report(per_event_precision, path, k):
    fieldnames = ["event_id", "k", "relevant_in_topk", "annotated_in_topk", "precision_at_k"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for event_id, stats in per_event_precision.items():
            writer.writerow({"event_id": event_id, **stats})
    if per_event_precision:
        mean_p = sum(v["precision_at_k"] for v in per_event_precision.values()) / len(per_event_precision)
    else:
        mean_p = 0.0
    return mean_p


def main():
    parser = argparse.ArgumentParser(description="Compute Precision@k from retrieval_results.csv")
    parser.add_argument("--k", type=int, default=3, help="k for Precision@k")
    parser.add_argument("--input", type=str,
                         default=os.path.join(OUTPUT_DIR, "retrieval_results.csv"))
    parser.add_argument("--output", type=str,
                         default=os.path.join(OUTPUT_DIR, "precision_at_k.csv"))
    args = parser.parse_args()

    rows = load_retrieval_results(args.input)
    per_event = compute_precision_at_k(rows, args.k)
    mean_p = write_report(per_event, args.output, args.k)

    print(f"[OK] Precision@{args.k} computed for {len(per_event)} annotated event(s)")
    print(f"     Mean Precision@{args.k} = {mean_p:.4f}")
    print(f"     Per-event detail -> {args.output}")


if __name__ == "__main__":
    main()