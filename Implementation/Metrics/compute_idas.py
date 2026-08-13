"""
compute_idas.py
------------------
Reads outputs/event_icdi.csv (produced by compute_icdi.py) and computes IDAS
(Infrastructure Deficiency / Absence Score) per event.

Each event's mean_absence_confidence is bucketed into an inferred
infrastructure status:

    mean_absence_confidence >= 0.66   -> "Absent"
    0.33 <= mean_absence_confidence < 0.66 -> "Unclear"
    mean_absence_confidence < 0.33    -> "Present"

Infrastructure Weight:
    Absent  = 1.0
    Unclear = 0.5
    Present = 0.1

    idas_score (per event) = icdi x infrastructure_weight
    IDAS (overall)          = mean(idas_score) across all events

Output: outputs/idas.csv — one row per event, plus a trailing "IDAS_TOTAL"
summary row with the overall IDAS.

Usage:
    python3 compute_idas.py --input outputs/event_icdi.csv \
        --output outputs/idas.csv
"""

import argparse
import csv
import os

from config import OUTPUT_DIR

INFRA_WEIGHT = {
    "Absent": 1.0,
    "Unclear": 0.5,
    "Present": 0.1,
}


def _to_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def infer_infra_status(mean_absence_confidence):
    if mean_absence_confidence >= 0.66:
        return "Absent"
    if mean_absence_confidence >= 0.33:
        return "Unclear"
    return "Present"


def load_event_icdi(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["_mean_ac"] = _to_float(row.get("mean_absence_confidence"), 0.0)
            row["_icdi"] = _to_float(row.get("icdi"), 0.0)
            rows.append(row)
    return rows


def compute_idas(rows):
    results = []
    for r in rows:
        status = infer_infra_status(r["_mean_ac"])
        weight = INFRA_WEIGHT[status]
        idas_score = round(r["_icdi"] * weight, 4)
        results.append({
            "event_id": r["event_id"],
            "zone_type": r.get("zone_type", ""),
            "mean_absence_confidence": round(r["_mean_ac"], 4),
            "infra_status": status,
            "infra_weight": weight,
            "icdi": round(r["_icdi"], 4),
            "idas_score": idas_score,
        })
    overall_idas = round(sum(x["idas_score"] for x in results) / len(results), 4) if results else 0.0
    return results, overall_idas


def write_idas(results, overall_idas, path):
    fieldnames = ["event_id", "zone_type", "mean_absence_confidence",
                  "infra_status", "infra_weight", "icdi", "idas_score"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        writer.writerow({
            "event_id": "IDAS_TOTAL", "zone_type": "", "mean_absence_confidence": "",
            "infra_status": "", "infra_weight": "", "icdi": "", "idas_score": overall_idas,
        })
    return path


def main():
    parser = argparse.ArgumentParser(description="Compute IDAS from event_icdi.csv")
    parser.add_argument("--input", type=str,
                         default=os.path.join(OUTPUT_DIR, "event_icdi.csv"))
    parser.add_argument("--output", type=str,
                         default=os.path.join(OUTPUT_DIR, "idas.csv"))
    args = parser.parse_args()

    rows = load_event_icdi(args.input)
    results, overall_idas = compute_idas(rows)
    out_path = write_idas(results, overall_idas, args.output)

    print(f"[OK] Computed IDAS for {len(results)} event(s)")
    print(f"     Overall IDAS = {overall_idas}")
    print(f"     -> {out_path}")


if __name__ == "__main__":
    main()