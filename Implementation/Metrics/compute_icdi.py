"""
compute_icdi.py
-----------------
Reads outputs/event_compliance.csv (produced by merge_events.py) and computes
the ICDI (Infrastructure Compliance Deficiency Index) per event:

    ICDI = absence_confidence x severity x zone_risk x retrieval_score

Zone risk mapping:
    School               = 1.5
    Pedestrian Crossing  = 1.4
    Residential          = 1.2
    Highway              = 1.0
    Unknown              = 1.0

event_compliance.csv has one row per (event, frame, rule) — an event can span
multiple frames and multiple rules. ICDI is first computed per row, then
aggregated to one ICDI value per event_id as the mean across its rows (the
event's "typical" deficiency severity). The per-row detail is preserved for
transparency in the raw ICDI column.

Output: outputs/event_icdi.csv with columns:
    event_id, start_time, end_time, zone_type, zone_risk, num_rows,
    mean_absence_confidence, mean_severity, mean_retrieval_score, icdi

Usage:
    python3 compute_icdi.py --input outputs/event_compliance.csv \
        --output outputs/event_icdi.csv
"""

import argparse
import csv
import os
from collections import defaultdict

from config import OUTPUT_DIR

ZONE_RISK = {
    "School": 1.5,
    "Pedestrian Crossing": 1.4,
    "Residential": 1.2,
    "Highway": 1.0,
    "Unknown": 1.0,
}


def _to_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_event_compliance(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["_absence_confidence"] = _to_float(row.get("absence_confidence"), 0.0)
            row["_severity"] = _to_float(row.get("severity"), 0.0)
            row["_retrieval_score"] = _to_float(row.get("retrieval_score"), 0.0)
            row["_zone_risk"] = ZONE_RISK.get(row.get("zone_type", "Unknown"), 1.0)
            row["_icdi"] = (row["_absence_confidence"] * row["_severity"]
                             * row["_zone_risk"] * row["_retrieval_score"])
            rows.append(row)
    return rows


def aggregate_by_event(rows):
    by_event = defaultdict(list)
    for r in rows:
        by_event[r["event_id"]].append(r)

    results = []
    for event_id, ev_rows in by_event.items():
        n = len(ev_rows)
        mean_ac = sum(r["_absence_confidence"] for r in ev_rows) / n
        mean_sev = sum(r["_severity"] for r in ev_rows) / n
        mean_score = sum(r["_retrieval_score"] for r in ev_rows) / n
        mean_icdi = sum(r["_icdi"] for r in ev_rows) / n
        # zone_type: most common among the event's rows
        zone_counts = defaultdict(int)
        for r in ev_rows:
            zone_counts[r.get("zone_type", "Unknown")] += 1
        zone_type = max(zone_counts, key=zone_counts.get) if zone_counts else "Unknown"

        results.append({
            "event_id": event_id,
            "start_time": ev_rows[0].get("start_time", ""),
            "end_time": ev_rows[0].get("end_time", ""),
            "zone_type": zone_type,
            "zone_risk": ZONE_RISK.get(zone_type, 1.0),
            "num_rows": n,
            "mean_absence_confidence": round(mean_ac, 4),
            "mean_severity": round(mean_sev, 4),
            "mean_retrieval_score": round(mean_score, 4),
            "icdi": round(mean_icdi, 4),
        })
    return results


def write_event_icdi(results, path):
    fieldnames = ["event_id", "start_time", "end_time", "zone_type", "zone_risk",
                  "num_rows", "mean_absence_confidence", "mean_severity",
                  "mean_retrieval_score", "icdi"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    return path


def main():
    parser = argparse.ArgumentParser(description="Compute ICDI from event_compliance.csv")
    parser.add_argument("--input", type=str,
                         default=os.path.join(OUTPUT_DIR, "event_compliance.csv"))
    parser.add_argument("--output", type=str,
                         default=os.path.join(OUTPUT_DIR, "event_icdi.csv"))
    args = parser.parse_args()

    rows = load_event_compliance(args.input)
    results = aggregate_by_event(rows)
    out_path = write_event_icdi(results, args.output)

    print(f"[OK] Computed ICDI for {len(results)} event(s) -> {out_path}")


if __name__ == "__main__":
    main()