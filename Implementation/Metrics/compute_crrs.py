"""
compute_crrs.py
-----------------
Reads outputs/event_icdi.csv (produced by compute_icdi.py), groups events that
are close together in time (to avoid double-counting the same physical hazard
detected across several nearby IMU events), and computes CRRS (Cumulative
Route Risk Score):

    CRRS = Sum(ICDI_i x frequency_weight_i) / number_of_events

Where:
    - Events are first sorted by start_time and clustered into groups: a new
      group starts whenever the gap between one event's end_time and the next
      event's start_time exceeds --gap seconds (default 30s).
    - frequency_weight_i = size of the group event i belongs to (i.e. how many
      times this kind of hazard recurred near that spot on the route). An
      isolated one-off event gets frequency_weight = 1; a hazard that
      re-triggered 4 times nearby gets frequency_weight = 4.
    - number_of_events = total number of events (post-grouping weighting is
      still applied per-individual-event, matching the formula literally).

This grouping choice is a modelling assumption (the spec did not pin down the
exact clustering/weighting rule) — see README.md for a note on this, and
adjust --gap / the weighting logic if your definition differs.

Output: outputs/route_risk.csv — one row per event (with its group + weight),
plus a trailing "ROUTE_TOTAL" summary row with the final CRRS.

Usage:
    python3 compute_crrs.py --gap 30 \
        --input outputs/event_icdi.csv \
        --output outputs/route_risk.csv
"""

import argparse
import csv
import os

from config import OUTPUT_DIR


def _to_float(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_event_icdi(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["_start"] = _to_float(row.get("start_time"), 0.0)
            row["_end"] = _to_float(row.get("end_time"), 0.0)
            row["_icdi"] = _to_float(row.get("icdi"), 0.0)
            rows.append(row)
    rows.sort(key=lambda r: r["_start"])
    return rows


def group_nearby_events(rows, gap_seconds):
    groups = []
    current_group = []
    prev_end = None
    for r in rows:
        if prev_end is not None and (r["_start"] - prev_end) > gap_seconds:
            groups.append(current_group)
            current_group = []
        current_group.append(r)
        prev_end = max(prev_end or r["_end"], r["_end"])
    if current_group:
        groups.append(current_group)
    return groups


def compute_crrs(rows, gap_seconds):
    groups = group_nearby_events(rows, gap_seconds)

    detail_rows = []
    weighted_sum = 0.0
    for gi, group in enumerate(groups, start=1):
        freq_weight = len(group)
        for r in group:
            weighted_icdi = r["_icdi"] * freq_weight
            weighted_sum += weighted_icdi
            detail_rows.append({
                "event_id": r["event_id"],
                "group_id": f"grp{gi:03d}",
                "start_time": r.get("start_time", ""),
                "end_time": r.get("end_time", ""),
                "icdi": round(r["_icdi"], 4),
                "frequency_weight": freq_weight,
                "weighted_icdi": round(weighted_icdi, 4),
            })

    n_events = len(rows)
    crrs = round(weighted_sum / n_events, 4) if n_events else 0.0
    return detail_rows, crrs, len(groups)


def write_route_risk(detail_rows, crrs, n_groups, n_events, path):
    fieldnames = ["event_id", "group_id", "start_time", "end_time", "icdi",
                  "frequency_weight", "weighted_icdi"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)
        writer.writerow({
            "event_id": "ROUTE_TOTAL",
            "group_id": f"{n_groups}_groups",
            "start_time": "",
            "end_time": "",
            "icdi": "",
            "frequency_weight": n_events,
            "weighted_icdi": crrs,
        })
    return path


def main():
    parser = argparse.ArgumentParser(description="Compute CRRS from event_icdi.csv")
    parser.add_argument("--gap", type=float, default=30.0,
                         help="Max time gap (sec) between events to still count as 'nearby'")
    parser.add_argument("--input", type=str,
                         default=os.path.join(OUTPUT_DIR, "event_icdi.csv"))
    parser.add_argument("--output", type=str,
                         default=os.path.join(OUTPUT_DIR, "route_risk.csv"))
    args = parser.parse_args()

    rows = load_event_icdi(args.input)
    detail_rows, crrs, n_groups = compute_crrs(rows, args.gap)
    out_path = write_route_risk(detail_rows, crrs, n_groups, len(rows), args.output)

    print(f"[OK] {len(rows)} events grouped into {n_groups} nearby-cluster(s)")
    print(f"     CRRS = {crrs}")
    print(f"     -> {out_path}")


if __name__ == "__main__":
    main()