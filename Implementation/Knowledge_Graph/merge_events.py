"""
merge_events.py
----------------
Merges the IMU-pipeline's events.csv with this pipeline's frame_reports.csv
on timestamp windows, producing outputs/event_compliance.csv.

events.csv (produced elsewhere, IMU pipeline) is expected to have columns:
    clip_id, start_time, end_time, decision, event_type, peak_confidence

events.csv has no event_id column, so one is generated deterministically as:
    "<clip_id>_evt<row_index>"

frame_reports.csv (produced by main_pipeline.py, PART 1 extended version) is
expected to have columns:
    frame_id, timestamp_sec, is_compliant, rule, irc_code, clause_id,
    clause_heading, description, retrieval_score, retrieval_rank, query_text,
    absence_confidence, zone_type, infrastructure_element, severity

For every event, every frame_reports.csv row whose timestamp_sec falls in
    start_time <= timestamp_sec <= end_time
is attached to that event. A single event may therefore produce multiple
output rows (one per frame x per violation-row in that frame).

Output: outputs/event_compliance.csv with columns:
    event_id, start_time, end_time, frame_id, timestamp, rule, clause_id,
    retrieval_score, absence_confidence, severity, zone_type

Usage:
    python3 merge_events.py --events events.csv \
        --frame-reports outputs/frame_reports.csv \
        --output outputs/event_compliance.csv
"""

import argparse
import csv
import os

from config import OUTPUT_DIR


def _to_float(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_events(path):
    events = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            events.append({
                "event_id": f"{row.get('clip_id', 'clip')}_evt{i:04d}",
                "clip_id": row.get("clip_id", ""),
                "start_time": _to_float(row.get("start_time"), 0.0),
                "end_time": _to_float(row.get("end_time"), 0.0),
                "decision": row.get("decision", ""),
                "event_type": row.get("event_type", ""),
                "peak_confidence": row.get("peak_confidence", ""),
            })
    return events


def load_frame_reports(path):
    frames = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["_timestamp"] = _to_float(row.get("timestamp_sec"), None)
            frames.append(row)
    return frames


def merge_events(events, frame_rows):
    out_rows = []
    for ev in events:
        for fr in frame_rows:
            ts = fr.get("_timestamp")
            if ts is None:
                continue
            if ev["start_time"] <= ts <= ev["end_time"]:
                out_rows.append({
                    "event_id": ev["event_id"],
                    "start_time": ev["start_time"],
                    "end_time": ev["end_time"],
                    "frame_id": fr.get("frame_id", ""),
                    "timestamp": ts,
                    "rule": fr.get("rule", ""),
                    "clause_id": fr.get("clause_id", ""),
                    "retrieval_score": fr.get("retrieval_score", ""),
                    "absence_confidence": fr.get("absence_confidence", ""),
                    "severity": fr.get("severity", ""),
                    "zone_type": fr.get("zone_type", ""),
                })
    return out_rows


def write_event_compliance(rows, path):
    fieldnames = ["event_id", "start_time", "end_time", "frame_id", "timestamp",
                  "rule", "clause_id", "retrieval_score", "absence_confidence",
                  "severity", "zone_type"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    parser = argparse.ArgumentParser(description="Merge IMU events.csv with frame_reports.csv")
    parser.add_argument("--events", type=str, required=True, help="Path to events.csv (IMU pipeline)")
    parser.add_argument("--frame-reports", type=str,
                         default=os.path.join(OUTPUT_DIR, "frame_reports.csv"),
                         help="Path to frame_reports.csv")
    parser.add_argument("--output", type=str,
                         default=os.path.join(OUTPUT_DIR, "event_compliance.csv"),
                         help="Path to write event_compliance.csv")
    args = parser.parse_args()

    events = load_events(args.events)
    frame_rows = load_frame_reports(args.frame_reports)
    merged = merge_events(events, frame_rows)
    out_path = write_event_compliance(merged, args.output)

    print(f"[OK] {len(events)} events, {len(frame_rows)} frame_report rows "
          f"-> {len(merged)} merged rows -> {out_path}")


if __name__ == "__main__":
    main()