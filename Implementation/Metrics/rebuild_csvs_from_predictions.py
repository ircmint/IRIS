"""
rebuild_csvs_from_predictions.py

run_inference.py writes verdicts.csv / gold_verdicts.csv / compliance_rows.csv /
retrieval_rows.csv as a side effect of generation. run_inference_v1_embed.py only
writes predictions_v1_embed.json (no CSVs), so compute_chr.py/compute_rhr.py/
compute_haa.py/compute_idas.py/compute_crrs.py would silently read STALE CSVs
left over from the old text-mode run if pointed at the embed-mode checkpoint's
results. This rebuilds those CSVs from predictions_v1_embed.json using the same
row-construction logic as run_inference.py, so the domain metrics are computed
against the actual embed-mode predictions.

Usage:
    python rebuild_csvs_from_predictions.py --predictions predictions_v1_embed.json
"""

import argparse
import csv
import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ABSENCE_FROM_PRESENCE = {"Yes": 0.9, "Unclear": 0.5, "No": 0.1}
SEVERITY_NUM = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}


def write_csv(rows, filename, fieldnames):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  wrote {len(rows)} rows -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="predictions_v1_embed.json")
    args = parser.parse_args()

    with open(os.path.join(BASE, args.predictions), encoding="utf-8") as f:
        predictions = json.load(f)

    verdict_rows, gold_rows, compliance_rows, retrieval_rows = [], [], [], []

    for p in predictions:
        if p.get("fail_reason") or not p.get("parsed"):
            continue
        parsed = p["parsed"]
        gold = p["gold"]
        event_id = p["event_id"]

        for rule in ("road_surface", "infrastructure"):
            if rule not in parsed or rule not in gold:
                continue
            pred_fold = parsed[rule]
            gold_fold = gold[rule]
            pred_verdict = "NON_COMPLIANT" if pred_fold.get("presence") == "Yes" else "COMPLIANT"
            gold_verdict = "NON_COMPLIANT" if gold_fold.get("presence") == "Yes" else "COMPLIANT"

            verdict_rows.append({
                "event_id": event_id, "frame_id": "0", "rule": rule,
                "verdict": pred_verdict, "cited_clause": pred_fold.get("cited_clause") or "",
                "reasoning": pred_fold.get("reasoning", ""),
            })
            gold_rows.append({"event_id": event_id, "frame_id": "0", "gold_verdict": gold_verdict})

            retrieved_key = "retrieved_irc35" if rule == "road_surface" else "retrieved_irc67"
            retrieved = gold.get(retrieved_key, [])
            retrieval_score = 1.0 if pred_fold.get("cited_clause") in retrieved else 0.5

            compliance_rows.append({
                "event_id": event_id, "start_time": "", "end_time": "",
                "zone_type": "Unknown",
                "absence_confidence": ABSENCE_FROM_PRESENCE.get(pred_fold.get("presence"), 0.5),
                "severity": SEVERITY_NUM.get(pred_fold.get("severity", "LOW"), 1),
                "retrieval_score": retrieval_score, "rule": rule,
            })

            for rank, clause_id in enumerate(retrieved, start=1):
                retrieval_rows.append({
                    "event_id": f"{event_id}_{rule}", "retrieval_rank": rank,
                    "clause_id": clause_id, "human_relevant": "",
                })

    write_csv(verdict_rows, "verdicts.csv", ["event_id", "frame_id", "rule", "verdict", "cited_clause", "reasoning"])
    write_csv(gold_rows, "gold_verdicts.csv", ["event_id", "frame_id", "gold_verdict"])
    write_csv(compliance_rows, "event_compliance.csv",
              ["event_id", "start_time", "end_time", "zone_type", "absence_confidence", "severity", "retrieval_score", "rule"])
    write_csv(retrieval_rows, "retrieval_results.csv", ["event_id", "retrieval_rank", "clause_id", "human_relevant"])


if __name__ == "__main__":
    main()
