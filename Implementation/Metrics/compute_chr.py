"""
compute_chr.py
----------------
Citation Hallucination Rate (CHR)

CHR = hallucinated_citations / total_citations

A citation is considered hallucinated if the cited clause is NOT one of the
gold clauses defined for that violation rule in gold_clause_mapping.json.

Outputs:
    outputs/chr_report.csv

Usage:
    python compute_chr.py
"""

import argparse
import csv
import json
import os

from config import OUTPUT_DIR


GOLD_MAPPING_PATH = os.path.join(
    os.path.dirname(__file__),
    "gold_clause_mapping.json"
)


def load_gold_mapping():
    with open(GOLD_MAPPING_PATH, "r") as f:
        return json.load(f)


def load_verdicts(path):
    rows = []

    try:
        f = open(path, newline="", encoding="utf-8")
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
        f.close()

    except UnicodeDecodeError:
        f = open(path, newline="", encoding="cp1252")
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
        f.close()

    return rows


def compute_chr(verdict_rows, gold_mapping):

    results = []

    hallucinated = 0
    total = 0

    for row in verdict_rows:

        if row["verdict"] != "NON_COMPLIANT":
            continue

        cited = str(row["cited_clause"]).strip()

        if cited == "":
            continue

        total += 1

        rule = row.get("rule", "").strip()

        expected = gold_mapping.get(rule, [])

        if cited in expected:
            is_hallucinated = False
            reason = "supported by gold mapping"
        else:
            is_hallucinated = True
            hallucinated += 1
            reason = (
                f"expected one of {expected}, "
                f"predicted {cited}"
            )

        results.append({
            "event_id": row["event_id"],
            "frame_id": row["frame_id"],
            "rule": rule,
            "cited_clause": cited,
            "gold_clause": ", ".join(expected),
            "hallucinated": is_hallucinated,
            "reason": reason
        })

    chr_value = hallucinated / total if total else 0.0

    return results, chr_value, hallucinated, total


def write_report(results,
                 chr_value,
                 hallucinated,
                 total,
                 output_path):

    fieldnames = [
        "event_id",
        "frame_id",
        "rule",
        "cited_clause",
        "gold_clause",
        "hallucinated",
        "reason"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        writer.writerows(results)

        writer.writerow({
            "event_id": "CHR_TOTAL",
            "frame_id": "",
            "rule": "",
            "cited_clause": f"{hallucinated}/{total}",
            "gold_clause": "",
            "hallucinated": "",
            "reason": f"CHR={chr_value:.4f}"
        })


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--verdicts",
        default=os.path.join(OUTPUT_DIR, "verdicts.csv")
    )

    parser.add_argument(
        "--output",
        default=os.path.join(OUTPUT_DIR, "chr_report.csv")
    )

    args = parser.parse_args()

    gold_mapping = load_gold_mapping()

    verdict_rows = load_verdicts(args.verdicts)

    results, chr_value, hallucinated, total = compute_chr(
        verdict_rows,
        gold_mapping
    )

    write_report(
        results,
        chr_value,
        hallucinated,
        total,
        args.output
    )

    print(f"Total citations      : {total}")
    print(f"Hallucinated         : {hallucinated}")
    print(f"CHR                  : {chr_value:.4f}")
    print(f"Report written to    : {args.output}")


if __name__ == "__main__":
    main()