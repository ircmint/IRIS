"""
compute_hcas.py
------------------
Hierarchical Clause Alignment Score (HCAS), and its three components:

    ILA  (IRC-Level Accuracy)      - did the model pick the right IRC document?
    SLA  (Section-Level Accuracy)  - given the right document, right section?
    CLA  (Clause-Level Accuracy)   - given the right document + section, exact clause?
    HCAS = (ILA + SLA + CLA) / 3   - averaged per citation, then across all citations

Unlike CHR (which treats every non-gold citation as equally wrong) or CCS
(which uses raw string-segment proximity), HCAS is EXPLICITLY hierarchical
and gated: you cannot get SLA credit unless ILA is already correct, and you
cannot get CLA credit unless both ILA and SLA are correct. This mirrors how
a human auditor actually checks a citation: book -> chapter -> paragraph.

--------------------------------------------------------------------------
REQUIRED INPUT SCHEMA (adjust to match your actual pipeline output)
--------------------------------------------------------------------------
1) gold_hierarchical_mapping.json
   Maps each rule to a list of acceptable gold citations, each with three
   explicit fields:

       {
         "no_footpath": [
           {"irc": "35", "section": "6", "clause": "6.2.3"},
           {"irc": "35", "section": "6", "clause": "6.2.4"}
         ],
         ...
       }

   (A rule can have more than one acceptable gold citation, same as
   gold_clause_mapping.json today.)

2) verdicts.csv must provide the predicted citation split into three
   columns: `cited_irc`, `cited_section`, `cited_clause`.

   If your pipeline instead emits a single combined string (e.g.
   "IRC35/6.2.3" or "35/6.2.3"), use --combined-col to point at that
   column instead and it will be parsed automatically -- see
   parse_combined_clause() below.

Outputs:
    outputs/hcas_report.csv

Usage:
    python compute_hcas.py
    python compute_hcas.py --combined-col cited_clause_full
"""

import argparse
import csv
import json
import os
import re

from config import OUTPUT_DIR


GOLD_MAPPING_PATH = os.path.join(
    os.path.dirname(__file__),
    "gold_hierarchical_mapping.json"
)

# Matches things like "IRC35/6.2.3", "IRC 35 / 6.2.3", "35/6.2.3"
COMBINED_PATTERN = re.compile(
    r"(?:IRC\s*)?(?P<irc>[\w-]+)\s*[/\-:]\s*(?P<clause>[\d.]+)",
    re.IGNORECASE
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


def parse_combined_clause(value):
    """
    Parses a combined citation string like 'IRC35/6.2.3' into
    (irc='35', section='6', clause='6.2.3'). The section is taken as the
    first dotted segment of the clause path, per the convention in the
    HCAS spec (clause '6.2.3' implies section '6').

    Returns (irc, section, clause) or (None, None, None) if unparseable.
    """
    if value is None:
        return None, None, None

    m = COMBINED_PATTERN.search(str(value).strip())
    if not m:
        return None, None, None

    irc = m.group("irc")
    clause = m.group("clause")
    section = clause.split(".")[0] if clause else None
    return irc, section, clause


def get_predicted_citation(row, combined_col):
    if combined_col:
        return parse_combined_clause(row.get(combined_col))

    irc = str(row.get("cited_irc", "")).strip() or None
    section = str(row.get("cited_section", "")).strip() or None
    clause = str(row.get("cited_clause", "")).strip() or None
    return irc, section, clause


def score_row(pred_irc, pred_section, pred_clause, gold_entries):
    """
    Computes ILA, SLA, CLA for a single prediction against a list of
    acceptable gold entries (each a dict with irc/section/clause).
    Credit is gated: SLA requires ILA, CLA requires SLA.
    """
    ila = any(g.get("irc") == pred_irc for g in gold_entries)

    sla = ila and any(
        g.get("irc") == pred_irc and g.get("section") == pred_section
        for g in gold_entries
    )

    cla = sla and any(
        g.get("irc") == pred_irc
        and g.get("section") == pred_section
        and g.get("clause") == pred_clause
        for g in gold_entries
    )

    ila_v, sla_v, cla_v = int(ila), int(sla), int(cla)
    hcas_v = round((ila_v + sla_v + cla_v) / 3, 4)
    return ila_v, sla_v, cla_v, hcas_v


def compute_hcas(verdict_rows, gold_mapping, combined_col):
    results = []
    total = 0
    ila_sum = sla_sum = cla_sum = hcas_sum = 0

    for row in verdict_rows:
        if row["verdict"] != "NON_COMPLIANT":
            continue

        pred_irc, pred_section, pred_clause = get_predicted_citation(row, combined_col)

        if pred_irc is None and pred_section is None and pred_clause is None:
            continue

        rule = row.get("rule", "").strip()
        gold_entries = gold_mapping.get(rule, [])

        ila_v, sla_v, cla_v, hcas_v = score_row(
            pred_irc, pred_section, pred_clause, gold_entries
        )

        total += 1
        ila_sum += ila_v
        sla_sum += sla_v
        cla_sum += cla_v
        hcas_sum += hcas_v

        results.append({
            "event_id": row.get("event_id", ""),
            "frame_id": row.get("frame_id", ""),
            "rule": rule,
            "predicted_irc": pred_irc,
            "predicted_section": pred_section,
            "predicted_clause": pred_clause,
            "gold_entries": json.dumps(gold_entries),
            "ILA": ila_v,
            "SLA": sla_v,
            "CLA": cla_v,
            "HCAS": hcas_v,
        })

    summary = {
        "ILA": round(ila_sum / total, 4) if total else 0.0,
        "SLA": round(sla_sum / total, 4) if total else 0.0,
        "CLA": round(cla_sum / total, 4) if total else 0.0,
        "HCAS": round(hcas_sum / total, 4) if total else 0.0,
    }

    return results, summary, total


def write_report(results, summary, total, output_path):
    fieldnames = [
        "event_id", "frame_id", "rule",
        "predicted_irc", "predicted_section", "predicted_clause",
        "gold_entries", "ILA", "SLA", "CLA", "HCAS",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        writer.writerow({
            "event_id": "HCAS_TOTAL",
            "frame_id": "",
            "rule": "",
            "predicted_irc": "",
            "predicted_section": "",
            "predicted_clause": "",
            "gold_entries": f"n={total}",
            "ILA": summary["ILA"],
            "SLA": summary["SLA"],
            "CLA": summary["CLA"],
            "HCAS": summary["HCAS"],
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verdicts",
        default=os.path.join(OUTPUT_DIR, "verdicts.csv")
    )
    parser.add_argument(
        "--output",
        default=os.path.join(OUTPUT_DIR, "hcas_report.csv")
    )
    parser.add_argument(
        "--combined-col",
        default=None,
        help=(
            "Name of a single column holding a combined citation string "
            "(e.g. 'IRC35/6.2.3') to parse instead of separate "
            "cited_irc/cited_section/cited_clause columns."
        )
    )
    args = parser.parse_args()

    gold_mapping = load_gold_mapping()
    verdict_rows = load_verdicts(args.verdicts)

    results, summary, total = compute_hcas(
        verdict_rows, gold_mapping, args.combined_col
    )

    write_report(results, summary, total, args.output)

    print(f"Total citations evaluated : {total}")
    print(f"ILA  (document accuracy)  : {summary['ILA']:.4f}")
    print(f"SLA  (+ section accuracy) : {summary['SLA']:.4f}")
    print(f"CLA  (+ clause accuracy)  : {summary['CLA']:.4f}")
    print(f"HCAS (overall)            : {summary['HCAS']:.4f}")
    print(f"Report written to         : {args.output}")


if __name__ == "__main__":
    main()