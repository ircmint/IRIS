"""
compute_haa.py
----------------
Hallucination-Adjusted Accuracy (HAA)

    Accuracy = correct_verdicts / total_verdicts
               (verdict == gold_verdict, standard label-match accuracy
               against a gold-labeled file)

    HAA = Accuracy * (1 - RHR)

RHR is read from the RHR_TOTAL summary row already produced by
compute_rhr.py (outputs/rhr_report.csv), rather than recomputed here --
this keeps compute_haa.py cheap (no NLI model load) and assumes you run
compute_rhr.py first. If you'd rather recompute RHR live inside this
script, see the --recompute-rhr flag.

Gold labels are expected in a CSV with at least the columns:
    event_id, frame_id, gold_verdict
(frame_id is optional -- if your gold file only has event_id, pass
--gold-key event_id)

Outputs:
    outputs/haa_report.csv

Usage:
    python compute_haa.py
    python compute_haa.py --gold outputs/gold_verdicts.csv
    python compute_haa.py --gold-key event_id
"""

import argparse
import csv
import os
import re

from config import OUTPUT_DIR


RHR_TOTAL_PATTERN = re.compile(r"RHR=([0-9]*\.?[0-9]+)")


def load_csv_rows(path):
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


def build_gold_map(gold_rows, gold_key_fields):
    gold_map = {}

    for row in gold_rows:
        key = tuple(str(row.get(field, "")).strip() for field in gold_key_fields)
        gold_map[key] = str(row.get("gold_verdict", "")).strip()

    return gold_map


def compute_accuracy(verdict_rows, gold_map, gold_key_fields):

    results = []

    correct = 0
    total = 0
    unmatched = 0

    for row in verdict_rows:

        key = tuple(str(row.get(field, "")).strip() for field in gold_key_fields)

        if key not in gold_map:
            unmatched += 1
            continue

        predicted = str(row.get("verdict", "")).strip()
        gold = gold_map[key]

        total += 1

        is_correct = predicted == gold

        if is_correct:
            correct += 1

        results.append({
            "event_id": row.get("event_id", ""),
            "frame_id": row.get("frame_id", ""),
            "rule": row.get("rule", "").strip(),
            "predicted_verdict": predicted,
            "gold_verdict": gold,
            "correct": is_correct
        })

    accuracy = correct / total if total else 0.0

    return results, accuracy, correct, total, unmatched


def load_rhr_value(rhr_report_path):
    rows = load_csv_rows(rhr_report_path)

    for row in rows:
        if row.get("event_id", "").strip() == "RHR_TOTAL":
            reason = row.get("reason", "")
            match = RHR_TOTAL_PATTERN.search(reason)
            if match:
                return float(match.group(1))

    raise ValueError(
        f"Could not find an RHR_TOTAL row with an RHR=<value> reason "
        f"in {rhr_report_path}. Run compute_rhr.py first, or pass "
        f"--recompute-rhr."
    )


def recompute_rhr_value(verdicts_path, kb_dir, model_name, device, hypothesis_field):
    # Only imported when needed, since this pulls in torch/transformers.
    import compute_rhr

    kb = compute_rhr.load_kb(kb_dir)
    verdict_rows = compute_rhr.load_verdicts(verdicts_path)
    tokenizer, model = compute_rhr.load_nli_model(model_name, device)

    (
        _results,
        rhr_value,
        _entail_count,
        _neutral_count,
        _contradiction_count,
        _skipped,
        _total
    ) = compute_rhr.compute_rhr(
        verdict_rows, kb, tokenizer, model, device, hypothesis_field
    )

    return rhr_value


def write_report(results, accuracy, correct, total, unmatched, rhr_value, haa_value, output_path):

    fieldnames = [
        "event_id",
        "frame_id",
        "rule",
        "predicted_verdict",
        "gold_verdict",
        "correct"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        writer.writerows(results)

        writer.writerow({
            "event_id": "HAA_TOTAL",
            "frame_id": "",
            "rule": "",
            "predicted_verdict": f"{correct}/{total}",
            "gold_verdict": f"unmatched={unmatched}",
            "correct": (
                f"Accuracy={accuracy:.4f} RHR={rhr_value:.4f} "
                f"HAA={haa_value:.4f}"
            )
        })


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--verdicts",
        default=os.path.join(OUTPUT_DIR, "verdicts.csv")
    )

    parser.add_argument(
        "--gold",
        default=os.path.join(OUTPUT_DIR, "gold_verdicts.csv")
    )

    parser.add_argument(
        "--gold-key",
        default="event_id,frame_id",
        help=(
            "Comma-separated column name(s) used to join verdicts to "
            "gold labels. Default: event_id,frame_id"
        )
    )

    parser.add_argument(
        "--rhr-report",
        default=os.path.join(OUTPUT_DIR, "rhr_report.csv")
    )

    parser.add_argument(
        "--recompute-rhr",
        action="store_true",
        help="Recompute RHR live instead of reading rhr_report.csv"
    )

    parser.add_argument(
        "--kb-dir",
        default=os.path.join(os.path.dirname(__file__), "kb")
    )

    parser.add_argument(
        "--model",
        default="cross-encoder/nli-deberta-v3-large"
    )

    parser.add_argument(
        "--hypothesis-field",
        default="reasoning"
    )

    parser.add_argument(
        "--device",
        default="cpu"
    )

    parser.add_argument(
        "--output",
        default=os.path.join(OUTPUT_DIR, "haa_report.csv")
    )

    args = parser.parse_args()

    gold_key_fields = [f.strip() for f in args.gold_key.split(",") if f.strip()]

    verdict_rows = load_csv_rows(args.verdicts)
    gold_rows = load_csv_rows(args.gold)

    gold_map = build_gold_map(gold_rows, gold_key_fields)

    results, accuracy, correct, total, unmatched = compute_accuracy(
        verdict_rows, gold_map, gold_key_fields
    )

    if args.recompute_rhr:
        rhr_value = recompute_rhr_value(
            args.verdicts,
            args.kb_dir,
            args.model,
            args.device,
            args.hypothesis_field
        )
    else:
        rhr_value = load_rhr_value(args.rhr_report)

    haa_value = accuracy * (1 - rhr_value)

    write_report(
        results, accuracy, correct, total, unmatched, rhr_value, haa_value, args.output
    )

    print(f"Total matched to gold : {total}")
    print(f"Unmatched (no gold)   : {unmatched}")
    print(f"Correct               : {correct}")
    print(f"Accuracy              : {accuracy:.4f}")
    print(f"RHR                   : {rhr_value:.4f}")
    print(f"HAA                   : {haa_value:.4f}")
    print(f"Report written to     : {args.output}")


if __name__ == "__main__":
    main()