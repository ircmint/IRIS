"""
compute_rhr.py
----------------
Response Hallucination Rate (RHR)

For every NON_COMPLIANT verdict, we treat:
    premise    = the actual IRC clause text (looked up from kb/*_kb.json
                 using the row's `cited_clause` id)
    hypothesis = the model-generated reasoning/justification text for
                 that verdict (see HYPOTHESIS_FIELDS below)

We run cross-encoder/nli-deberta-v3-large (an MNLI-finetuned cross-encoder
built on microsoft/deberta-v3-large -- Microsoft itself never published an
MNLI checkpoint under the "-v3-" name) to classify the relationship as
Entailment / Neutral / Contradiction.

    RHR = (Neutral + Contradiction) / Total

A high RHR means the model is frequently generating reasoning that is
NOT actually supported by the clause it cited -- i.e. it is
"hallucinating" a justification.

Outputs:
    outputs/rhr_report.csv

Usage:
    python compute_rhr.py
    python compute_rhr.py --hypothesis-field justification
    python compute_rhr.py --device cuda
"""

import argparse
import csv
import glob
import json
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import OUTPUT_DIR


KB_DIR = os.path.join(os.path.dirname(__file__), "kb")

MODEL_NAME = "cross-encoder/nli-deberta-v3-large"

# The verdicts.csv column that holds the model's generated reasoning text
# for a given verdict row. We try these in order and use the first one
# that exists and is non-empty for a given row. Edit this list if your
# verdicts.csv uses a different column name.
HYPOTHESIS_FIELDS = [
    "reasoning",
    "justification",
    "explanation",
    "verdict_reason",
    "response",
    "output_text",
]

# Delimiters used to split a cited_clause cell that references more than
# one clause id (e.g. "IRC-3.2, IRC-4.1").
CLAUSE_DELIMITERS = [",", ";", "|", "/"]


def load_kb(kb_dir):
    """
    Loads all kb/*_kb.json files into a single {clause_id: clause_text}
    dict. Supports two common schemas:

        1) {"IRC-3.2": "Every building shall ...", ...}
        2) [{"clause_id": "IRC-3.2", "text": "Every building shall ..."}, ...]
           (also accepts "id" and "clause_text" as key aliases)
    """

    kb = {}

    for path in sorted(glob.glob(os.path.join(kb_dir, "*_kb.json"))):

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            for clause_id, value in data.items():
                if isinstance(value, str):
                    kb[str(clause_id).strip()] = value
                elif isinstance(value, dict):
                    text = value.get("text") or value.get("clause_text") or ""
                    kb[str(clause_id).strip()] = text

        elif isinstance(data, list):
            for entry in data:
                clause_id = (
                    entry.get("clause_id")
                    or entry.get("id")
                    or entry.get("clause")
                )
                text = entry.get("text") or entry.get("clause_text") or ""
                if clause_id:
                    kb[str(clause_id).strip()] = text

    return kb


def split_clause_ids(cited):
    ids = [cited]

    for delim in CLAUSE_DELIMITERS:
        if delim in cited:
            ids = [c.strip() for c in cited.split(delim)]
            break

    return [c for c in ids if c]


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


def get_hypothesis_text(row, hypothesis_field):
    fields = [hypothesis_field] + [
        f for f in HYPOTHESIS_FIELDS if f != hypothesis_field
    ]

    for field in fields:
        value = row.get(field, "")
        if value and str(value).strip():
            return str(value).strip()

    return ""


def load_nli_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model


def classify_pair(premise, hypothesis, tokenizer, model, device):
    """
    Returns (label, probs_dict) where label is one of
    "ENTAILMENT" / "NEUTRAL" / "CONTRADICTION".
    """

    inputs = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits[0]

    probs = torch.softmax(logits, dim=-1)

    id2label = model.config.id2label

    probs_dict = {
        id2label[i].upper(): float(probs[i])
        for i in range(len(probs))
    }

    top_idx = int(torch.argmax(probs))
    label = id2label[top_idx].upper()

    return label, probs_dict


LABEL_PRIORITY = {"ENTAILMENT": 2, "NEUTRAL": 1, "CONTRADICTION": 0}


def classify_against_clauses(cited, hypothesis, kb, tokenizer, model, device):
    """
    A cited_clause cell may reference multiple clause ids. We classify
    the hypothesis against each resolvable clause and keep the
    "best" result for the model (ENTAILMENT beats NEUTRAL beats
    CONTRADICTION), mirroring a charitable reading of a multi-citation
    answer. Also returns the clause text(s) actually used and any ids
    that could not be found in the KB.
    """

    clause_ids = split_clause_ids(cited)

    best_label = None
    best_probs = None
    best_clause_id = None
    used_texts = []
    missing_ids = []

    for clause_id in clause_ids:
        clause_text = kb.get(clause_id)

        if not clause_text:
            missing_ids.append(clause_id)
            continue

        used_texts.append(clause_text)

        label, probs = classify_pair(
            clause_text, hypothesis, tokenizer, model, device
        )

        if best_label is None or LABEL_PRIORITY[label] > LABEL_PRIORITY[best_label]:
            best_label = label
            best_probs = probs
            best_clause_id = clause_id

    return best_label, best_probs, best_clause_id, used_texts, missing_ids


def compute_rhr(verdict_rows, kb, tokenizer, model, device, hypothesis_field):

    results = []

    entail_count = 0
    neutral_count = 0
    contradiction_count = 0
    skipped = 0
    total = 0

    for row in verdict_rows:

        if row.get("verdict", "").strip() != "NON_COMPLIANT":
            continue

        cited = str(row.get("cited_clause", "")).strip()

        if cited == "":
            continue

        hypothesis = get_hypothesis_text(row, hypothesis_field)

        if hypothesis == "":
            skipped += 1
            continue

        (
            label,
            probs,
            used_clause_id,
            used_texts,
            missing_ids
        ) = classify_against_clauses(
            cited, hypothesis, kb, tokenizer, model, device
        )

        if label is None:
            # None of the cited clause ids were found in the KB at all
            skipped += 1
            continue

        total += 1

        if label == "ENTAILMENT":
            entail_count += 1
            reason = "reasoning is supported by cited clause text"
        elif label == "NEUTRAL":
            neutral_count += 1
            reason = "reasoning is not clearly supported or refuted by the clause"
        else:
            contradiction_count += 1
            reason = "reasoning contradicts the cited clause text"

        if missing_ids:
            reason += f" (unresolved clause ids: {', '.join(missing_ids)})"

        results.append({
            "event_id": row.get("event_id", ""),
            "frame_id": row.get("frame_id", ""),
            "rule": row.get("rule", "").strip(),
            "cited_clause": cited,
            "matched_clause_id": used_clause_id or "",
            "hypothesis_text": hypothesis,
            "predicted_label": label,
            "entailment_prob": f"{probs.get('ENTAILMENT', 0.0):.4f}",
            "neutral_prob": f"{probs.get('NEUTRAL', 0.0):.4f}",
            "contradiction_prob": f"{probs.get('CONTRADICTION', 0.0):.4f}",
            "reason": reason
        })

    rhr_value = (
        (neutral_count + contradiction_count) / total if total else 0.0
    )

    return (
        results,
        rhr_value,
        entail_count,
        neutral_count,
        contradiction_count,
        skipped,
        total
    )


def write_report(
    results,
    rhr_value,
    entail_count,
    neutral_count,
    contradiction_count,
    skipped,
    total,
    output_path
):

    fieldnames = [
        "event_id",
        "frame_id",
        "rule",
        "cited_clause",
        "matched_clause_id",
        "hypothesis_text",
        "predicted_label",
        "entailment_prob",
        "neutral_prob",
        "contradiction_prob",
        "reason"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        writer.writerows(results)

        writer.writerow({
            "event_id": "RHR_TOTAL",
            "frame_id": "",
            "rule": "",
            "cited_clause": "",
            "matched_clause_id": "",
            "hypothesis_text": "",
            "predicted_label": (
                f"E={entail_count} N={neutral_count} "
                f"C={contradiction_count} skipped={skipped}"
            ),
            "entailment_prob": "",
            "neutral_prob": "",
            "contradiction_prob": "",
            "reason": f"RHR={rhr_value:.4f} over total={total}"
        })


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--verdicts",
        default=os.path.join(OUTPUT_DIR, "verdicts.csv")
    )

    parser.add_argument(
        "--kb-dir",
        default=KB_DIR
    )

    parser.add_argument(
        "--output",
        default=os.path.join(OUTPUT_DIR, "rhr_report.csv")
    )

    parser.add_argument(
        "--model",
        default=MODEL_NAME
    )

    parser.add_argument(
        "--hypothesis-field",
        default="reasoning",
        help=(
            "verdicts.csv column containing the model's generated "
            "reasoning text to check against the cited clause. "
            f"Falls back through {HYPOTHESIS_FIELDS} if not found."
        )
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    kb = load_kb(args.kb_dir)

    verdict_rows = load_verdicts(args.verdicts)

    tokenizer, model = load_nli_model(args.model, args.device)

    (
        results,
        rhr_value,
        entail_count,
        neutral_count,
        contradiction_count,
        skipped,
        total
    ) = compute_rhr(
        verdict_rows,
        kb,
        tokenizer,
        model,
        args.device,
        args.hypothesis_field
    )

    write_report(
        results,
        rhr_value,
        entail_count,
        neutral_count,
        contradiction_count,
        skipped,
        total,
        args.output
    )

    print(f"Total evaluated       : {total}")
    print(f"Entailment            : {entail_count}")
    print(f"Neutral               : {neutral_count}")
    print(f"Contradiction         : {contradiction_count}")
    print(f"Skipped (no data)     : {skipped}")
    print(f"RHR                   : {rhr_value:.4f}")
    print(f"Report written to     : {args.output}")


if __name__ == "__main__":
    main()