"""
compute_standard_metrics.py

Reads predictions.json (from run_inference.py) and computes the standard
ML/NLG metrics requested alongside the domain-specific ones:

  - Accuracy, Macro-F1: per-fold (road_surface, infrastructure), 3-class
    (Yes/No/Unclear) presence classification against gold.
  - BLEU, ROUGE-L, CIDEr: comparing the model's generated `reasoning` text
    against the gold `reasoning` text, per fold, corpus-averaged.

Events that failed schema validation (see run_inference.py's fail_reason)
are EXCLUDED from these metrics, not scored as wrong-but-included or
silently imputed -- the exclusion count is reported so it's visible.

Usage:
    python compute_standard_metrics.py
"""

import json
import os

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score

from cider_scorer import compute_cider

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    with open(os.path.join(BASE, "predictions.json"), encoding="utf-8") as f:
        predictions = json.load(f)

    valid = [p for p in predictions if p["fail_reason"] is None]
    failed = len(predictions) - len(valid)
    print(f"Total val events: {len(predictions)} | Valid: {len(valid)} | "
          f"Excluded (schema/parse failure): {failed}\n")

    smoothing = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    report = {}
    for rule in ("road_surface", "infrastructure"):
        pred_presence = [p["parsed"][rule]["presence"] for p in valid]
        gold_presence = [p["gold"][rule]["presence"] for p in valid]

        acc = accuracy_score(gold_presence, pred_presence)
        macro_f1 = f1_score(gold_presence, pred_presence, average="macro",
                             labels=["Yes", "No", "Unclear"], zero_division=0)

        pred_reasoning = [p["parsed"][rule].get("reasoning", "") for p in valid]
        gold_reasoning = [p["gold"][rule].get("reasoning", "") for p in valid]

        bleu_scores = [
            sentence_bleu([g.lower().split()], c.lower().split(), smoothing_function=smoothing)
            for c, g in zip(pred_reasoning, gold_reasoning) if g.strip() and c.strip()
        ]
        bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

        rouge_l_scores = [
            rouge.score(g, c)["rougeL"].fmeasure
            for c, g in zip(pred_reasoning, gold_reasoning) if g.strip() and c.strip()
        ]
        rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0

        pairs = [(c, [g]) for c, g in zip(pred_reasoning, gold_reasoning) if g.strip() and c.strip()]
        cider = compute_cider([c for c, _ in pairs], [g for _, g in pairs]) if pairs else 0.0

        report[rule] = {
            "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
            "bleu": round(bleu, 4), "rouge_l": round(rouge_l, 4), "cider": round(cider, 4),
            "n_scored": len(valid),
        }

        print(f"[{rule}]")
        print(f"  Accuracy   : {acc:.4f}")
        print(f"  Macro-F1   : {macro_f1:.4f}")
        print(f"  BLEU       : {bleu:.4f}")
        print(f"  ROUGE-L    : {rouge_l:.4f}")
        print(f"  CIDEr      : {cider:.4f}")
        print()

    out_path = os.path.join(BASE, "outputs", "standard_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"n_total": len(predictions), "n_valid": len(valid),
                    "n_failed": failed, "per_fold": report}, f, indent=2)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
