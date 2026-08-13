"""
build_kb_and_mapping.py

Builds the two dependencies compute_rhr.py / compute_chr.py / compute_haa.py
need that were not present anywhere in the project:

  1. kb/irc35_kb.json, kb/irc67_kb.json -- clause_id -> clause_text lookup,
     built directly from irc35_index.json / irc67_index.json (already
     extracted from the real IRC PDFs earlier in this project).

  2. gold_clause_mapping.json -- {rule_name: [expected_clause_ids]}, used by
     compute_chr.py to decide whether a citation is hallucinated.

IMPORTANT, DISCLOSE THIS: there is no independently human-verified "correct
clause per rule" mapping anywhere in this project. gold_clause_mapping.json
is DERIVED, not independently validated: for each rule (road_surface ->
IRC35, infrastructure -> IRC67), it's built from the set of clause_ids that
actually appear as `cited_clause` in the labeled_dataset_local.json gold
labels (the teacher-labeled, quality-gated training data from earlier).
This means CHR as computed here measures "did the model cite something the
teacher-labeling process also cited at some point," not "did the model cite
the objectively correct IRC clause." Report it with that caveat -- it is a
real, honest proxy given what data exists, not a fabricated ground truth.

Usage:
    python build_kb_and_mapping.py
"""

import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(BASE), "Dataset")


def build_kb():
    os.makedirs(os.path.join(BASE, "kb"), exist_ok=True)
    for code in ("irc35", "irc67"):
        with open(os.path.join(DATASET_DIR, f"{code}_index_scoped.json"), encoding="utf-8") as f:
            clauses = json.load(f)
        # Normalize to underscore format ("IRC35_4.2.1") -- V1's citations
        # now come from the real Knowledge RAG (V2/RAG/indexes/chroma),
        # which uses underscore IDs, but this legacy PDF-regex index still
        # uses hyphens ("IRC35-4.2.1"). Without normalizing, CHR/RHR compare
        # the same clause under two different strings and falsely report
        # every real citation as hallucinated. See conversation log.
        kb = {c["clause_id"].replace("-", "_"): c["text"] for c in clauses}
        out_path = os.path.join(BASE, "kb", f"{code}_kb.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(kb, f, indent=2, ensure_ascii=False)
        print(f"  {code}_kb.json: {len(kb)} clauses -> {out_path}")


def build_gold_clause_mapping():
    with open(os.path.join(DATASET_DIR, "v1_dataset_with_rag_fixed.json"), encoding="utf-8") as f:
        events = json.load(f)

    mapping = {"road_surface": set(), "infrastructure": set()}
    for e in events:
        gold = e["gold_label"]
        for rule in ("road_surface", "infrastructure"):
            fold = gold.get(rule, {})
            clause = fold.get("cited_clause")
            if clause and clause not in ("null", "None", ""):
                mapping[rule].add(clause.replace("-", "_"))  # normalize, see build_kb() comment

    mapping = {k: sorted(v) for k, v in mapping.items()}
    out_path = os.path.join(BASE, "gold_clause_mapping.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"  gold_clause_mapping.json: road_surface={len(mapping['road_surface'])} "
          f"clauses, infrastructure={len(mapping['infrastructure'])} clauses -> {out_path}")
    print("  CAVEAT: derived from training citations, not independently human-verified.")


if __name__ == "__main__":
    print("Building kb/*.json ...")
    build_kb()
    print("Building gold_clause_mapping.json ...")
    build_gold_clause_mapping()
