"""
build_v2_dataset.py

Builds the V2 (CGPA) training dataset -- pure JSON manipulation, no torch,
no live Chroma query needed, because the source file already carries real
Knowledge-RAG retrieval results.

SOURCE FIX (the actual bug found in the old V2 attempt): the old
build_cgpa_dataset.py read `Dataset/labeled_dataset_local_scoped.json`,
which has presence="Yes" for ALL 626 events in BOTH folds -- that is the
raw/unremapped teacher label, not what V1 was actually trained and
evaluated on. The correct source is `v1_dataset_with_rag_fixed.json`
(this project's actual training/eval gold set), which has the real,
severity-remapped presence distribution AND already-cached real
Knowledge RAG retrieval (`retrieved_irc35`/`retrieved_irc67`, top-5 each)
plus MM RAG hints. Verified: all 19 positive gold citations (11
road_surface + 8 infrastructure) fall inside their own event's top-5
retrieval -- zero retriever misses, no force-appending needed.

For each event and fold:
    gold_pointer_index = index of cited_clause within that event's own
        retrieved-5 list, if presence == "Yes"
                        = 5 (the "no clause applies" slot CGPA expects)
        otherwise -- this is the overwhelming majority case (only ~11/626
        and ~8/626 events are "Yes"), so this dataset is EXTREMELY
        imbalanced toward "no clause applies". See class-balance report
        printed at the end -- do not train without addressing this.

Output: v2_dataset.json -- 626 records with clip_path, telemetry_filtered,
gps_context, environment (unchanged), and per-fold {retrieved, gold_pointer_index,
cited_clause, presence, severity, reasoning}, plus mm_rag hint (auxiliary only).

Usage:
    python build_v2_dataset.py
"""

import json
import os

BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(BASE, "..", "OLD_V2_CGPA", "V2_root", "v1_dataset_with_rag_fixed.json")
OUTPUT = os.path.join(BASE, "v2_dataset.json")


def normalize(clause_id):
    return clause_id.replace("-", "_") if clause_id else clause_id


def resolve_pointer(retrieved: list, gold_cited: str, presence: str) -> dict:
    if presence != "Yes":
        return {"gold_pointer_index": len(retrieved), "cited_clause": None}

    gold_norm = normalize(gold_cited)
    for idx, clause_id in enumerate(retrieved):
        if normalize(clause_id) == gold_norm:
            return {"gold_pointer_index": idx, "cited_clause": gold_cited}

    # Should not happen (verified 0 misses on this source file) -- if the
    # source data ever changes, force-append rather than silently drop.
    return {"gold_pointer_index": len(retrieved), "cited_clause": None,
            "retriever_missed": True, "retrieved": retrieved + [gold_cited]}


def main():
    with open(SOURCE, encoding="utf-8") as f:
        events = json.load(f)

    output = []
    pointer_dist = {"road_surface": {}, "infrastructure": {}}
    presence_dist = {"road_surface": {}, "infrastructure": {}}

    for event in events:
        gold = event["gold_label"]
        record = {
            "event_id": event["event_id"], "evasive_action": event["evasive_action"],
            "clip_path": event["clip_path"], "telemetry_filtered": event["telemetry_filtered"],
            "gps_context": event["gps_context"], "environment": gold["environment"],
            "mm_rag": event.get("mm_rag"),
        }

        for rule, retkey in (("road_surface", "retrieved_irc35"), ("infrastructure", "retrieved_irc67")):
            fold = gold[rule]
            retrieved = gold.get(retkey, [])
            resolved = resolve_pointer(retrieved, fold.get("cited_clause"), fold.get("presence"))

            record[rule] = {
                "presence": fold.get("presence"), "severity": fold.get("severity"),
                "reasoning": fold.get("reasoning"), "retrieved": retrieved,
                **resolved,
            }

            presence_dist[rule][fold.get("presence")] = presence_dist[rule].get(fold.get("presence"), 0) + 1
            pi = resolved["gold_pointer_index"]
            key = "NO_CLAUSE" if pi == len(retrieved) and fold.get("presence") != "Yes" else f"idx_{pi}"
            pointer_dist[rule][key] = pointer_dist[rule].get(key, 0) + 1

        output.append(record)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Done: {len(output)} events -> {OUTPUT}\n")
    for rule in ("road_surface", "infrastructure"):
        print(f"[{rule}] presence distribution: {presence_dist[rule]}")
        print(f"[{rule}] gold_pointer_index distribution: {pointer_dist[rule]}")
        n_yes = presence_dist[rule].get("Yes", 0)
        n_total = len(output)
        print(f"[{rule}] class imbalance ratio (no-clause : has-clause) = "
              f"{n_total - n_yes}:{n_yes} (~{(n_total - n_yes) / max(n_yes, 1):.0f}:1)\n")

    print("WARNING: this imbalance is severe (single-digit positive examples per fold "
          "across 626 events). This is the exact condition that collapsed the old CGPA "
          "attempt (SESSION_SUMMARY.md: 'root cause was deeper -- severe class imbalance'). "
          "train_tagnet_v2.py MUST NOT train pointer_loss naively against this distribution -- "
          "see its presence-gating + oversampling logic before running on Ada.")


if __name__ == "__main__":
    main()
