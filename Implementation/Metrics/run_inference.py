"""
run_inference.py  --  run on Ada (needs the trained checkpoint + GPU).

Runs the trained epoch_2 LoRA checkpoint (text-injection mode, see
TAGNET_PROJECT_STATUS.md for why) over the exact same validation split used
during training (val_events = events[:n_val], n_val = int(len*0.15) --
replicated from train_tagnet_vlm.py's main() so this is a genuine held-out
eval, not data the model was trained on).

For each val event:
  1. Build the same text-mode prompt used in training (telemetry/GPS
     serialized into text, top-1 retrieved clause per fold).
  2. Generate (not teacher-forced -- this is real inference).
  3. Parse + validate the three-fold JSON (same schema gates as labeling).
  4. Record prediction alongside the gold label for that event.

Outputs:
  predictions.json        -- full prediction+gold detail per event
  outputs/verdicts.csv     -- for compute_chr.py / compute_rhr.py / compute_haa.py
  outputs/gold_verdicts.csv-- for compute_haa.py
  outputs/event_compliance.csv -- for compute_icdi.py
  outputs/retrieval_results.csv -- for compute_precision.py (human_relevant
                                    left BLANK -- no manual annotation exists,
                                    see caveat printed at the end)

Usage:
    python run_inference.py --checkpoint ~/tagnet/checkpoints/epoch_2
"""

import argparse
import csv
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Dataset"))
from train_tagnet_vlm import TagNetVLMDataset, collate_and_prompt  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(BASE), "Dataset")
OUTPUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_PRESENCE = {"Yes", "No", "Unclear"}
REQUIRED_SCHEMA_KEYS = {
    "environment": ["density", "visibility", "road_type", "reasoning"],
    "road_surface": ["presence", "cited_clause", "severity", "reasoning"],
    "infrastructure": ["presence", "cited_clause", "severity", "reasoning"],
}


def normalize_and_validate(parsed: dict) -> str | None:
    for fold in ("road_surface", "infrastructure"):
        if fold not in parsed:
            return f"missing_fold:{fold}"
        presence = parsed[fold].get("presence")
        if isinstance(presence, str):
            parsed[fold]["presence"] = presence.strip().capitalize()
        severity = parsed[fold].get("severity")
        if isinstance(severity, str):
            parsed[fold]["severity"] = severity.strip().upper()
        for k in REQUIRED_SCHEMA_KEYS[fold]:
            if k not in parsed[fold]:
                return f"missing_key:{fold}.{k}"
        if parsed[fold]["presence"] not in VALID_PRESENCE:
            return f"invalid_presence:{fold}"
    return None


def load_model(checkpoint_dir: str):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    from peft import PeftModel

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", quantization_config=bnb, device_map="auto")
    model = PeftModel.from_pretrained(base, checkpoint_dir)
    model.eval()
    return model, processor


def generate_one(model, processor, clip_path: str, prompt: str) -> str:
    from qwen_vl_utils import process_vision_info
    messages = [{
        "role": "user",
        "content": [{"type": "video", "video": clip_path, "nframes": 2, "max_pixels": 100352},
                     {"type": "text", "text": prompt}],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    inputs = processor(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    trimmed = generated[0][inputs["input_ids"].shape[1]:]
    return processor.decode(trimmed, skip_special_tokens=True)


SEVERITY_NUM = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
ABSENCE_FROM_PRESENCE = {"Yes": 1.0, "Unclear": 0.5, "No": 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.expanduser("~/tagnet/checkpoints_v1_rag_5ep/epoch_4"))
    parser.add_argument("--data", default=os.path.join(DATASET_DIR, "v1_dataset_with_rag_fixed.json"))
    parser.add_argument("--irc35_index", default=os.path.join(DATASET_DIR, "irc35_index_scoped.json"))
    parser.add_argument("--irc67_index", default=os.path.join(DATASET_DIR, "irc67_index_scoped.json"))
    parser.add_argument("--clips_dir", default=os.path.join(DATASET_DIR, "clips"))
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        events = json.load(f)
    with open(args.irc35_index, encoding="utf-8") as f:
        irc35_index = json.load(f)
    with open(args.irc67_index, encoding="utf-8") as f:
        irc67_index = json.load(f)

    # Stratified split, SAME logic/seed as train_tagnet_vlm.py's main() --
    # an unshuffled prefix split previously landed val entirely on one
    # class; this must match training's actual held-out set exactly.
    import random
    random.seed(42)
    by_stratum = {}
    for e in events:
        key = (e["gold_label"]["road_surface"]["presence"], e["gold_label"]["infrastructure"]["presence"])
        by_stratum.setdefault(key, []).append(e)
    val_events = []
    for key, group in by_stratum.items():
        g = group[:]
        random.shuffle(g)
        n_val_group = max(1, int(len(g) * args.val_fraction)) if len(g) > 1 else 0
        val_events.extend(g[:n_val_group])
    random.shuffle(val_events)
    if args.limit:
        val_events = val_events[: args.limit]
    print(f"Running inference on {len(val_events)} held-out validation events "
          f"(matches training's val split exactly)")

    model, processor = load_model(args.checkpoint)
    ds = TagNetVLMDataset(val_events, irc35_index, irc67_index, clips_dir=args.clips_dir)

    predictions = []
    verdict_rows, gold_rows, compliance_rows, retrieval_rows = [], [], [], []
    parse_failures = 0

    for i in range(len(val_events)):
        item, prompt = collate_and_prompt([ds[i]], "text")
        event = val_events[i]
        gold = event["gold_label"]

        raw_output = generate_one(model, processor, item["clip_path"], prompt)
        try:
            cleaned = raw_output.strip().strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
            parsed = json.loads(cleaned, strict=False)
            fail_reason = normalize_and_validate(parsed)
        except Exception as exc:
            parsed = None
            fail_reason = f"parse_error:{type(exc).__name__}"

        predictions.append({
            "event_id": event["event_id"], "raw_output": raw_output,
            "parsed": parsed, "fail_reason": fail_reason, "gold": gold,
        })
        if fail_reason:
            parse_failures += 1
            print(f"  [{i}] event {event['event_id']}: FAILED ({fail_reason})")
            if i < 3:
                print(f"    RAW OUTPUT: {raw_output[:500]!r}")
            continue

        for rule in ("road_surface", "infrastructure"):
            pred_fold = parsed[rule]
            gold_fold = gold[rule]
            pred_verdict = "NON_COMPLIANT" if pred_fold["presence"] == "Yes" else "COMPLIANT"
            gold_verdict = "NON_COMPLIANT" if gold_fold["presence"] == "Yes" else "COMPLIANT"

            verdict_rows.append({
                "event_id": event["event_id"], "frame_id": "0", "rule": rule,
                "verdict": pred_verdict, "cited_clause": pred_fold.get("cited_clause") or "",
                "reasoning": pred_fold.get("reasoning", ""),
            })
            gold_rows.append({"event_id": event["event_id"], "frame_id": "0", "gold_verdict": gold_verdict})

            retrieved_key = "retrieved_irc35" if rule == "road_surface" else "retrieved_irc67"
            retrieved = gold.get(retrieved_key, [])
            retrieval_score = 1.0 if pred_fold.get("cited_clause") in retrieved else 0.5

            compliance_rows.append({
                "event_id": event["event_id"], "start_time": event.get("clip_start_s", ""),
                "end_time": event.get("clip_end_s", ""),
                "zone_type": event.get("gps_context", {}).get("zone_type", "Unknown"),
                "absence_confidence": ABSENCE_FROM_PRESENCE.get(pred_fold["presence"], 0.5),
                "severity": SEVERITY_NUM.get(pred_fold.get("severity", "LOW"), 1),
                "retrieval_score": retrieval_score, "rule": rule,
            })

            for rank, clause_id in enumerate(retrieved, start=1):
                retrieval_rows.append({
                    "event_id": f"{event['event_id']}_{rule}", "retrieval_rank": rank,
                    "clause_id": clause_id, "human_relevant": "",  # NOT annotated, see caveat
                })

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(val_events)} done, {parse_failures} parse failures so far")

    with open(os.path.join(BASE, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    def write_csv(rows, filename, fieldnames):
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {filename}: {len(rows)} rows -> {path}")

    write_csv(verdict_rows, "verdicts.csv", ["event_id", "frame_id", "rule", "verdict", "cited_clause", "reasoning"])
    write_csv(gold_rows, "gold_verdicts.csv", ["event_id", "frame_id", "gold_verdict"])
    write_csv(compliance_rows, "event_compliance.csv",
              ["event_id", "start_time", "end_time", "zone_type", "absence_confidence", "severity", "retrieval_score", "rule"])
    write_csv(retrieval_rows, "retrieval_results.csv", ["event_id", "retrieval_rank", "clause_id", "human_relevant"])

    print(f"\nDone: {len(val_events)} val events, {parse_failures} schema/parse failures "
          f"({parse_failures/len(val_events)*100:.1f}%)")
    print("CAVEAT: retrieval_results.csv's human_relevant column is EMPTY -- no manual "
          "relevance annotation exists in this project, so compute_precision.py will report "
          "0 annotated events until someone hand-labels a sample. Do not fabricate this.")


if __name__ == "__main__":
    main()
