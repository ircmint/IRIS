"""
run_inference_v1_embed.py

Inference for V1's diagram-accurate embed-mode checkpoint (TelemetryAdapter +
ContextAdapter embedding fusion). The existing run_inference.py only knows
how to generate with plain input_ids (text-mode) -- it has no code path for
splicing adapter pseudo-tokens into inputs_embeds before generation, so a
separate script is needed here rather than reusing it directly.

Same stratified val split (seed=42) as training, so this is genuinely
held-out.

Usage:
    python run_inference_v1_embed.py --checkpoint ~/tagnet/checkpoints_v1_embed_final/epoch_4
"""

import argparse
import csv
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Dataset"))
from train_tagnet_vlm import TagNetVLMForTraining, PROMPT_TEMPLATE_EMBED_MODE  # noqa: E402
from tagnet_vlm import TAGNetVLMConfig  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, "..", "Dataset")
OUTPUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

VALID_PRESENCE = {"Yes", "No", "Unclear"}
REQUIRED_SCHEMA_KEYS = {
    "environment": ["density", "visibility", "road_type", "reasoning"],
    "road_surface": ["presence", "cited_clause", "severity", "reasoning"],
    "infrastructure": ["presence", "cited_clause", "severity", "reasoning"],
}


def normalize_and_validate(parsed: dict):
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


def generate_embed_mode(model_wrapper, clip_path: str, prompt: str, telemetry: dict, gps_context: dict) -> str:
    """Mirrors forward_embed_mode's embedding construction, but calls generate() instead of computing a loss."""
    from qwen_vl_utils import process_vision_info

    user_turn = [{"role": "user", "content": [
        {"type": "video", "video": clip_path, "nframes": 2, "max_pixels": 100352},
        {"type": "text", "text": prompt},
    ]}]
    text = model_wrapper.processor.apply_chat_template(user_turn, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(user_turn)
    inputs = model_wrapper.processor(text=[text], images=imgs, videos=vids,
                                      padding=True, return_tensors="pt").to(model_wrapper.model.device)

    base = model_wrapper.model.get_base_model()
    merged_embeds = base.get_input_embeddings()(inputs["input_ids"])
    telemetry_tok = model_wrapper.telemetry_adapter(model_wrapper.telemetry_to_vec(telemetry))
    context_tok = model_wrapper.context_adapter(model_wrapper.context_to_vec(gps_context))
    adapter_toks = torch.cat([telemetry_tok, context_tok], dim=1)
    inputs_embeds = torch.cat([adapter_toks, merged_embeds], dim=1)
    attention_mask = torch.cat([
        torch.ones(1, 2, dtype=inputs["attention_mask"].dtype, device=model_wrapper.model.device),
        inputs["attention_mask"],
    ], dim=1)

    with torch.no_grad():
        generated = model_wrapper.model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=300, do_sample=False,
            pixel_values_videos=inputs.get("pixel_values_videos"), video_grid_thw=inputs.get("video_grid_thw"))
    # generate() with inputs_embeds returns ONLY the newly generated tokens
    # (no prompt to trim, unlike the input_ids path) -- decode directly.
    return model_wrapper.processor.batch_decode(generated, skip_special_tokens=True)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.expanduser("~/tagnet/checkpoints_v1_embed_final/epoch_4"))
    parser.add_argument("--data", default=os.path.join(DATASET_DIR, "v1_dataset_with_rag_fixed.json"))
    parser.add_argument("--irc35_index", default=os.path.join(DATASET_DIR, "irc35_index_scoped.json"))
    parser.add_argument("--irc67_index", default=os.path.join(DATASET_DIR, "irc67_index_scoped.json"))
    parser.add_argument("--clips_dir", default=os.path.join(DATASET_DIR, "clips"))
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--all_events", action="store_true",
                         help="Run over all events in --data, not just the held-out val split.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        events = json.load(f)
    with open(args.irc35_index, encoding="utf-8") as f:
        irc35_index = json.load(f)
    with open(args.irc67_index, encoding="utf-8") as f:
        irc67_index = json.load(f)
    irc35_lookup = {c["clause_id"]: c["text"] for c in irc35_index}
    irc67_lookup = {c["clause_id"]: c["text"] for c in irc67_index}

    if args.all_events:
        val_events = events
        print(f"Running over ALL events (not just held-out val split): {len(val_events)}")
    else:
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
        print(f"Val events: {len(val_events)}")

    cfg = TAGNetVLMConfig(backbone_name="Qwen/Qwen2.5-VL-3B-Instruct", use_qlora=True)
    model_wrapper = TagNetVLMForTraining(cfg, adapter_mode="embed")
    from peft import PeftModel
    model_wrapper.model = PeftModel.from_pretrained(model_wrapper.model.get_base_model(), args.checkpoint)
    adapters_path = os.path.join(args.checkpoint, "adapters.pt")
    if os.path.exists(adapters_path):
        saved = torch.load(adapters_path, map_location=model_wrapper.model.device)
        model_wrapper.telemetry_adapter.load_state_dict(saved["telemetry_adapter"])
        model_wrapper.context_adapter.load_state_dict(saved["context_adapter"])
    model_wrapper.model.eval()

    predictions = []
    parse_failures = 0
    for i, event in enumerate(val_events):
        clip_path = os.path.join(args.clips_dir, os.path.basename(event["clip_path"]))
        gold = event["gold_label"]
        irc35_ids = gold.get("retrieved_irc35", [])[:1]
        irc67_ids = gold.get("retrieved_irc67", [])[:1]
        irc35_str = "\n".join(f"[{cid}] {irc35_lookup.get(cid, '')[:120]}" for cid in irc35_ids) or "(none)"
        irc67_str = "\n".join(f"[{cid}] {irc67_lookup.get(cid, '')[:120]}" for cid in irc67_ids) or "(none)"
        prompt = PROMPT_TEMPLATE_EMBED_MODE.format(irc35_clauses=irc35_str, irc67_clauses=irc67_str)

        raw_output = None
        try:
            raw_output = generate_embed_mode(model_wrapper, clip_path, prompt,
                                              event["telemetry_filtered"], event["gps_context"])
            cleaned = raw_output.strip().strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
            parsed = json.loads(cleaned, strict=False)
            fail_reason = normalize_and_validate(parsed)
        except Exception as exc:
            parsed = None
            fail_reason = f"parse_error:{type(exc).__name__}:{exc}"

        if fail_reason:
            parse_failures += 1
        predictions.append({"event_id": event["event_id"], "raw_output": raw_output,
                             "parsed": parsed, "fail_reason": fail_reason, "gold": gold})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(val_events)} done, {parse_failures} parse failures so far")

    out_path = args.out or os.path.join(BASE, "predictions_v1_embed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    print(f"\nDone: {len(val_events)} val events, {parse_failures} failures ({parse_failures/len(val_events)*100:.1f}%)")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
