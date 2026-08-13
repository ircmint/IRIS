"""
train_tagnet_v2.py  --  RUN ON ADA. NOT executable in this sandbox (local
torch install here is broken -- DLL load failure unrelated to this code --
and training needs real GPU memory regardless). Structure mirrors
TAGNET/V1/code/train_tagnet_vlm.py, which DID train successfully on Ada
(job 2658600, gnode050) -- same smoke-test-first workflow, same stratified
split + oversampling pattern (that part is proven, not new).

SMOKE-TEST FIRST, exactly like V1 did:
    python train_tagnet_v2.py --smoke_test --adapter_mode embed
    # if that raises a shape-mismatch error in the embedding splice:
    python train_tagnet_v2.py --smoke_test --adapter_mode text

WHAT THIS TRAINS:
  Qwen2-VL-2B-Instruct backbone (QLoRA 4-bit NF4, rank=16/alpha=32,
  dropout=0.05, Tier 1 q_proj/v_proj) + TelemetryAdapter + ContextAdapter +
  AdaptiveModalityGate (AMG) + ClauseEncoder + ClauseGroundedPointerAttention
  (CGPA), all from tagnet_v2_final.py. Loss = LM teacher-forced loss (the
  three-fold JSON, cited_clause always null in the text target -- CGPA
  resolves it separately) + presence-gated pointer_loss + a small AMG
  entropy regularizer.

THE ONE THING THIS SCRIPT EXISTS TO GET RIGHT (see build_v2_dataset.py):
  v2_dataset.json has only 11 (road_surface) / 8 (infrastructure) positive
  events out of 626 -- a ~56:1 / ~77:1 imbalance. This is the exact
  documented cause of the old CGPA attempt's collapse (SESSION_SUMMARY.md:
  persisted even after fixing the original label-masking bug). Two
  independent mitigations, not just one:
    1. PRESENCE GATE (tagnet_v2_final.py's should_invoke_cgpa): pointer_loss
       is only computed on rows where gold presence=="Yes" for that fold --
       CGPA never has to learn the skewed "is there a clause at all"
       decision, only "which one" among already-positive events.
    2. OVERSAMPLING (same technique V1's own trainer already validated,
       see train_tagnet_vlm.py lines ~355-366): any event with presence
       =="Yes" in either fold is repeated in the training stream.

Usage:
    python train_tagnet_v2.py --data v2_dataset.json \
        --clauses_jsonl RAG/ingest/clauses.jsonl \
        --output_dir ./checkpoints --epochs 3 --grad_accum_steps 8 --adapter_mode embed
"""

import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tagnet_v2_final import (  # noqa: E402
    TAGNetVLMConfig, TelemetryAdapter, ContextAdapter, AdaptiveModalityGate,
    ClauseEncoder, ClauseGroundedPointerAttention, apply_gate, gate_entropy_regularizer,
    pointer_loss, should_invoke_cgpa, GateDiversityTracker, ReasoningClauseAligner,
    entailment_alignment_loss,
)

OVERSAMPLE_FACTOR = 25  # higher than V1's 15 -- V2's positive count (11/8) is smaller than V1's (11 total)


# =============================================================================
# DATASET
# =============================================================================

class TagNetV2Dataset(Dataset):
    """
    Wraps v2_dataset.json events (see build_v2_dataset.py) into training items.

    CLAUSE TEXT SOURCE (real bug found and fixed on re-audit): v2_dataset.json's
    `retrieved` clause IDs are underscore-formatted (e.g. "IRC35_4.2.1"),
    built from Chroma metadata as f"{doc}_{clause}" by
    retrieve_from_knowledge_rag() / build_cgpa_dataset.py's retrieve(). The
    master clause index (Dataset/irc35_index.json / irc67_index.json) uses a
    DIFFERENT, hyphen-formatted, PDF-regex-extracted ID namespace
    ("IRC35-141") that does not correspond 1:1 with the RAG's ingested
    clauses. An earlier version of this dataset class looked text up in that
    wrong index -- verified: 0/10 sampled real IDs matched, meaning every
    clause fed to ClauseEncoder during training would have been the bare ID
    string, not real clause text, silently crippling CGPA's ability to learn
    semantic matching (no crash, so nothing would have caught this before a
    full Ada run completed and produced suspiciously bad pointer accuracy).
    Fixed: clause text now comes from RAG/ingest/clauses.jsonl (the actual
    text Chroma indexed), keyed the same way retrieval builds IDs -- verified
    0/6328 real retrieved IDs across the full dataset are missing from it.
    """

    def __init__(self, events: list, clause_text: dict, clips_dir: str):
        self.events = events
        self.clause_text = clause_text  # {"IRC35_4.2.1": "...", ...} from RAG/ingest/clauses.jsonl
        self.clips_dir = clips_dir

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        event = self.events[idx]
        clip_path = os.path.join(self.clips_dir, os.path.basename(event["clip_path"]))

        rs, infra = event["road_surface"], event["infrastructure"]
        rs_clauses = [{"clause_id": cid, "text": self.clause_text.get(cid, "")[:200]}
                      for cid in rs["retrieved"]]
        infra_clauses = [{"clause_id": cid, "text": self.clause_text.get(cid, "")[:200]}
                         for cid in infra["retrieved"]]

        target = {
            "environment": event["environment"],
            "road_surface": {"presence": rs["presence"], "cited_clause": None,
                              "severity": rs["severity"], "reasoning": rs["reasoning"]},
            "infrastructure": {"presence": infra["presence"], "cited_clause": None,
                                "severity": infra["severity"], "reasoning": infra["reasoning"]},
        }

        # Gold cited-clause TEXT (not just ID) for the experimental
        # reasoning-entailment auxiliary loss -- looked up the same way
        # rs_clauses/infra_clauses are, from RAG/ingest/clauses.jsonl.
        rs_cited_text = self.clause_text.get(rs.get("cited_clause") or "", "")
        infra_cited_text = self.clause_text.get(infra.get("cited_clause") or "", "")

        return {
            "clip_path": clip_path,
            "telemetry": event["telemetry_filtered"],
            "gps_context": event["gps_context"],
            "mm_rag_hint": format_mmrag_hint(event.get("mm_rag")),
            "rs_clauses": rs_clauses, "infra_clauses": infra_clauses,
            "rs_gold_pointer": rs["gold_pointer_index"], "rs_presence": rs["presence"],
            "infra_gold_pointer": infra["gold_pointer_index"], "infra_presence": infra["presence"],
            "rs_cited_text": rs_cited_text, "infra_cited_text": infra_cited_text,
            "target_json": json.dumps(target, separators=(",", ":")),
        }


def format_mmrag_hint(mm_rag: dict) -> str:
    """
    Turns v2_dataset.json's mm_rag field into a short prompt line so the
    Multimodal RAG signal is actually used during training, not just carried
    as unused metadata. HONEST SCOPE LIMIT preserved in the wording itself
    (see build_mmrag_dataset.py / V2_STATUS.md): plate_index.pkl crops carry
    no clause_id linkage, so this is presented to the model as a visual-
    similarity hint only, never as a citation or as something CGPA consumes
    -- it's an extra piece of evidence for the free-text presence/severity/
    reasoning fields, same tier as telemetry/context, not a ground-truth
    signal.
    """
    if not mm_rag:
        return "(no MM RAG detection attempted for this event)"
    parts = []
    for key, label in (("sign_match", "sign"), ("marking_match", "marking")):
        m = mm_rag.get(key)
        if m:
            parts.append(f"detected {label} visually resembles reference plate "
                          f"'{m.get('source_page', '?')}' (similarity={m.get('similarity', 0):.2f})")
    if not parts:
        return "(no confident MM RAG match; classical-CV detector found no candidate above threshold)"
    return "; ".join(parts) + " -- visual-similarity hint only, NOT a resolved citation"


# REAL BUG FIXED HERE (found by inspecting a raw failing generation, not
# hypothetical): the old template said "Respond ONLY in the required JSON
# schema" but never actually SHOWED the schema anywhere in the prompt --
# the model only ever learned it implicitly from teacher-forced targets,
# with nothing to anchor against at generation time. Observed failure mode
# matched exactly: the environment fold dropped its real keys
# (visibility, reasoning) and invented fields belonging to OTHER folds
# (cited_clause, time_of_day) -- schema bleed between folds. Fixed by
# spelling out the exact schema inline, per fold, matching
# tagnet_v2_final.py's THREE_FOLD_PROMPT_TEMPLATE (which already had this
# right -- this file's simplified copy was the one that regressed it).
PROMPT_TEMPLATE = """You are auditing a road scene for IRC compliance across THREE independent folds.
Reason about each fold separately. Each fold has its OWN exact set of keys -- do not mix keys between folds.

--- FOLD 1: ENVIRONMENT --- (keys: density, visibility, road_type, reasoning -- NO cited_clause here)
--- FOLD 2: ROAD SURFACE (IRC 35) --- candidates: {irc35_clauses}
  (keys: presence, cited_clause, severity, reasoning)
--- FOLD 3: INFRASTRUCTURE (IRC 67) --- candidates: {irc67_clauses}
  (keys: presence, cited_clause, severity, reasoning)

Telemetry: {telemetry_summary}
Context: {context_summary}
MM RAG visual hint (Multimodal RAG, catalog similarity only -- use only as supporting evidence
for presence/severity/reasoning, never as a citation): {mm_rag_hint}

Respond ONLY in this exact JSON schema. Leave "cited_clause" as null -- it is
resolved separately by the pointer-attention mechanism, not generated here.
{{
  "environment": {{"density": "...", "visibility": "...", "road_type": "...", "reasoning": "..."}},
  "road_surface": {{"presence": "Yes/No/Unclear", "cited_clause": null, "severity": "LOW/MEDIUM/HIGH/CRITICAL", "reasoning": "..."}},
  "infrastructure": {{"presence": "Yes/No/Unclear", "cited_clause": null, "severity": "LOW/MEDIUM/HIGH/CRITICAL", "reasoning": "..."}}
}}
"""


def stratified_split_and_oversample(events: list, val_fraction: float, seed: int = 42):
    """Same technique as V1's train_tagnet_vlm.py (proven on Ada, job 2658600) --
    stratify by (road_surface.presence, infrastructure.presence) so the tiny
    positive class isn't accidentally absent from val, then oversample
    positives in TRAIN ONLY."""
    import random
    random.seed(seed)
    by_stratum = {}
    for e in events:
        key = (e["road_surface"]["presence"], e["infrastructure"]["presence"])
        by_stratum.setdefault(key, []).append(e)

    val_events, train_events = [], []
    for key, group in by_stratum.items():
        random.shuffle(group)
        n_val = max(1, int(len(group) * val_fraction)) if len(group) > 1 else 0
        val_events.extend(group[:n_val])
        train_events.extend(group[n_val:])

    oversampled = []
    for e in train_events:
        is_minority = e["road_surface"]["presence"] == "Yes" or e["infrastructure"]["presence"] == "Yes"
        oversampled.extend([e] * (OVERSAMPLE_FACTOR if is_minority else 1))
    random.shuffle(oversampled)
    random.shuffle(val_events)
    return oversampled, val_events


# =============================================================================
# MODEL WRAPPER
# =============================================================================

class TagNetV2ForTraining(nn.Module):
    def __init__(self, cfg: TAGNetVLMConfig, adapter_mode: str):
        super().__init__()
        self.cfg = cfg
        self.adapter_mode = adapter_mode

        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(cfg.backbone_name)
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg.backbone_name, quantization_config=bnb_config, device_map="auto")
        base_model.config.use_cache = False
        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})

        text_config = base_model.config.get_text_config()
        real_embed_dim = text_config.hidden_size
        if real_embed_dim != cfg.embed_dim:
            print(f"WARNING: cfg.embed_dim={cfg.embed_dim} but model.config.hidden_size="
                  f"{real_embed_dim}. Using the real value.")
            cfg.embed_dim = real_embed_dim

        lora_config = LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                                  target_modules=cfg.lora_target_modules, bias="none", task_type="CAUSAL_LM")
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()

        dev, dt = base_model.device, torch.bfloat16
        self.telemetry_adapter = TelemetryAdapter(4, cfg.telemetry_adapter_hidden, cfg.embed_dim).to(dev, dt)
        self.context_adapter = ContextAdapter(6, cfg.context_adapter_hidden, cfg.embed_dim).to(dev, dt)
        self.amg = AdaptiveModalityGate(cfg.embed_dim).to(dev, dt)
        self.clause_encoder = ClauseEncoder(cfg.sentence_embed_dim, cfg.embed_dim).to(dev, dt)
        self.cgpa = ClauseGroundedPointerAttention(cfg.embed_dim, cfg.cgpa_n_heads).to(dev, dt)
        # Experimental, targets RHR -- see ReasoningClauseAligner's docstring
        # for the honest caveat (root cause is likely gold label quality,
        # this is a best-effort nudge, not a guaranteed fix).
        self.reasoning_aligner = ReasoningClauseAligner(cfg.embed_dim, cfg.sentence_embed_dim).to(dev, dt)
        # Real fix for AMG's collapse (see GateDiversityTracker's docstring)
        self.gate_tracker = GateDiversityTracker()

        from sentence_transformers import SentenceTransformer
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(dev))

    def embed_clauses(self, clauses: list) -> torch.Tensor:
        if not clauses:
            return torch.zeros(1, 0, self.cfg.sentence_embed_dim, device=self.model.device, dtype=torch.bfloat16)
        texts = [c["text"] or c["clause_id"] for c in clauses]
        with torch.no_grad():
            emb = self.sentence_model.encode(texts, convert_to_tensor=True)
        return emb.unsqueeze(0).to(self.model.device, torch.bfloat16)

    def telemetry_to_vec(self, telemetry: dict) -> torch.Tensor:
        return torch.tensor([[telemetry.get("peak_lat_accel", 0.0), telemetry.get("peak_lon_accel", 0.0),
                               telemetry.get("peak_yaw_rate", 0.0), telemetry.get("t_peak", 0.0)]],
                             dtype=torch.bfloat16, device=self.model.device)

    def context_to_vec(self, gps_context: dict) -> torch.Tensor:
        zone_onehot = {"highway": 0, "urban_arterial": 1, "congested_urban": 2, "stationary": 3, "unknown": 4}
        zone_vec = [0.0] * 5
        zone_vec[zone_onehot.get(gps_context.get("zone_type", "unknown"), 4)] = 1.0
        speed = gps_context.get("speed_ms") or 0.0
        return torch.tensor([zone_vec + [speed]], dtype=torch.bfloat16, device=self.model.device)

    def pointer_loss_for_item(self, item, gen_hidden_states) -> torch.Tensor:
        """
        Computes pointer_loss ONLY for folds where gold presence=="Yes"
        (should_invoke_cgpa) -- see module docstring. Returns 0.0 (no grad
        contribution) if neither fold is positive for this item, which is
        the common case; oversampling (OVERSAMPLE_FACTOR) is what ensures
        positive items still show up often enough across an epoch.

        DTYPE FIX (found on live Ada smoke test, real bug): `gen_hidden_states`
        comes from outputs.hidden_states[-1][:, -1, :], which HF returns in
        Float32 regardless of the model's compute dtype -- but clause_encoder
        and cgpa were cast to bfloat16 in __init__. nn.Linear requires
        matching dtypes, so this raised "mat1 and mat2 must have the same
        dtype, but got Float and BFloat16" the moment pointer_loss_for_item
        was actually exercised (self-test's mock tensors never caught this,
        since they were never run through a real HF forward pass). Cast the
        query once, here, rather than at every call site.
        """
        gen_hidden_states = gen_hidden_states.to(torch.bfloat16)
        total = torch.tensor(0.0, device=self.model.device)
        n = 0
        for clauses, gold_ptr, presence, h in [
            (item["rs_clauses"], item["rs_gold_pointer"], item["rs_presence"], gen_hidden_states),
            (item["infra_clauses"], item["infra_gold_pointer"], item["infra_presence"], gen_hidden_states),
        ]:
            if not should_invoke_cgpa(presence) or not clauses:
                continue
            clause_embeds = self.clause_encoder(self.embed_clauses(clauses))
            attn_weights, _ = self.cgpa(h, clause_embeds)
            gold_idx = torch.tensor([gold_ptr], device=self.model.device)
            total = total + pointer_loss(attn_weights, gold_idx)
            n += 1
        return total / n if n else total

    def entailment_loss_for_item(self, item, gen_hidden_states) -> torch.Tensor:
        """
        EXPERIMENTAL (see ReasoningClauseAligner's docstring for the honest
        caveat). Presence-gated the same way pointer_loss_for_item is --
        only rows where a real clause applies have a real clause embedding
        to align toward.
        """
        gen_hidden_states = gen_hidden_states.to(torch.bfloat16)
        total = torch.tensor(0.0, device=self.model.device)
        n = 0
        for cited_text, presence in [
            (item["rs_cited_text"], item["rs_presence"]),
            (item["infra_cited_text"], item["infra_presence"]),
        ]:
            if not should_invoke_cgpa(presence) or not cited_text:
                continue
            with torch.no_grad():
                gold_embed = self.sentence_model.encode([cited_text], convert_to_tensor=True)
            gold_embed = gold_embed.to(self.model.device, torch.bfloat16)
            projected = self.reasoning_aligner(gen_hidden_states)
            total = total + entailment_alignment_loss(projected, gold_embed)
            n += 1
        return total / n if n else total

    def forward_text_mode(self, item):
        from qwen_vl_utils import process_vision_info
        prompt = PROMPT_TEMPLATE.format(
            irc35_clauses=", ".join(c["clause_id"] for c in item["rs_clauses"]) or "(none)",
            irc67_clauses=", ".join(c["clause_id"] for c in item["infra_clauses"]) or "(none)",
            telemetry_summary=json.dumps(item["telemetry"]), context_summary=json.dumps(item["gps_context"]),
            mm_rag_hint=item["mm_rag_hint"])

        user_turn = [{"role": "user", "content": [
            {"type": "video", "video": item["clip_path"], "nframes": 2, "max_pixels": 100352},
            {"type": "text", "text": prompt}]}]
        image_inputs, video_inputs = process_vision_info(user_turn)
        prefix_text = self.processor.apply_chat_template(user_turn, tokenize=False, add_generation_prompt=True)
        prefix_inputs = self.processor(text=[prefix_text], images=image_inputs, videos=video_inputs,
                                        padding=True, return_tensors="pt").to(self.model.device)
        prefix_len = prefix_inputs["input_ids"].shape[1]

        full_conv = user_turn + [{"role": "assistant", "content": item["target_json"]}]
        full_text = self.processor.apply_chat_template(full_conv, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=[full_text], images=image_inputs, videos=video_inputs,
                                 padding=True, return_tensors="pt").to(self.model.device)

        labels = inputs["input_ids"].clone()
        labels[:, :prefix_len] = -100
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100

        outputs = self.model(**inputs, labels=labels, output_hidden_states=True)
        h_last = outputs.hidden_states[-1][:, -1, :]  # same degraded-but-functioning fallback as inference
        ptr_loss = self.pointer_loss_for_item(item, h_last)
        entail_loss = self.entailment_loss_for_item(item, h_last)
        # text mode never computes AMG at all (telemetry/context go into the
        # text prompt directly, no embedding splice) -- no gate to track here.
        return outputs.loss, ptr_loss, entail_loss, torch.tensor(0.0, device=self.model.device)

    def forward_embed_mode(self, item):
        """AMG-gated telemetry/context tokens spliced into inputs_embeds.
        NEEDS SMOKE-TEST (same caveat as V1's forward_embed_mode) -- video-
        token merge conventions are transformers-version-sensitive."""
        from qwen_vl_utils import process_vision_info
        prompt = PROMPT_TEMPLATE.format(
            irc35_clauses=", ".join(c["clause_id"] for c in item["rs_clauses"]) or "(none)",
            irc67_clauses=", ".join(c["clause_id"] for c in item["infra_clauses"]) or "(none)",
            telemetry_summary="(fused, see AMG tokens)", context_summary="(fused, see AMG tokens)",
            mm_rag_hint=item["mm_rag_hint"])

        user_turn = [{"role": "user", "content": [
            {"type": "video", "video": item["clip_path"], "nframes": 2, "max_pixels": 100352},
            {"type": "text", "text": prompt}]}]
        image_inputs, video_inputs = process_vision_info(user_turn)
        prefix_text = self.processor.apply_chat_template(user_turn, tokenize=False, add_generation_prompt=True)
        prefix_inputs = self.processor(text=[prefix_text], images=image_inputs, videos=video_inputs,
                                        padding=True, return_tensors="pt").to(self.model.device)
        prefix_len = prefix_inputs["input_ids"].shape[1]

        full_conv = user_turn + [{"role": "assistant", "content": item["target_json"]}]
        full_text = self.processor.apply_chat_template(full_conv, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=[full_text], images=image_inputs, videos=video_inputs,
                                 padding=True, return_tensors="pt").to(self.model.device)

        base = self.model.get_base_model()
        merged_embeds = base.get_input_embeddings()(inputs["input_ids"])
        vision_out = base.visual(inputs["pixel_values_videos"], grid_thw=inputs["video_grid_thw"])

        # REAL BUG FOUND ON LIVE ADA EVAL (not hypothetical): transformers'
        # own Qwen2VLForConditionalGeneration.forward() only does the
        # video-feature masked_scatter merge inside an `if inputs_embeds is
        # None:` block (verified by reading the installed v4.45.2 source
        # directly). Since this code always passes inputs_embeds explicitly,
        # that block never ran -- pixel_values_videos/video_grid_thw were
        # silently ignored by the model itself, meaning `merged_embeds`
        # above only ever contained placeholder-token embeddings, not real
        # visual content. The completed 2-epoch training run trained the
        # decoder BLIND to video (AMG's gating decision still saw real
        # vision features via vision_out for the line below -- only the
        # decoder's input sequence was missing them). Fixed by replicating
        # transformers' own merge logic manually, using the real
        # video_token_id from config (not hardcoded).
        video_mask = (inputs["input_ids"] == base.config.video_token_id).unsqueeze(-1).expand_as(merged_embeds)
        merged_embeds = merged_embeds.masked_scatter(video_mask, vision_out.to(merged_embeds.dtype))
        video_summary = vision_out.mean(dim=0, keepdim=True).to(torch.bfloat16)

        telemetry_tok = self.telemetry_adapter(self.telemetry_to_vec(item["telemetry"]))
        context_tok = self.context_adapter(self.context_to_vec(item["gps_context"]))
        gate_out = self.amg(video_summary, telemetry_tok.mean(dim=1), context_tok.mean(dim=1))
        gated_telemetry, gated_context = apply_gate(
            telemetry_tok, context_tok, gate_out["gate_telemetry"], gate_out["gate_context"])
        adapter_toks = torch.cat([gated_telemetry, gated_context], dim=1)

        inputs_embeds = torch.cat([adapter_toks, merged_embeds], dim=1)
        attention_mask = torch.cat([
            torch.ones(1, 2, dtype=inputs["attention_mask"].dtype, device=self.model.device),
            inputs["attention_mask"]], dim=1)
        labels = inputs["input_ids"].clone()
        labels[:, :prefix_len] = -100
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels = torch.cat([torch.full((1, 2), -100, dtype=labels.dtype, device=self.model.device), labels], dim=1)

        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels,
                              pixel_values_videos=inputs.get("pixel_values_videos"),
                              video_grid_thw=inputs.get("video_grid_thw"), output_hidden_states=True)
        h_last = outputs.hidden_states[-1][:, -1, :]
        ptr_loss = self.pointer_loss_for_item(item, h_last)
        entail_loss = self.entailment_loss_for_item(item, h_last)

        entropy_reg = gate_entropy_regularizer(gate_out)
        # Real AMG-collapse fix: update the running EMA with THIS step's
        # gate_telemetry, then reward this step's gate for differing from
        # that recent average -- see GateDiversityTracker's docstring for
        # why entropy_reg alone couldn't detect/prevent the collapse.
        self.gate_tracker.update(float(gate_out["gate_telemetry"].item()))
        diversity_loss = self.gate_tracker.diversity_loss(gate_out["gate_telemetry"])
        # Weights raised 10x-6x vs the previous run (0.01/0.05 -> 0.1/0.3).
        # Previously these regularizers' gradients died inside the saturated
        # softmax before reaching gate_mlp's weights at all, so their exact
        # weight barely mattered. Now that AdaptiveModalityGate normalizes
        # its logits before softmax (see tagnet_v2_final.py), the gradient
        # path is live again, and since the primary LM loss gives ~0 signal
        # to AMG's split on its own, these need real weight to be the thing
        # that actually drives the gate to depend on its input.
        gate_reg = 0.1 * entropy_reg + 0.3 * diversity_loss

        return outputs.loss, ptr_loss, entail_loss, gate_reg


def load_clause_text(clauses_jsonl_path: str) -> dict:
    """Builds {"IRC35_4.2.1": "clause text...", ...} from RAG/ingest/clauses.jsonl
    -- the same underscore ID scheme retrieve_from_knowledge_rag() and
    build_v2_dataset.py's `retrieved` lists use. See TagNetV2Dataset's
    docstring: the master index files (Dataset/irc35_index.json /
    irc67_index.json) use an incompatible hyphen ID namespace and must NOT
    be used here -- verified 0/6328 real retrieved IDs would match there."""
    clause_text = {}
    with open(clauses_jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            clause_text[f"{rec['doc']}_{rec['clause']}"] = rec["text"]
    return clause_text


def evaluate_val_loss(model: "TagNetV2ForTraining", val_ds: TagNetV2Dataset, adapter_mode: str, limit: int) -> float:
    """
    Quick per-epoch early-stopping signal -- teacher-forced LM loss (no
    backward, no generation) over the first `limit` held-out val events.
    Deliberately a SUBSET, not the full val split: this runs once per
    epoch across up to 8 epochs, so keeping it fast matters; the real,
    full 94-event generation-based eval (run_inference_v2.py + all metric
    scripts) is a separate, slower run done once at the end, not repeated
    per epoch.
    """
    model.model.eval()
    total_loss, n = 0.0, 0
    fwd = model.forward_embed_mode if adapter_mode == "embed" else model.forward_text_mode
    with torch.no_grad():
        for i in range(min(limit, len(val_ds))):
            item = val_ds[i]
            lm_loss, _, _, _ = fwd(item)
            total_loss += lm_loss.item()
            n += 1
    model.model.train()
    return total_loss / n if n else float("inf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="v2_dataset.json")
    parser.add_argument("--clauses_jsonl", default="RAG/ingest/clauses.jsonl")
    parser.add_argument("--clips_dir", default="../Dataset/clips")
    parser.add_argument("--output_dir", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--adapter_mode", choices=["embed", "text"], default="embed")
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=2,
                         help="Stop if quick val loss hasn't improved for this many consecutive epochs.")
    parser.add_argument("--val_check_limit", type=int, default=20,
                         help="Number of val events used for the quick per-epoch early-stop check "
                              "(not the full eval -- that's a separate, slower pipeline run afterward).")
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        events = json.load(f)
    clause_text = load_clause_text(args.clauses_jsonl)
    print(f"Loaded {len(clause_text)} clause texts from {args.clauses_jsonl}")

    train_events, val_events = stratified_split_and_oversample(events, args.val_fraction)
    print(f"Train (post-oversample x{OVERSAMPLE_FACTOR}): {len(train_events)} | Val: {len(val_events)}")
    from collections import Counter
    print("Val road_surface presence:", Counter(e["road_surface"]["presence"] for e in val_events))
    print("Val infrastructure presence:", Counter(e["infrastructure"]["presence"] for e in val_events))

    train_ds = TagNetV2Dataset(train_events, clause_text, args.clips_dir)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=lambda b: b[0])
    val_ds = TagNetV2Dataset(val_events, clause_text, args.clips_dir)

    cfg = TAGNetVLMConfig(backbone_name="Qwen/Qwen2-VL-2B-Instruct")
    model = TagNetV2ForTraining(cfg, adapter_mode=args.adapter_mode)

    trainable = [p for p in model.model.parameters() if p.requires_grad]
    trainable += list(model.telemetry_adapter.parameters()) + list(model.context_adapter.parameters())
    trainable += list(model.clause_encoder.parameters()) + list(model.cgpa.parameters())
    trainable += list(model.reasoning_aligner.parameters())
    # AMG's gate_mlp gets its own param group with much stronger weight decay
    # (0.1 vs the default 0.01 everywhere else). Reason: AdaptiveModalityGate
    # now LayerNorm-bounds its logits before softmax (see tagnet_v2_final.py),
    # which keeps the gate's OUTPUT confidence capped -- but LayerNorm's own
    # backward gradient still shrinks as the pre-norm weight magnitude grows
    # unbounded (verified locally: forcing gate_mlp's last-layer weights to
    # 1000x their init scale dropped the gradient reaching those weights to
    # ~2e-6). Strong decay keeps gate_mlp's weights from drifting into that
    # regime in the first place, so the diversity/entropy regularizers keep
    # a live gradient path for the full run, not just the first few steps.
    optimizer = torch.optim.AdamW([
        {"params": trainable, "lr": args.lr},
        {"params": list(model.amg.parameters()), "lr": args.lr, "weight_decay": 0.1},
    ])

    os.makedirs(args.output_dir, exist_ok=True)
    step = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_epoch = None
    for epoch in range(args.epochs):
        for item in train_loader:
            fwd = model.forward_embed_mode if args.adapter_mode == "embed" else model.forward_text_mode
            lm_loss, ptr_loss, entail_loss, gate_reg = fwd(item)
            loss = lm_loss + ptr_loss + 0.05 * entail_loss + gate_reg
            (loss / args.grad_accum_steps).backward()
            step += 1
            if step % args.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                print(f"epoch {epoch} step {step} lm_loss {lm_loss.item():.4f} ptr_loss {ptr_loss.item():.4f} "
                      f"entail_loss {entail_loss.item():.4f} gate_reg {gate_reg.item():.4f} "
                      f"gate_tel_ema {model.gate_tracker.ema_mean if model.gate_tracker.ema_mean is not None else float('nan'):.4f}")

            if args.smoke_test and step >= 3:
                print(f"peak_gpu_memory_gb={torch.cuda.max_memory_allocated()/1e9:.2f}")
                print("SMOKE TEST PASSED: 3 steps completed without error.")
                return

        ckpt_path = f"{args.output_dir}/epoch_{epoch}"
        model.model.save_pretrained(ckpt_path)
        torch.save({
            "telemetry_adapter": model.telemetry_adapter.state_dict(),
            "context_adapter": model.context_adapter.state_dict(),
            "amg": model.amg.state_dict(),
            "clause_encoder": model.clause_encoder.state_dict(),
            "cgpa": model.cgpa.state_dict(),
            "reasoning_aligner": model.reasoning_aligner.state_dict(),
        }, f"{ckpt_path}/v2_modules.pt")
        print(f"Saved checkpoint: {ckpt_path}")

        val_loss = evaluate_val_loss(model, val_ds, args.adapter_mode, args.val_check_limit)
        print(f"epoch {epoch} quick val_loss (first {min(args.val_check_limit, len(val_ds))} val events, "
              f"teacher-forced LM loss only) = {val_loss:.4f}")
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            # Written every time it improves (not just at the end) so it's
            # already correct even if the process crashes or the node dies
            # partway through a later epoch.
            with open(os.path.join(args.output_dir, "best_epoch.txt"), "w") as f:
                f.write(f"epoch_{best_epoch}\n")
        else:
            epochs_without_improvement += 1
            print(f"  no improvement over best ({best_val_loss:.4f} @ epoch {best_epoch}) "
                  f"for {epochs_without_improvement} epoch(s)")
        if epochs_without_improvement >= args.early_stop_patience:
            print(f"EARLY STOPPING at epoch {epoch} -- no val_loss improvement for "
                  f"{args.early_stop_patience} consecutive epochs. Best checkpoint: epoch_{best_epoch}")
            break

    print(f"Training loop finished. Best quick-val checkpoint: epoch_{best_epoch} (val_loss={best_val_loss:.4f}). "
          f"Run the full eval pipeline against that checkpoint, not necessarily the last one saved.")


if __name__ == "__main__":
    main()
