"""
ZS + FT + V1 compliance pipeline for all 4 riders (R1-R4).
One script — pass --rider R1|R2|R3|R4 and --model ZS|FT|V1.

Outputs (all on gnode050):
  $SCRATCH_ROOT/{RiderName}/results/{model}_results.jsonl
  $SCRATCH_ROOT/{RiderName}/panels/{model}/panel_{eventid:03d}.png

Usage (run on gnode050 via nohup):
  nohup python rider_pipeline.py --rider R1 --model ZS > $SCRATCH_ROOT/Rider1_NJ/log_zs.txt 2>&1 &
"""

# ============================================================
# PATHS — edit or set environment variables before running
#   IRASTE_SCRATCH : root of your scratch/working directory
#   IRASTE_HOME    : your home directory on the compute node
#   HF_HOME        : HuggingFace cache directory
# ============================================================
import os as _os
SCRATCH_ROOT = _os.environ.get("IRASTE_SCRATCH", "/scratch/<your_username>")
HOME_ROOT    = _os.environ.get("IRASTE_HOME",    "/home/<your_username>")
HF_CACHE     = _os.environ.get("HF_HOME",        _os.path.join(SCRATCH_ROOT, "hf_cache"))

import os, sys, json, re, math, argparse, time

# Must be set before any HF imports so trust_remote_code modules go to scratch, not /share3
os.environ["HF_HOME"]           = HF_CACHE
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE
os.environ["HF_MODULES_CACHE"]  = os.path.join(SCRATCH_ROOT, "hf_modules")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(SCRATCH_ROOT, "hf_cache/st_cache")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--rider",  required=True, choices=["R1","R2","R3","R4","Pilot","Motor"])
parser.add_argument("--model",  required=True, choices=["ZS","FT","V1"])
args = parser.parse_args()

# ── RIDER CONFIG ─────────────────────────────────────────────────────────────
RIDER_INFO = {
    "R1":    {"name":"Rider1_NJ",      "location":"Outer Ring Road, Hyderabad, Telangana, India", "zone":"urban"},
    "R2":    {"name":"Rider2_AZ",      "location":"Outer Ring Road, Hyderabad, Telangana, India", "zone":"urban"},
    "R3":    {"name":"Rider3_VA",      "location":"Outer Ring Road, Hyderabad, Telangana, India", "zone":"urban"},
    "R4":    {"name":"Rider4_UC",      "location":"Outer Ring Road, Hyderabad, Telangana, India", "zone":"urban"},
    "Pilot": {"name":"Pilot_GX019940", "location":"Outer Ring Road, Hyderabad, Telangana, India", "zone":"urban"},
    "Motor": {"name":"Motor_GX0422",  "location":"Outer Ring Road, Hyderabad, Telangana, India", "zone":"urban"},
}
RIDER  = RIDER_INFO[args.rider]
RIDER_NAME   = RIDER["name"]
LOCATION_NAME= RIDER["location"]
ZONE         = RIDER["zone"]

SCRATCH   = fos.path.join(SCRATCH_ROOT, "{RIDER_NAME}")
if args.rider == "Pilot":
    DATASET = os.path.join(SCRATCH_ROOT, "Pilot_dataset.json")
elif args.rider == "Motor":
    DATASET = os.path.join(SCRATCH_ROOT, "Motor_dataset.json")
else:
    DATASET = fos.path.join(SCRATCH_ROOT, "{RIDER_NAME}_dataset.json")
OUT_DIR   = f"{SCRATCH}/{args.model}/results"
PANEL_DIR = f"{SCRATCH}/{args.model}/panels"
os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(PANEL_DIR, exist_ok=True)

# ── MODEL PATHS ───────────────────────────────────────────────────────────────
HF_CACHE    = HF_CACHE
ZS_MODEL_ID = f"{HF_CACHE}/hub/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/895c3a49bc3fa70a340399125c650a463535e71c"
FT_MODEL_ID = os.path.join(HOME_ROOT, "tagnet/InternVL3_adapter/best_adapter")  # used inside load_ft()
V1_CKPT     = os.path.join(HOME_ROOT, "tagnet/checkpoints_v1_embed_final/epoch_4")
BASE_3B     = f"{HF_CACHE}/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots"

CHROMA_PATH = os.path.join(SCRATCH_ROOT, "chroma_kg/chroma")
IRC35_SECS  = {"3","4","6","7","8","11"}
IRC67_SECS  = {"3","11","13","14","15","16","17","24","25","26"}

# ── KG RETRIEVAL ─────────────────────────────────────────────────────────────
import chromadb
from sentence_transformers import SentenceTransformer, util as st_util

_MINILM_PATH = os.path.join(SCRATCH_ROOT, "hf_cache/st_cache/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
EMB_MODEL = SentenceTransformer(_MINILM_PATH)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
# Single unified collection with doc="IRC35" or doc="IRC67"
irc_col = chroma_client.get_collection("irc_clauses")

def retrieve_kg(query, top_k=15):
    emb = EMB_MODEL.encode(query).tolist()
    r = irc_col.query(query_embeddings=[emb], n_results=min(top_k, irc_col.count()),
                      include=["metadatas","documents","distances"])
    clauses35, clauses67 = [], []
    for meta, doc, dist in zip(r["metadatas"][0], r["documents"][0], r["distances"][0]):
        sec = str(meta.get("section",""))
        cid = meta.get("clause","") or meta.get("clause_id","")
        entry = {"clause_id": cid, "text": doc, "distance": dist, "section": sec}
        if meta.get("doc","") == "IRC35" and sec in IRC35_SECS:
            clauses35.append(entry)
        elif meta.get("doc","") == "IRC67" and sec in IRC67_SECS:
            clauses67.append(entry)
    clauses35 = clauses35[:5]; clauses67 = clauses67[:5]
    def crrs(clauses):
        if not clauses: return 0.0
        sims = [1/(1+c["distance"]) for c in clauses]
        return round(float(np.mean(sims)),4)
    return clauses35, clauses67, crrs(clauses35), crrs(clauses67)

def build_scene_query(tel, action):
    ay   = tel.get("ay",0) if isinstance(tel,dict) else 0
    return f"{action} zone={ZONE} lateral_accel={ay:.2f} location={LOCATION_NAME}"

# ── IMAGE UTILS ──────────────────────────────────────────────────────────────
MAX_IMG_W_FT = 672
MAX_IMG_W_ZS = 448
MAX_IMG_W_V1 = 672

def load_img(path, max_w):
    """Load a PIL image from a path — handles both image files and mp4 clips."""
    if path.lower().endswith((".mp4",".avi",".mov",".mkv")):
        import cv2
        frame = None
        for seek_frac in [0.5, 0.0, 0.25, 0.75]:
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * seek_frac))
            ok, frame = cap.read(); cap.release()
            if ok:
                break
        if frame is None or not ok:
            raise RuntimeError(f"Cannot read frame from {path}")
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        img = Image.open(path).convert("RGB")
    w, h = img.size
    if w > max_w:
        img = img.resize((max_w, int(h * max_w / w)), Image.LANCZOS)
    return img

def is_garbage(text):
    if not text: return True
    if text.startswith("!"): return True
    alpha = sum(c.isalpha() for c in text)
    return alpha < 5

# ── ZS MODEL ─────────────────────────────────────────────────────────────────
def load_zs():
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        ZS_MODEL_ID, torch_dtype=torch.float16, device_map="cuda"
    )
    proc = AutoProcessor.from_pretrained(ZS_MODEL_ID)
    def infer(img, prompt, max_tok=120):
        messages=[{"role":"user","content":[
            {"type":"image","image":img},
            {"type":"text","text":prompt}
        ]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = proc(text=[text], images=image_inputs, videos=video_inputs,
                      padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=max_tok,
                                 temperature=0.1, do_sample=False)
        ids = [o[len(i):] for i,o in zip(inputs.input_ids, ids)]
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    return infer

# ── FT MODEL (InternVL3-8B + LoRA adapter) ───────────────────────────────────
def load_ft():
    import cv2
    from transformers import AutoProcessor
    from peft import PeftModel

    INTERNVL_BASE = os.path.join(SCRATCH_ROOT, "hf_cache/models--OpenGVLab--InternVL3-8B/snapshots/853e3a797a661694b1b8ece0cb72dc2b23e3dac9")
    FT_ADAPTER    = os.path.join(HOME_ROOT, "tagnet/InternVL3_adapter/best_adapter")

    # Patch old transformers: InternVLChatModel lacks all_tied_weights_keys
    import transformers.quantizers.base as _qbase
    import transformers.integrations.accelerate as _tacc
    _orig_gknc = _qbase.get_keys_to_not_convert
    def _patched_gknc(model):
        if not hasattr(model, "all_tied_weights_keys"):
            model.all_tied_weights_keys = {}
        return _orig_gknc(model)
    _qbase.get_keys_to_not_convert = _patched_gknc
    _orig_iiadm = _tacc._init_infer_auto_device_map
    def _patched_iiadm(model, *a, **kw):
        if not hasattr(model, "all_tied_weights_keys"):
            model.all_tied_weights_keys = {}
        return _orig_iiadm(model, *a, **kw)
    _tacc._init_infer_auto_device_map = _patched_iiadm

    from transformers import AutoModel, BitsAndBytesConfig
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    base = AutoModel.from_pretrained(
        INTERNVL_BASE, quantization_config=bnb_cfg,
        device_map="auto", trust_remote_code=True,
    ).eval()
    from transformers import AutoTokenizer
    tok_tmp = AutoTokenizer.from_pretrained(INTERNVL_BASE, trust_remote_code=True)
    img_ctx_id = tok_tmp.convert_tokens_to_ids("<IMG_CONTEXT>")
    base.img_context_token_id = img_ctx_id  # required by InternVL's generate()
    model = PeftModel.from_pretrained(base, FT_ADAPTER)
    model.eval()
    from torchvision import transforms as T
    tok = AutoTokenizer.from_pretrained(INTERNVL_BASE, trust_remote_code=True)
    IMG_MEAN = (0.485, 0.456, 0.406); IMG_STD = (0.229, 0.224, 0.225)
    img_transform = T.Compose([
        T.Resize((448, 448), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMG_MEAN, std=IMG_STD),
    ])
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    IMG_START_TOKEN   = "<img>"
    IMG_END_TOKEN     = "</img>"
    NUM_IMG_TOKENS    = 256  # standard for 448x448 single tile

    def _get_pixel_values(img):
        if isinstance(img, str):
            if img.lower().endswith((".mp4",".avi",".mov",".mkv")):
                cap = cv2.VideoCapture(img)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
                ok, frame = cap.read(); cap.release()
                if not ok:
                    cap = cv2.VideoCapture(img); ok, frame = cap.read(); cap.release()
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                img = Image.open(img).convert("RGB")
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img = img.convert("RGB")
        return img_transform(img).unsqueeze(0)  # (1, 3, 448, 448)

    def infer(img, prompt, max_tok=120):
        pixel_values = _get_pixel_values(img)
        img_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * NUM_IMG_TOKENS + IMG_END_TOKEN
        question = f"{img_tokens}\n{prompt}"
        messages = [{"role":"user","content":question}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tok(text, return_tensors="pt").to("cuda")
        device = next(model.parameters()).device
        pixel_values = pixel_values.to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            out = model.generate(
                **model_inputs, pixel_values=pixel_values,
                max_new_tokens=max_tok, do_sample=False,
                eos_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0][model_inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return infer

# ── V1 MODEL (TAG-Net: Qwen2.5-VL-3B + QLoRA + TelemetryAdapter) ─────────────
def load_v1():
    sys.path.insert(0, os.path.join(HOME_ROOT, "tagnet/Dataset"))
    sys.path.insert(0, os.path.join(HOME_ROOT, "tagnet"))
    from train_tagnet_vlm import TagNetVLMForTraining
    from tagnet_vlm import TAGNetVLMConfig
    from peft import PeftModel
    from qwen_vl_utils import process_vision_info

    cfg = TAGNetVLMConfig(backbone_name=os.path.join(SCRATCH_ROOT, "qwen25vl3b_dl"), use_qlora=True)
    wrapper = TagNetVLMForTraining(cfg, adapter_mode="embed")
    wrapper.model = PeftModel.from_pretrained(wrapper.model.get_base_model(), V1_CKPT)
    adapters_path = f"{V1_CKPT}/adapters.pt"
    if os.path.exists(adapters_path):
        saved = torch.load(adapters_path, map_location=wrapper.model.device)
        wrapper.telemetry_adapter.load_state_dict(saved["telemetry_adapter"])
        wrapper.context_adapter.load_state_dict(saved["context_adapter"])
    wrapper.model.eval()
    proc = wrapper.processor

    def infer(img, prompt, max_tok=120, tel=None, gps=None):
        if isinstance(img, str):
            import cv2
            if img.lower().endswith((".mp4",".avi",".mov",".mkv")):
                cap = cv2.VideoCapture(img)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
                ok, frame = cap.read(); cap.release()
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                img = Image.open(img).convert("RGB")
        messages = [{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":prompt}]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(messages)
        inputs = proc(text=[text], images=imgs, videos=vids, padding=True,
                      return_tensors="pt").to(wrapper.model.device)
        base = wrapper.model.get_base_model()
        merged = base.get_input_embeddings()(inputs["input_ids"])
        tel_d = tel or {}; gps_d = gps or {}
        tel_tok = wrapper.telemetry_adapter(wrapper.telemetry_to_vec(tel_d))
        ctx_tok = wrapper.context_adapter(wrapper.context_to_vec(gps_d))
        inputs_embeds = torch.cat([torch.cat([tel_tok, ctx_tok], dim=1), merged], dim=1)
        attn_mask = torch.cat([
            torch.ones(1, 2, dtype=inputs["attention_mask"].dtype, device=wrapper.model.device),
            inputs["attention_mask"],
        ], dim=1)
        with torch.no_grad():
            generated = wrapper.model.generate(
                inputs_embeds=inputs_embeds, attention_mask=attn_mask,
                max_new_tokens=max_tok, do_sample=False,
                pixel_values_videos=inputs.get("pixel_values_videos"),
                video_grid_thw=inputs.get("video_grid_thw"),
            )
        return proc.batch_decode(generated, skip_special_tokens=True)[0].strip()
    return infer

# ── PROMPTS ──────────────────────────────────────────────────────────────────
def p_env(action):
    return (f"You are a road safety analyst. Location: {LOCATION_NAME}.\n"
            f"Describe the road environment visible in this dashcam frame (2 sentences):\n"
            f"(1) Traffic density, road type, lane markings.\n"
            f"(2) Weather/lighting conditions. Plain text. Max 40 words.")

def p_evasive(action):
    return (f"You are a road safety analyst. Location: {LOCATION_NAME}.\n"
            f"Evasive action: {action.replace('_',' ')}.\n"
            f"Explain what hazard caused this action (2-3 sentences):\n"
            f"(1) What vehicle/obstacle triggered it.\n"
            f"(2) How the rider responded.\n"
            f"(3) Safety risk level (LOW/MED/HIGH). Plain text. Max 60 words.")

def p_road_kg(retrieved_35, env_text):
    clauses = "\n".join(
        f"- [{c['clause_id']}] {c['text'][:180]}" for c in retrieved_35
    ) or "(none)"
    return (f"Location: {LOCATION_NAME}. Scene: {env_text}\n\n"
            f"Assess road surface (2 sentences):\n"
            f"(1) Lane markings, surface damage, signage.\n"
            f"(2) State COMPLIANT or NON_COMPLIANT. Cite ONE clause from:\n\n"
            f"IRC-35 clauses:\n{clauses}\n\nPlain text. Max 40 words.")

def p_infra_kg(retrieved_67, env_text):
    clauses = "\n".join(
        f"- [{c['clause_id']}] {c['text'][:180]}" for c in retrieved_67
    ) or "(none)"
    return (f"Location: {LOCATION_NAME}. Scene: {env_text}\n\n"
            f"Assess road infrastructure (2 sentences):\n"
            f"(1) Safety barriers, signage, road geometry.\n"
            f"(2) State COMPLIANT or NON_COMPLIANT. Cite ONE clause from:\n\n"
            f"IRC-67 clauses:\n{clauses}\n\nPlain text. Max 40 words.")

def extract_verdict_clause(text, prefix):
    verdict = "COMPLIANT"
    if re.search(r"\bNON[_\s-]?COMPLIANT\b", text, re.I):
        verdict = "NON_COMPLIANT"
    m = re.search(r'\[([A-Z0-9]+\.\d[\d.]*)\]', text)
    clause = m.group(1) if m else ""
    return verdict, clause

# ── PANEL (same format as R3/R4) ─────────────────────────────────────────────
try:
    FONT_B = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",   15)
    FONT_R = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",        13)
    FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",        11)
except:
    FONT_B = FONT_R = FONT_S = ImageFont.load_default()

PANEL_W, PANEL_H = 1200, 600
LEFT_W,  RIGHT_W = 900, 300
FRAME_W, FRAME_H = LEFT_W, 480
FACTS_H          = PANEL_H - FRAME_H

BORDER_RED   = (220, 50, 50)
BORDER_GREEN = (0, 140, 0)
BORDER_BLACK = (20, 20, 20)
BG_RIGHT     = (20, 24, 32)
BG_FACTS     = (30, 30, 30)

def wrap(text, font, max_w, draw):
    words = text.split(); lines = []; cur = ""
    for w in words:
        test = (cur + " " + w).strip()
        bb = draw.textbbox((0,0), test, font=font)
        if bb[2] - bb[0] <= max_w: cur = test
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    return lines

def draw_section(draw, x, y, w, label, body, max_h):
    draw.text((x+6, y+3), label, font=FONT_B, fill=(120,220,120))
    y += 20
    for line in wrap(body, FONT_R, w-12, draw):
        if y + 15 > max_h: break
        draw.text((x+6, y), line, font=FONT_R, fill=(220,220,220))
        y += 15
    return y + 4

def make_panel(clip_path, event, env, evas, road, infra,
               v35, c35, v67, c67, model_name, event_id):
    panel = Image.new("RGB", (PANEL_W, PANEL_H), (10,10,10))

    # --- Left: dashcam frame ---
    try:
        import cv2
        cap = cv2.VideoCapture(clip_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)//2))
        ok, fr = cap.read(); cap.release()
        frame_img = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)) if ok else None
    except:
        frame_img = None

    if frame_img:
        fw, fh = frame_img.size
        scale = min(FRAME_W/fw, FRAME_H/fh)
        nw, nh = int(fw*scale), int(fh*scale)
        frame_img = frame_img.resize((nw, nh), Image.LANCZOS)
        ox = (FRAME_W - nw)//2; oy = (FRAME_H - nh)//2
        panel.paste(frame_img, (ox, oy))

    draw = ImageDraw.Draw(panel)
    # Red border around frame area
    draw.rectangle([0, 0, LEFT_W-1, FRAME_H-1], outline=BORDER_RED, width=3)

    # Facts strip (black background)
    panel.paste(Image.new("RGB", (LEFT_W, FACTS_H), BG_FACTS), (0, FRAME_H))
    draw.rectangle([0, FRAME_H, LEFT_W-1, PANEL_H-1], outline=BORDER_BLACK, width=3)
    facts_y = FRAME_H + 4

    action_label = event.get("evasive_action","").replace("_"," ")
    loc = LOCATION_NAME
    draw.text((6, facts_y),    f"Rider: {RIDER_NAME}  |  Location: {loc}  |  Action: {action_label}", font=FONT_S, fill=(200,200,200))
    draw.text((6, facts_y+14), f"IRC-35: {v35}  Clause: {c35 or '—'}   |   IRC-67: {v67}  Clause: {c67 or '—'}", font=FONT_S, fill=(180,200,180))
    draw.text((6, facts_y+28), f"Model: {model_name}  |  Event #{event_id:03d}", font=FONT_S, fill=(160,160,160))

    # --- Right: reasoning panel ---
    panel.paste(Image.new("RGB", (RIGHT_W, PANEL_H), BG_RIGHT), (LEFT_W, 0))
    draw.rectangle([LEFT_W, 0, PANEL_W-1, PANEL_H-1], outline=BORDER_GREEN, width=3)

    ry = 8; rx = LEFT_W + 6; rw = RIGHT_W - 12
    draw.text((rx, ry), f"COMPLIANCE ANALYSIS", font=FONT_B, fill=(100,200,100)); ry += 20
    draw.text((rx, ry), f"{model_name} | {loc}", font=FONT_S, fill=(140,180,140)); ry += 18

    for label, body in [("ENVIRONMENT", env), ("EVASIVE REASONING", evas),
                        ("ROAD SURFACE", road), ("INFRASTRUCTURE", infra)]:
        ry = draw_section(draw, rx, ry, rw, label, body or "—", PANEL_H-4)

    out_path = f"{PANEL_DIR}/panel_{event_id:03d}.png"
    panel.save(out_path)
    return out_path

# ── INFERENCE LOOP ────────────────────────────────────────────────────────────
print(f"Loading {args.model} model for {RIDER_NAME}...")

IS_V1 = (args.model == "V1")
if   args.model == "ZS": infer_fn = load_zs(); MAX_W = MAX_IMG_W_ZS; FALLBACK_W = 336
elif args.model == "FT": infer_fn = load_ft(); MAX_W = MAX_IMG_W_FT; FALLBACK_W = 448
else:                    infer_fn = load_v1(); MAX_W = MAX_IMG_W_V1; FALLBACK_W = 448

def infer(img, prompt, max_tok=120, tel=None, gps=None):
    if IS_V1:
        return infer_fn(img, prompt, max_tok, tel=tel, gps=gps)
    return infer_fn(img, prompt, max_tok)

with open(DATASET) as f:
    events = json.load(f)

print(f"Dataset: {len(events)} events | Model: {args.model} | Rider: {RIDER_NAME}")
results = []
OUT_JSONL = f"{OUT_DIR}/{args.model.lower()}_results.jsonl"  # e.g. zs_results.jsonl

# Resume: load already-done event IDs so we can skip them
done_ids = set()
if os.path.exists(OUT_JSONL):
    for line in open(OUT_JSONL):
        if line.strip():
            try: done_ids.add(json.loads(line)["event_id"])
            except: pass
if done_ids:
    print(f"Resuming: skipping {len(done_ids)} already-processed events")
fout = open(OUT_JSONL, "a" if done_ids else "w")

for ev in events:
    eid    = ev["event_id"]
    if eid in done_ids:
        print(f"[{eid:2d}/{len(events)}] SKIP (already done)")
        continue
    action = ev["evasive_action"]
    clip   = ev.get("clip_path") or ev.get("clip")
    tel    = ev.get("telemetry", {})
    gps    = ev.get("gps", {"speed_kmh":30,"zone_type":ZONE})
    gps["zone_type"] = ZONE

    print(f"[{eid:2d}/{len(events)}] {action} | clip={os.path.basename(clip)}")
    if not os.path.exists(clip):
        print(f"  [SKIP] clip not found: {clip}"); continue

    img     = load_img(clip, MAX_W)
    img_sm  = load_img(clip, FALLBACK_W)

    # -- Call 1: environment --
    env = infer(img, p_env(action), max_tok=80, tel=tel, gps=gps)
    if is_garbage(env): env = infer(img_sm, p_env(action), max_tok=80, tel=tel, gps=gps)

    # -- KG retrieval --
    query = build_scene_query(tel, action)
    cl35, cl67, crrs35, crrs67 = retrieve_kg(query)

    # -- Call 2: evasive reasoning --
    evas = infer(img, p_evasive(action), max_tok=120, tel=tel, gps=gps)
    if is_garbage(evas): evas = infer(img_sm, p_evasive(action), max_tok=120, tel=tel, gps=gps)

    # -- Call 3: road surface --
    road = infer(img, p_road_kg(cl35, env), max_tok=80, tel=tel, gps=gps)
    if is_garbage(road): road = infer(img_sm, p_road_kg(cl35, env), max_tok=80, tel=tel, gps=gps)

    # -- Call 4: infrastructure --
    infra = infer(img, p_infra_kg(cl67, env), max_tok=80, tel=tel, gps=gps)
    if is_garbage(infra): infra = infer(img_sm, p_infra_kg(cl67, env), max_tok=80, tel=tel, gps=gps)

    v35, c35 = extract_verdict_clause(road,  "IRC-35")
    v67, c67 = extract_verdict_clause(infra, "IRC-67")

    rec = {
        "event_id":        eid,
        "rider":           RIDER_NAME,
        "location":        ev.get("location", LOCATION_NAME),
        "evasive_action":  action,
        "env":             env,
        "evasive_reasoning": evas,
        "road_surface":    road,
        "infrastructure":  infra,
        "irc35_verdict":   v35,
        "irc35_clause":    c35,
        "irc67_verdict":   v67,
        "irc67_clause":    c67,
        "retrieved_irc35": ";".join(c["clause_id"] for c in cl35),
        "retrieved_irc67": ";".join(c["clause_id"] for c in cl67),
        "crrs35":          crrs35,
        "crrs67":          crrs67,
    }
    results.append(rec)
    fout.write(json.dumps(rec) + "\n")
    fout.flush()

    make_panel(clip, ev, env, evas, road, infra, v35, c35, v67, c67, args.model, eid)
    print(f"  env={env[:60]}...")
    print(f"  evas={evas[:60]}...")
    print(f"  road={road[:40]}  v35={v35}")
    print(f"  infra={infra[:40]}  v67={v67}")

fout.close()
print(f"\nDone. {len(results)} events → {OUT_JSONL}")
print(f"Panels → {PANEL_DIR}/")
