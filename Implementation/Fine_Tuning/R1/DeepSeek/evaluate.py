"""
evaluate.py - Evaluate the fine-tuned DeepSeek-VL2-tiny hand-rolled LoRA adapter
(Mobile iRASTE evasive-event classification) on the held-out validation split.

Rebuilds the EXACT same train/val split as finetune.py (same SEED, VAL_SPLIT,
same rng.permutation logic) so the val set here is exactly the val set that was
held out during training.

Metrics reported (same schema as the Qwen2 / LLaVA-NeXT-Video eval reports):
    n_events_evaluated, n_generation_errors,
    accuracy, precision_macro, recall_macro, f1_macro,
    precision_weighted, recall_weighted, f1_weighted,
    bleu4_mean, meteor_mean, rougeL_mean, cider_mean,
    json_parse_failure_rate, hallucination_rate
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

import os
os.environ['XFORMERS_DISABLED'] = '1'
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE
os.environ['HF_HOME'] = HF_CACHE
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys, json, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
from PIL import Image
from transformers import AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

for pkg in ["wordnet", "omw-1.4", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"corpora/{pkg}" if pkg not in ("punkt", "punkt_tab") else f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

DEEPSEEK_REPO = os.path.join(SCRATCH_ROOT, "DeepSeek-VL2")
if os.path.isdir(DEEPSEEK_REPO):
    sys.path.insert(0, DEEPSEEK_REPO)
from deepseek_vl2.models import DeepseekVLV2Processor

# ============================================================
# CONFIG - must match finetune.py exactly so the val split, label
# derivation, and telemetry/conversation format are identical
# ============================================================
BASE_DIR      = SCRATCH_ROOT
MODEL_PATH    = BASE_DIR + '/models/deepseek-vl2-tiny'
CSV_PATH      = os.path.join(HOME_ROOT, "IRASTE/DAY_2/candidate_events_updated.csv")
VIDEO_DIR     = os.path.join(HOME_ROOT, "IRASTE")
VIDEO_FILENAME= 'GX019940.MP4'
FRAMES_DIR    = BASE_DIR + '/frames'
OUTPUT_DIR    = BASE_DIR + '/deepseek_vl2_out'
BEST_CKPT_DIR = OUTPUT_DIR + '/best_adapter'
EVAL_OUT_DIR  = OUTPUT_DIR + '/eval_results'
os.makedirs(EVAL_OUT_DIR, exist_ok=True)

NUM_FRAMES  = 4       # matches finetune.py after the OOM-driven scope reduction
IMAGE_SIZE  = 384
MIN_CLIP_DURATION = 1.0
LORA_R      = 8
VAL_SPLIT   = 0.15
SEED        = 42
MAX_NEW_TOKENS = 200
LABEL_TAXONOMY = ['not_evasive', 'swerve', 'acceleration', 'lane_change', 'other']

torch.manual_seed(SEED)


# ============================================================
# Same helpers as finetune.py (kept identical on purpose)
# ============================================================
def derive_label(notes):
    n = str(notes).lower()
    if any(k in n for k in ['not evasive', 'not_evasive', 'false', 'no evasive']):
        return 'not_evasive'
    if any(k in n for k in ['swerve', 'sharp turn', 'sharp_turn', 'weave']):
        return 'swerve'
    if any(k in n for k in ['accelerat', 'speed up', 'throttle']):
        return 'acceleration'
    if any(k in n for k in ['lane change', 'lane_change', 'overtake', 'merge']):
        return 'lane_change'
    return 'other'


def build_telemetry(row):
    parts = []
    for col, lbl in [('peak_abs_z_jerk', 'peak_z_jerk'), ('peak_abs_z_az', 'peak_z_az'), ('duration', 'duration_s')]:
        if col in row.index and pd.notna(row[col]):
            parts.append(lbl + '=' + '{:.3f}'.format(float(row[col])))
    return ', '.join(parts) if parts else 'no telemetry'


def safe_float(val, default=0.0):
    try:
        v = float(val)
        return default if (v != v) else v
    except Exception:
        return default


def get_clip_bounds(row):
    start = row.get('adjusted_start')
    end   = row.get('adjusted_end')
    if pd.isna(start): start = row.get('start_time')
    if pd.isna(end):   end   = row.get('end_time')
    start, end = safe_float(start, 0.0), safe_float(end, 0.0)
    if end <= start:
        end = start + 2.0
    if end - start < MIN_CLIP_DURATION:
        center = (start + end) / 2.0
        half = MIN_CLIP_DURATION / 2.0
        start, end = max(center - half, 0.0), center + half
    return start, end

def load_frames(video_path, start, end, n=NUM_FRAMES):
    # Per-event frame extraction by real timestamp - matches the fixed finetune.py
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        frames = []
    else:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        start_f = max(int(start * fps), 0)
        end_f   = max(int(end * fps), start_f + 1)
        idxs = np.linspace(start_f, end_f, n, dtype=int)
        frames = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                frames.append(img)
        cap.release()
    blank = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE))
    while len(frames) < n:
        frames.append(frames[-1] if frames else blank)
    return frames[:n]


def build_schema_prompt(telem, frames):
    placeholders = ''.join(['<image>' + chr(10) for _ in frames])
    user_text = placeholders
    user_text += 'Analyse these ' + str(len(frames)) + ' dashcam frames.' + chr(10)
    user_text += 'Telemetry: ' + telem + chr(10)
    user_text += 'Classify into one of: ' + ', '.join(LABEL_TAXONOMY) + chr(10)
    user_text += 'Respond ONLY with valid JSON: {"label":"<class>","confidence":<0-1>,"reasoning":"<explanation>"}'
    return [
        {'role': '<|User|>', 'content': user_text, 'images': frames},
        {'role': '<|Assistant|>', 'content': ''},
    ]


# Hand-rolled LoRA - must match finetune.py's LoRALinear exactly (fp32 adapter
# weights, per the NaN-loss fix) so the saved state_dict loads correctly.
class LoRALinear(nn.Module):
    def __init__(self, linear, r=8, alpha=16):
        super().__init__()
        self.linear = linear
        d_in = linear.weight.shape[1]
        d_out = linear.weight.shape[0]
        self.lora_A = nn.Parameter(torch.randn(r, d_in, device=linear.weight.device, dtype=torch.float32) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, r, device=linear.weight.device, dtype=torch.float32))
        self.scale = alpha / r
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        lora_out = (x.float() @ self.lora_A.T @ self.lora_B.T) * self.scale
        return self.linear(x) + lora_out.to(x.dtype)


def inject_lora(model, r=LORA_R, alpha=16):
    targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(t in name for t in targets):
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(module, r=r, alpha=alpha))
            count += 1
    print('LoRA layers injected:', count)
    return model


def extract_json(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Prefer the LAST {...} block, non-greedy per-match, since the prompt
    # itself contains a literal JSON template earlier in the string (in case
    # any prompt text leaks through despite trimming to generated tokens).
    matches = re.findall(r"\{.*?\}", text, re.DOTALL)
    for candidate in reversed(matches):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    # fall back to greedy whole-blob match (handles nested braces)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def main():
    print('Loading CSV')
    df = pd.read_csv(CSV_PATH)
    records = []
    for i, row in df.iterrows():
        notes = str(row.get('notes', '')).strip()
        start, end = get_clip_bounds(row)
        records.append({
            'clip_id': str(row.get('clip_id', 'clip_' + str(i))),
            'label': derive_label(notes),
            'notes': notes,
            'telemetry': build_telemetry(row),
            'peak_confidence': safe_float(row.get('peak_confidence', 0.9), 0.9),
            'video_path': os.path.join(VIDEO_DIR, VIDEO_FILENAME),
            'start': start,
            'end': end,
        })

    n_val = max(1, int(len(records) * VAL_SPLIT))
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(records))
    val_idx = idx[:n_val].tolist()
    val_r = [records[i] for i in val_idx]
    print('Val events (same split as training):', len(val_r))

    print('Loading processor from best_adapter dir (saved during training)')
    processor = DeepseekVLV2Processor.from_pretrained(BEST_CKPT_DIR)
    tokenizer = processor.tokenizer

    print('Loading base model')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).cuda()
    model.config.use_cache = True  # need KV cache for generation
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    print('Injecting LoRA architecture (to match saved state_dict)')
    model.language = inject_lora(model.language, r=LORA_R, alpha=16)

    lora_path = os.path.join(BEST_CKPT_DIR, 'lora_weights.pt')
    print('Loading LoRA weights from', lora_path)
    state = torch.load(lora_path, map_location='cuda', weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    loaded_keys = set(state.keys()) - set(missing)
    print(f'Loaded {len(loaded_keys)} / {len(state)} LoRA tensors into model ('
          f'{len(unexpected)} unexpected keys ignored)')
    if len(loaded_keys) != len(state):
        missing_lora = [k for k in state.keys() if k in missing]
        print('WARNING: some LoRA keys failed to load:', missing_lora[:10])

    model.eval()

    gt_labels, pred_labels = [], []
    gt_reasonings, pred_reasonings = [], []
    n_generation_errors = 0
    n_json_parse_failures = 0
    n_hallucinated_label = 0
    all_results = []

    for i, r in enumerate(val_r):
        try:
            frames = load_frames(r['video_path'], r['start'], r['end'])
            conv = build_schema_prompt(r['telemetry'], frames)
            prep = processor(conversations=conv, images=frames, force_batchify=True, system_prompt='')
            prep = prep.to(model.device, dtype=torch.float16)

            with torch.no_grad():
                inputs_embeds = model.prepare_inputs_embeds(**prep)
                outputs = model.generate(
                    inputs_embeds=inputs_embeds,
                    input_ids=prep.input_ids,
                    images=prep.images,
                    images_seq_mask=prep.images_seq_mask,
                    images_spatial_crop=prep.images_spatial_crop,
                    attention_mask=prep.attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                )
            # generate() with inputs_embeds+input_ids returns the FULL sequence
            # (prompt tokens + newly generated tokens). Trim to just the new
            # tokens before decoding - otherwise the decoded text re-includes
            # the schema prompt (which itself contains a literal JSON template
            # like {"label":"<class>",...}), breaking JSON extraction entirely.
            prompt_len = prep.input_ids.shape[1]
            gen_ids_trimmed = outputs[:, prompt_len:]
            output_text = tokenizer.decode(gen_ids_trimmed[0].cpu().tolist(), skip_special_tokens=True)
        except Exception as e:
            print(f'[event {i} clip={r["clip_id"]}] GENERATION ERROR: {repr(e)}')
            n_generation_errors += 1
            continue

        gt_label = r['label']
        gt_reasoning = r['notes'] if r['notes'] else ('Event classified as ' + r['label'] + '.')

        parsed = extract_json(output_text)
        if parsed is None or 'label' not in parsed:
            n_json_parse_failures += 1
            pred_label = 'other'
            pred_reasoning = ''
            all_results.append({
                'event_idx': i, 'clip_id': r['clip_id'], 'gt_label': gt_label,
                'gt_reasoning': gt_reasoning, 'raw_output': output_text, 'parsed': False,
            })
        else:
            pred_label = str(parsed.get('label', 'other'))
            pred_reasoning = str(parsed.get('reasoning', ''))
            if pred_label not in LABEL_TAXONOMY:
                n_hallucinated_label += 1
                pred_label = 'other'
            all_results.append({
                'event_idx': i, 'clip_id': r['clip_id'], 'gt_label': gt_label,
                'gt_reasoning': gt_reasoning, 'pred_label': parsed.get('label', ''),
                'pred_reasoning': pred_reasoning, 'raw_output': output_text, 'parsed': True,
            })

        gt_labels.append(gt_label)
        pred_labels.append(pred_label)
        gt_reasonings.append(gt_reasoning)
        pred_reasonings.append(pred_reasoning)
        print(f'[event {i}] gt={gt_label} pred={pred_label}')

    n_events_evaluated = len(gt_labels)

    pd.DataFrame(all_results).to_csv(os.path.join(EVAL_OUT_DIR, 'predictions.csv'), index=False)

    accuracy = accuracy_score(gt_labels, pred_labels) if n_events_evaluated else float('nan')
    if n_events_evaluated:
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            gt_labels, pred_labels, average='macro', zero_division=0)
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            gt_labels, pred_labels, average='weighted', zero_division=0)
    else:
        precision_macro = recall_macro = f1_macro = float('nan')
        precision_weighted = recall_weighted = f1_weighted = float('nan')

    smoothing = SmoothingFunction().method1
    bleu4_scores, meteor_scores, rougeL_scores = [], [], []
    r_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    cider_gts, cider_res = {}, {}

    for idx, (gt, pred) in enumerate(zip(gt_reasonings, pred_reasonings)):
        gt_tokens = nltk.word_tokenize(gt.lower())
        pred_tokens = nltk.word_tokenize(pred.lower()) if pred else ['']
        bleu4_scores.append(sentence_bleu(
            [gt_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing))
        try:
            meteor_scores.append(meteor_score([gt_tokens], pred_tokens))
        except Exception:
            meteor_scores.append(0.0)
        rougeL_scores.append(r_scorer.score(gt, pred)['rougeL'].fmeasure)
        cider_gts[str(idx)] = [gt]
        cider_res[str(idx)] = [pred if pred.strip() else '.']

    bleu4_mean = sum(bleu4_scores) / len(bleu4_scores) if bleu4_scores else float('nan')
    meteor_mean = sum(meteor_scores) / len(meteor_scores) if meteor_scores else float('nan')
    rougeL_mean = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else float('nan')
    if cider_gts:
        cider_mean, _ = Cider().compute_score(cider_gts, cider_res)
    else:
        cider_mean = float('nan')

    denom = n_events_evaluated if n_events_evaluated else 1
    json_parse_failure_rate = n_json_parse_failures / denom
    hallucination_rate = n_hallucinated_label / denom

    summary = {
        'n_events_evaluated': n_events_evaluated,
        'n_generation_errors': n_generation_errors,
        'accuracy': float(accuracy),
        'precision_macro': float(precision_macro),
        'recall_macro': float(recall_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(precision_weighted),
        'recall_weighted': float(recall_weighted),
        'f1_weighted': float(f1_weighted),
        'bleu4_mean': float(bleu4_mean),
        'meteor_mean': float(meteor_mean),
        'rougeL_mean': float(rougeL_mean),
        'cider': float(cider_mean),
        'json_parse_failure_rate': float(json_parse_failure_rate),
        'hallucination_rate': float(hallucination_rate),
    }

    print('\n===== EVALUATION SUMMARY =====')
    for k, v in summary.items():
        print(f'{k}: {v}')

    with open(os.path.join(EVAL_OUT_DIR, 'evaluation_report.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('\nSaved:', os.path.join(EVAL_OUT_DIR, 'evaluation_report.json'))
    print('Saved:', os.path.join(EVAL_OUT_DIR, 'predictions.csv'))


if __name__ == '__main__':
    main()
