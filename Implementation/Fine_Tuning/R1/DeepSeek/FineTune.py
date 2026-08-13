
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
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE
os.environ['HF_HOME'] = HF_CACHE
import sys, json, hashlib, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoModelForCausalLM
warnings.filterwarnings('ignore')

DEEPSEEK_REPO = os.path.join(SCRATCH_ROOT, "DeepSeek-VL2")
if os.path.isdir(DEEPSEEK_REPO): sys.path.insert(0, DEEPSEEK_REPO)
from deepseek_vl2.models import DeepseekVLV2Processor

BASE_DIR         = SCRATCH_ROOT
MODEL_PATH       = BASE_DIR + '/models/deepseek-vl2-tiny'
CSV_PATH         = os.path.join(HOME_ROOT, "IRASTE/DAY_2/candidate_events_updated.csv")
VIDEO_DIR        = os.path.join(HOME_ROOT, "IRASTE")
VIDEO_FILENAME   = 'GX019940.MP4'
FRAMES_DIR       = BASE_DIR + '/frames'
OUTPUT_DIR       = BASE_DIR + '/deepseek_vl2_out'
BEST_CKPT_DIR    = OUTPUT_DIR + '/best_adapter'
TENSOR_CACHE_BASE= OUTPUT_DIR + '/tensor_cache'
PREPROCESS_VERSION = 20  # bumped: fixed per-event frame extraction (was clip-constant before)
MIN_CLIP_DURATION = 1.0
NUM_FRAMES = 4
IMAGE_SIZE = 384
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 2e-4
MAX_EPOCHS = 10
PATIENCE = 3
LORA_R = 8
VAL_SPLIT = 0.15
SEED = 42
LABEL_TAXONOMY = ['not_evasive', 'swerve', 'acceleration', 'lane_change', 'other']
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(BEST_CKPT_DIR, exist_ok=True)
torch.manual_seed(SEED)

def derive_label(notes):
    n = str(notes).lower()
    if any(k in n for k in ['not evasive','not_evasive','false','no evasive']): return 'not_evasive'
    if any(k in n for k in ['swerve','sharp turn','sharp_turn','weave']): return 'swerve'
    if any(k in n for k in ['accelerat','speed up','throttle']): return 'acceleration'
    if any(k in n for k in ['lane change','lane_change','overtake','merge']): return 'lane_change'
    return 'other'

def build_telemetry(row):
    parts = []
    for col, lbl in [('peak_abs_z_jerk','peak_z_jerk'),('peak_abs_z_az','peak_z_az'),('duration','duration_s')]:
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
    # Per-event frame extraction by real timestamp (mirrors qwen2.py's sample_frames) -
    # replaces the old buggy load_frames(clip_id) which returned the same static
    # frames for every event regardless of its actual time window.
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

def build_conversation(telem, frames, target_json):
    placeholders = ''.join(['<image>' + chr(10) for _ in frames])
    user_text = placeholders
    user_text += 'Analyse these ' + str(len(frames)) + ' dashcam frames.' + chr(10)
    user_text += 'Telemetry: ' + telem + chr(10)
    user_text += 'Classify into one of: ' + ', '.join(LABEL_TAXONOMY) + chr(10)
    user_text += 'Respond ONLY with valid JSON: {"label":"<class>","confidence":<0-1>,"reasoning":"<explanation>"}'
    return [
        {'role': '<|User|>', 'content': user_text, 'images': frames},
        {'role': '<|Assistant|>', 'content': target_json},
    ]

# Manual LoRA without peft/bitsandbytes
class LoRALinear(nn.Module):
    def __init__(self, linear, r=8, alpha=16):
        super().__init__()
        self.linear = linear
        d_in  = linear.weight.shape[1]
        d_out = linear.weight.shape[0]
        # LoRA adapter weights are kept in fp32 even though the frozen base is fp16.
        # Reason: with fp16 adapter weights, AdamW's eps=1e-8 underflows to exactly 0.0
        # in fp16 arithmetic, so sqrt(exp_avg_sq)+eps can be 0 -> 0/0 = NaN on the very
        # first optimizer step for any LoRA param with a zero (or tiny) gradient - which
        # is the normal case for lora_A early on since lora_B is zero-initialized.
        self.lora_A = nn.Parameter(torch.randn(r, d_in,  device=linear.weight.device, dtype=torch.float32) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, r, device=linear.weight.device, dtype=torch.float32))
        self.scale  = alpha / r
        for p in self.linear.parameters():
            p.requires_grad = False
    def forward(self, x):
        lora_out = (x.float() @ self.lora_A.T @ self.lora_B.T) * self.scale
        return self.linear(x) + lora_out.to(x.dtype)

def inject_lora(model, r=LORA_R, alpha=16):
    targets = ['q_proj','k_proj','v_proj','o_proj']
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(t in name for t in targets):
            parts = name.split('.')
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(module, r=r, alpha=alpha))
            count += 1
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('LoRA layers injected:', count)
    print('Trainable:', trainable, '/ Total:', total, '({:.2f}%)'.format(100*trainable/total))
    return model

def save_lora(model, path):
    os.makedirs(path, exist_ok=True)
    state = {n: p for n, p in model.named_parameters() if p.requires_grad}
    torch.save(state, os.path.join(path, 'lora_weights.pt'))
    print('LoRA saved to', path)

class EvasiveDataset(Dataset):
    def __init__(self, records, processor, cache_dir, split='train'):
        self.records   = records
        self.processor = processor
        self.cache_dir = cache_dir
        self.split     = split
        os.makedirs(cache_dir, exist_ok=True)
    def __len__(self): return len(self.records)
    def _cp(self, idx):
        r = self.records[idx]
        key = 'v' + str(PREPROCESS_VERSION) + '_' + self.split + '_' + str(idx) + '_' + r['clip_id'] + '_' + '{:.3f}_{:.3f}'.format(r['start'], r['end'])
        return os.path.join(self.cache_dir, hashlib.md5(key.encode()).hexdigest()[:12] + '.pt')
    def __getitem__(self, idx):
        cp = self._cp(idx)
        if os.path.exists(cp):
            return torch.load(cp, weights_only=False)
        r = self.records[idx]
        target_json = json.dumps({'label': r['label'], 'confidence': round(r['peak_confidence'], 3), 'reasoning': r['notes'] if r['notes'] else 'Event classified as ' + r['label'] + '.'})
        frames = load_frames(r['video_path'], r['start'], r['end'])
        conv = build_conversation(r['telemetry'], frames, target_json)
        enc  = self.processor(conversations=conv, images=frames, force_batchify=True, system_prompt='')
        input_ids      = enc.input_ids.squeeze(0)
        attention_mask = enc.attention_mask.squeeze(0)
        target_ids = self.processor.tokenizer(target_json, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze(0)
        labels = torch.full_like(input_ids, -100)
        tlen   = len(target_ids)
        matched = False
        for sp in range(len(input_ids) - tlen, -1, -1):
            if torch.equal(input_ids[sp:sp+tlen], target_ids[:tlen]):
                labels[sp:sp+tlen] = input_ids[sp:sp+tlen]
                matched = True
                break
        if not matched: labels[-tlen:] = input_ids[-tlen:]
        item = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'label_str': r['label']}
        for k in ['images', 'images_seq_mask', 'images_spatial_crop']:
            v = getattr(enc, k, None)
            if v is not None:
                item[k] = v.squeeze(0) if (hasattr(v,'dim') and v.dim() > 0) else v
        torch.save(item, cp)
        return item

def collate_fn(batch):
    out = {
        'input_ids':      torch.stack([b['input_ids']      for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'labels':         torch.stack([b['labels']         for b in batch]),
    }
    for k in ['images','images_seq_mask','images_spatial_crop']:
        if k in batch[0] and batch[0][k] is not None:
            try:    out[k] = torch.stack([b[k] for b in batch])
            except: out[k] = [b[k] for b in batch]
    return out

def main():
    print('Loading CSV')
    df = pd.read_csv(CSV_PATH)
    records = []
    for i, row in df.iterrows():
        notes = str(row.get('notes','')).strip()
        start, end = get_clip_bounds(row)
        records.append({
            'clip_id':         str(row.get('clip_id', 'clip_'+str(i))),
            'label':           derive_label(notes),
            'notes':           notes,
            'telemetry':       build_telemetry(row),
            'peak_confidence': safe_float(row.get('peak_confidence', 0.9), 0.9),
            'video_path':      os.path.join(VIDEO_DIR, VIDEO_FILENAME),
            'start':           start,
            'end':             end,
        })
    label_counts = {l: sum(1 for r in records if r['label']==l) for l in LABEL_TAXONOMY}
    print('Label distribution:', label_counts)
    weights = [1.0/max(label_counts[r['label']],1) for r in records]
    n_val   = max(1, int(len(records)*VAL_SPLIT))
    rng     = np.random.default_rng(SEED)
    idx     = rng.permutation(len(records))
    val_idx, train_idx = idx[:n_val].tolist(), idx[n_val:].tolist()
    train_r = [records[i] for i in train_idx]
    val_r   = [records[i] for i in val_idx]
    tw      = [weights[i] for i in train_idx]
    print('Train:', len(train_r), 'Val:', len(val_r))

    print('Loading processor')
    processor = DeepseekVLV2Processor.from_pretrained(MODEL_PATH)

    print('Loading model')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    ).cuda()
    model.config.use_cache = False
    print('Submodules:', [n for n,_ in model.named_children()])

    print('Freezing base model before LoRA injection')
    for p in model.parameters():
        p.requires_grad = False

    print('Injecting LoRA')
    model.language = inject_lora(model.language, r=LORA_R, alpha=16)

    print('Enabling gradient checkpointing on language model (activation-memory fix)')
    model.language.gradient_checkpointing_enable()
    model.language.config.use_cache = False
    if hasattr(model.language, 'enable_input_require_grads'):
        model.language.enable_input_require_grads()

    cache_hash = hashlib.md5((str(PREPROCESS_VERSION)+CSV_PATH+str(NUM_FRAMES)).encode()).hexdigest()[:8]
    cache_dir  = TENSOR_CACHE_BASE + '_' + cache_hash
    print('Cache:', cache_dir)

    train_ds     = EvasiveDataset(train_r, processor, cache_dir, 'train')
    val_ds       = EvasiveDataset(val_r,   processor, cache_dir, 'val')
    sampler      = WeightedRandomSampler(tw, len(tw), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,   collate_fn=collate_fn, num_workers=0)

    optimizer  = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    best_val   = float('inf')
    patience_c = 0
    step       = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        model.vision.eval()
        for batch in train_loader:
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if 'images' in batch and isinstance(batch['images'], torch.Tensor):
                batch['images'] = batch['images'].to(torch.float16)
            with torch.no_grad():
                inputs_embeds = model.prepare_inputs_embeds(
                    input_ids=batch['input_ids'],
                    images=batch.get('images'),
                    images_seq_mask=batch.get('images_seq_mask'),
                    images_spatial_crop=batch.get('images_spatial_crop'),
                )
            out  = model.language(inputs_embeds=inputs_embeds, attention_mask=batch['attention_mask'], labels=batch['labels'])
            loss = out.loss / GRAD_ACCUM
            loss.backward()
            step += 1
            if step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            if step % 20 == 0:
                print('epoch', epoch, 'step', step, 'loss', round(out.loss.item(), 4))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if 'images' in batch and isinstance(batch['images'], torch.Tensor):
                batch['images'] = batch['images'].to(torch.float16)
                inputs_embeds = model.prepare_inputs_embeds(
                    input_ids=batch['input_ids'],
                    images=batch.get('images'),
                    images_seq_mask=batch.get('images_seq_mask'),
                    images_spatial_crop=batch.get('images_spatial_crop'),
                )
                out = model.language(inputs_embeds=inputs_embeds, attention_mask=batch['attention_mask'], labels=batch['labels'])
                val_losses.append(out.loss.item())
        val_loss = float(np.mean(val_losses))
        print('epoch', epoch, 'val_loss', round(val_loss, 4))
        if val_loss < best_val:
            best_val   = val_loss
            patience_c = 0
            save_lora(model, BEST_CKPT_DIR)
            processor.save_pretrained(BEST_CKPT_DIR)
            print('New best', round(val_loss,4), 'saved')
        else:
            patience_c += 1
            print('No improvement', patience_c, '/', PATIENCE)
            if patience_c >= PATIENCE:
                print('Early stopping.')
                break
    print('Training complete.')

if __name__ == '__main__':
    main()
