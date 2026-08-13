
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
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from PIL import Image
import sys, pandas as pd, os
sys.path.insert(0, '.')
from motor_zero_shot_infer import get_clip_bounds, get_pretrimmed_bounds, sample_frames
from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

model_path = os.path.join(SCRATCH_ROOT, "models/paligemma2-3b-pt-224")
processor = AutoProcessor.from_pretrained(model_path)
model = PaliGemmaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map={'': 0})
model.eval()

df = pd.read_csv('gold_candidates.csv')


def frames_for(i):
    row = df.iloc[i]
    clip_id = row['clip_id']
    orig_start, orig_end = get_clip_bounds(row)
    video_path = os.path.join('upload_clips', f'{clip_id}__{orig_start:.3f}_{orig_end:.3f}.mp4')
    start, end = get_pretrimmed_bounds(row)
    return sample_frames(video_path, start, end), row


def montage(frames):
    w, h = frames[0].size
    m = Image.new('RGB', (w * 2, h * 2))
    positions = [(0, 0), (w, 0), (0, h), (w, h)]
    for f, pos in zip(frames, positions):
        m.paste(f.resize((w, h)), pos)
    return m


def run(img, prompt, max_new=40):
    inputs = processor(text=prompt, images=img, return_tensors='pt').to(model.device, model.dtype)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    return processor.batch_decode(gen[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]


f0, r0 = frames_for(0)
m0 = montage(f0)
print('SANITY color:', repr(run(m0, 'answer en What color is the sky?')))
print('SANITY road:', repr(run(m0, 'answer en Is there a road visible? yes or no')))

for i in [0, 1, 5, 6, 10, 13, 15, 30, 32, 33]:
    fr, row = frames_for(i)
    m = montage(fr)
    out = run(m, 'answer en Is this two-wheeler rider performing a sudden evasive maneuver (swerve, hard brake, near-miss avoidance)? Answer yes or no and briefly why.')
    print(f'event {i:2d} gold={row["decision"]:9s}:', repr(out[:120]))
