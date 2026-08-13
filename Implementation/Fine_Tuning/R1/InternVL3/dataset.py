"""
Dataset for InternVL3-8B-hf QLoRA fine-tuning - structure ported from the
working Qwen2-VL reference script (FineTune.py): frame resize cap, disk
caching of processed tensors, adjusted_start/end clip bounds with minimum
duration padding, and ALL rows (confirm + reject) since reject-row notes
carry real signal ("riding straight, no change" -> not_evasive).

Uses AutoProcessor + the native transformers InternVL API (NOT trust_remote_code).
"""

import os

import torch
from decord import VideoReader, cpu
from PIL import Image
from torch.utils.data import Dataset

from labels import (
    build_schema_instructions,
    build_target_json,
    build_telemetry_summary,
    get_clip_bounds,
)

RESIZE_MAX_SIDE = 448   # same cap as the Qwen2-VL reference - native GoPro res
                        # (1920x1080) blows up vision-token count and OOMs


class EvasiveEventDataset(Dataset):
    def __init__(self, df, video_path, processor, taxonomy, cache_dir, split_name,
                 num_frames=4, min_clip_duration=1.0):
        self.df = df.reset_index(drop=True)
        self.video_path = video_path
        self.processor = processor
        self.schema_instructions = build_schema_instructions(taxonomy)
        self.num_frames = num_frames
        self.min_clip_duration = min_clip_duration
        self.cache_dir = cache_dir
        self.split_name = split_name
        os.makedirs(cache_dir, exist_ok=True)
        self._mem_cache = {}

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        self.fps = float(vr.get_avg_fps())
        self.max_frame = len(vr) - 1
        del vr

    def __len__(self):
        return len(self.df)

    def _sample_frames(self, start, end):
        vr = VideoReader(self.video_path, ctx=cpu(0), num_threads=1)
        start_idx = max(0, int(start * self.fps))
        end_idx = min(self.max_frame, max(int(end * self.fps), start_idx + 1))
        idxs = torch.linspace(start_idx, end_idx, steps=self.num_frames).long().tolist()

        frames = []
        for i in idxs:
            img = Image.fromarray(vr[i].asnumpy()).convert("RGB")
            img.thumbnail((RESIZE_MAX_SIDE, RESIZE_MAX_SIDE), Image.LANCZOS)
            frames.append(img)
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        return frames

    def _build_messages(self, row):
        content = [{"type": "image"} for _ in range(self.num_frames)]
        content.append({"type": "text", "text": f"{self.schema_instructions}\n{build_telemetry_summary(row)}"})
        return [{"role": "user", "content": content}]

    def _build_prompt(self, row):
        """Kept for eval.py compatibility - returns the templated prompt text."""
        messages = self._build_messages(row)
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _compute(self, row):
        start, end = get_clip_bounds(row, self.min_clip_duration)
        frames = self._sample_frames(start, end)

        messages = self._build_messages(row)
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        target_json = build_target_json(row)

        # apply_chat_template inserts the model's actual image placeholder
        # tokens (not a hand-written "<image>" string) - hand-writing it was
        # the bug: the processor validates placeholder count against its own
        # configured token, which didn't match our literal text.
        # crop_to_patches=False: InternVL's processor defaults to dynamic
        # tiling (crop_to_patches=True, up to 12 patches/image), which turns
        # our 4 frames into 12 vision-token blocks - too expensive on an
        # 11GB GPU. One tile per frame instead, same conservative choice
        # the Qwen2-VL reference made for the same hardware reason.
        prompt_enc = self.processor(
            images=[frames], text=[prompt], return_tensors="pt", crop_to_patches=False
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        full_text = prompt + target_json + self.processor.tokenizer.eos_token
        full_enc = self.processor(
            images=[frames], text=[full_text], return_tensors="pt", crop_to_patches=False
        )

        input_ids = full_enc["input_ids"][0]
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "pixel_values": full_enc["pixel_values"],  # already [num_frames, 3, H, W] - no batch dim to strip
            "input_ids": input_ids,
            "attention_mask": full_enc["attention_mask"][0],
            "labels": labels,
        }

    def __getitem__(self, i):
        if i in self._mem_cache:
            return self._mem_cache[i]

        cache_path = os.path.join(self.cache_dir, f"{self.split_name}_{i}.pt")
        if os.path.exists(cache_path):
            item = torch.load(cache_path)
            self._mem_cache[i] = item
            return item

        row = self.df.iloc[i]
        item = self._compute(row)
        torch.save(item, cache_path)
        self._mem_cache[i] = item
        return item