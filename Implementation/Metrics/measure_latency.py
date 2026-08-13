"""
measure_latency.py

Standalone latency benchmark for the trained checkpoint -- run separately
from run_inference.py so it gets precise per-call timing without being
confounded by the main inference pass's I/O/logging overhead.

Reports exactly what the architecture diagram's METRICS panel calls for:
  - ms/sample (end-to-end: video decode + forward + generation)
  - tokens/sec (generation throughput)
  - peak GPU memory during inference
  - param counts (total / trainable) for the record

Usage:
    python measure_latency.py --checkpoint ~/tagnet/checkpoints_scoped/epoch_7 --n 10
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Dataset"))
from train_tagnet_vlm import TagNetVLMDataset, collate_and_prompt  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(os.path.dirname(BASE), "Dataset")


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


def generate_timed(model, processor, clip_path: str, prompt: str) -> dict:
    from qwen_vl_utils import process_vision_info
    messages = [{
        "role": "user",
        "content": [{"type": "video", "video": clip_path, "nframes": 2, "max_pixels": 100352},
                     {"type": "text", "text": prompt}],
    }]
    t0 = time.time()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    inputs = processor(text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt").to(model.device)
    t_preprocess = time.time()

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    t_generate = time.time()

    n_new_tokens = generated.shape[1] - inputs["input_ids"].shape[1]
    return {
        "preprocess_ms": (t_preprocess - t0) * 1000,
        "generate_ms": (t_generate - t_preprocess) * 1000,
        "total_ms": (t_generate - t0) * 1000,
        "n_new_tokens": n_new_tokens,
        "tokens_per_sec": n_new_tokens / (t_generate - t_preprocess) if t_generate > t_preprocess else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.expanduser("~/tagnet/checkpoints_scoped/epoch_7"))
    parser.add_argument("--data", default=os.path.join(DATASET_DIR, "labeled_dataset_local_scoped.json"))
    parser.add_argument("--irc35_index", default=os.path.join(DATASET_DIR, "irc35_index_scoped.json"))
    parser.add_argument("--irc67_index", default=os.path.join(DATASET_DIR, "irc67_index_scoped.json"))
    parser.add_argument("--clips_dir", default=os.path.join(DATASET_DIR, "clips"))
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    with open(args.data, encoding="utf-8") as f:
        events = json.load(f)[: args.n]
    with open(args.irc35_index, encoding="utf-8") as f:
        irc35_index = json.load(f)
    with open(args.irc67_index, encoding="utf-8") as f:
        irc67_index = json.load(f)

    print(f"Loading checkpoint {args.checkpoint} ...")
    t_load_start = time.time()
    model, processor = load_model(args.checkpoint)
    load_time_s = time.time() - t_load_start

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model load time: {load_time_s:.1f}s")
    print(f"Total params: {total_params:,} | Trainable (LoRA): {trainable_params:,}")

    ds = TagNetVLMDataset(events, irc35_index, irc67_index, clips_dir=args.clips_dir)
    results = []
    for i in range(len(events)):
        item, prompt = collate_and_prompt([ds[i]], "text")
        r = generate_timed(model, processor, item["clip_path"], prompt)
        results.append(r)
        print(f"  sample {i}: total={r['total_ms']:.0f}ms "
              f"(preprocess={r['preprocess_ms']:.0f}ms, generate={r['generate_ms']:.0f}ms), "
              f"{r['n_new_tokens']} tokens, {r['tokens_per_sec']:.1f} tok/s")

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    def mean(key):
        return sum(r[key] for r in results) / len(results)

    summary = {
        "n_samples": len(results),
        "mean_total_ms": round(mean("total_ms"), 1),
        "mean_preprocess_ms": round(mean("preprocess_ms"), 1),
        "mean_generate_ms": round(mean("generate_ms"), 1),
        "mean_tokens_per_sec": round(mean("tokens_per_sec"), 2),
        "peak_gpu_memory_gb": round(peak_mem_gb, 2),
        "model_load_time_s": round(load_time_s, 1),
        "total_params": total_params,
        "trainable_params": trainable_params,
    }

    print("\n=== LATENCY SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out_path = os.path.join(BASE, "outputs", "latency_summary.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"per_sample": results, "summary": summary}, f, indent=2)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
