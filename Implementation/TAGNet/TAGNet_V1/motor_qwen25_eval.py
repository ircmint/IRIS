"""
Evaluate the QLoRA-tuned Qwen2.5-VL-3B-Instruct on all 34 motor events
(train-set eval — model was trained on all events, no held-out split).

Same metrics as Custom_Data rider evals and DeepSeek:
  accuracy, macro/weighted P/R/F1, BLEU-4, METEOR, ROUGE-L, CIDEr,
  json_parse_failure_rate, hallucination_rate.

Run:
    source $SCRATCH_ROOT/envs/qwen2vl/bin/activate
    cd $SCRATCH_ROOT
    CUDA_VISIBLE_DEVICES=1 HF_HOME=$SCRATCH_ROOT/hf_cache \
        python motor_qwen25_eval.py 2>&1 | tee motor_qwen25_eval.log
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

import json, os, re, time
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

MODEL_PATH   = os.path.join(SCRATCH_ROOT, "Qwen2.5-VL-3B-Instruct")
ADAPTER_PATH = os.path.join(SCRATCH_ROOT, "runs/motor_qwen25_qlora/best_adapter")
DATA_ROOT    = SCRATCH_ROOT
CLIPS_DIR    = os.path.join(DATA_ROOT, "upload_clips")
CSV_PATH     = os.path.join(DATA_ROOT, "gold_candidates.csv")
NUM_FRAMES   = 4
RESIZE_MAX   = 448
NUM_TOL      = 0.15


def get_label(row):
    return "evasive" if str(row.get("decision","")).strip().lower() == "confirm" else "not_evasive"

def get_reasoning(row):
    v = row.get("notes")
    return str(v).strip() if pd.notna(v) and str(v).strip() else "No additional notes provided."

def get_clip_path(row):
    cid = row["clip_id"]
    start = row.get("adjusted_start") if pd.notna(row.get("adjusted_start")) else row.get("start_time")
    end   = row.get("adjusted_end")   if pd.notna(row.get("adjusted_end"))   else row.get("end_time")
    start, end = float(start), float(end)
    if end - start < 1.0:
        c = (start+end)/2.0; start, end = max(c-0.5,0.0), c+0.5
    return os.path.join(CLIPS_DIR, f"{cid}__{start:.3f}_{end:.3f}.mp4")

def build_taxonomy(df):
    df = df[df["decision"].str.lower().isin(["confirm","reject"])].copy().reset_index(drop=True)
    df["_label"] = df.apply(get_label, axis=1)
    return df, sorted(df["_label"].unique().tolist())

def build_telemetry(row):
    return (f"Telemetry: peak jerk={row.get('peak_abs_z_jerk','NA')}, "
            f"peak lateral accel={row.get('peak_abs_z_az','NA')}, "
            f"duration={row.get('duration','NA')}s, IMU decision={row.get('decision','NA')}.")

def extract_numbers(text):
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", text)]

def is_hallucinated(pred, input_nums, valid_labels):
    if pred.get("label") not in valid_labels: return True
    c = pred.get("confidence")
    if not isinstance(c, (int,float)) or not (0<=c<=1): return True
    for n in extract_numbers(pred.get("reasoning","")):
        if not any(abs(n-m)<=NUM_TOL*max(abs(m),1e-6) for m in input_nums): return True
    return False


def load_model():
    processor = AutoProcessor.from_pretrained(ADAPTER_PATH)
    base = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map={"":0}
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()
    return model, processor


def generate(model, processor, row, taxonomy):
    labels = ", ".join(taxonomy)
    schema = (
        "You are analyzing a short dashcam clip from a two-wheeler for evasive-maneuver "
        "classification. Respond ONLY with a JSON object:\n"
        f'{{"label": "<one of: {labels}>", "confidence": <float 0-1>, '
        '"reasoning": "<short justification>"}}\nNo text outside the JSON.'
    )
    messages = [{"role": "user", "content": [
        {"type": "video", "video": get_clip_path(row), "nframes": NUM_FRAMES, "max_pixels": RESIZE_MAX*RESIZE_MAX},
        {"type": "text", "text": f"{schema}\n{build_telemetry(row)}"},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[prompt], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    completion = out[0, inputs["input_ids"].shape[1]:]
    return processor.decode(completion, skip_special_tokens=True)


def main():
    model, processor = load_model()
    with open(os.path.join(ADAPTER_PATH, "taxonomy.json")) as f:
        taxonomy = json.load(f)
    print("Taxonomy:", taxonomy)

    df = pd.read_csv(CSV_PATH)
    df, _ = build_taxonomy(df)
    print(f"Evaluating {len(df)} events (train-set eval, no held-out split).", flush=True)

    gold_labels, pred_labels = [], []
    gold_reasoning, pred_reasoning = [], []
    parse_failures = hallucinations = gen_errors = 0
    cider_gts, cider_res = {}, {}
    per_event_rows = []

    smoothing = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    for idx in range(len(df)):
        t0 = time.time()
        row = df.iloc[idx]
        gold_label = get_label(row)
        gold_reason = get_reasoning(row)
        input_nums = extract_numbers(build_telemetry(row))

        try:
            raw = generate(model, processor, row, taxonomy)
        except Exception as e:
            gen_errors += 1
            print(f"[{idx+1}/{len(df)}] GEN ERROR: {e}", flush=True)
            gold_labels.append(gold_label); gold_reasoning.append(gold_reason)
            pred_labels.append("GEN_ERROR"); pred_reasoning.append(""); hallucinations += 1
            per_event_rows.append({"idx":idx,"clip_id":row.get("clip_id",""),"gold_label":gold_label,
                "pred_label":"GEN_ERROR","gold_reasoning":gold_reason,"pred_reasoning":"",
                "parse_failed":False,"generation_error":True,"hallucinated":True})
            continue

        print(f"[{idx+1}/{len(df)}] {time.time()-t0:.1f}s gold={gold_label} out={raw[:80]!r}", flush=True)
        gold_labels.append(gold_label); gold_reasoning.append(gold_reason)

        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            pred = json.loads(m.group(0)) if m else json.loads(raw)
        except (json.JSONDecodeError, AttributeError):
            parse_failures += 1; hallucinations += 1
            pred_labels.append("PARSE_FAIL"); pred_reasoning.append("")
            per_event_rows.append({"idx":idx,"clip_id":row.get("clip_id",""),"gold_label":gold_label,
                "pred_label":"PARSE_FAIL","gold_reasoning":gold_reason,"pred_reasoning":raw,
                "parse_failed":True,"generation_error":False,"hallucinated":True})
            continue

        pl = pred.get("label","PARSE_FAIL"); pr = pred.get("reasoning","")
        pred_labels.append(pl); pred_reasoning.append(pr)
        halluc = is_hallucinated(pred, input_nums, taxonomy)
        if halluc: hallucinations += 1
        per_event_rows.append({"idx":idx,"clip_id":row.get("clip_id",""),"gold_label":gold_label,
            "pred_label":pl,"gold_reasoning":gold_reason,"pred_reasoning":pr,
            "parse_failed":False,"generation_error":False,"hallucinated":halluc})
        cider_gts[str(idx)] = [gold_reason]; cider_res[str(idx)] = [pr]

    n = len(df)
    acc = accuracy_score(gold_labels, pred_labels)
    pm,rm,fm,_ = precision_recall_fscore_support(gold_labels,pred_labels,labels=taxonomy,average="macro",zero_division=0)
    pw,rw,fw,_ = precision_recall_fscore_support(gold_labels,pred_labels,labels=taxonomy,average="weighted",zero_division=0)

    bleu_s, meteor_s, rouge_s = [], [], []
    for g,p in zip(gold_reasoning, pred_reasoning):
        if not p: bleu_s.append(0.0); meteor_s.append(0.0); rouge_s.append(0.0); continue
        bleu_s.append(sentence_bleu([g.split()],p.split(),smoothing_function=smoothing))
        meteor_s.append(meteor_score([g.split()],p.split()))
        rouge_s.append(rouge.score(g,p)["rougeL"].fmeasure)

    try:
        from pycocoevalcap.cider.cider import Cider
        cider_score,_ = Cider().compute_score(cider_gts, cider_res)
    except ImportError:
        cider_score = None

    results = {
        "NOTE": "TRAIN-SET EVAL — model trained on all events, no held-out split",
        "n_events_evaluated": n, "n_generation_errors": gen_errors, "taxonomy": taxonomy,
        "accuracy": round(acc,4),
        "precision_macro": round(pm,4), "recall_macro": round(rm,4), "f1_macro": round(fm,4),
        "precision_weighted": round(pw,4), "recall_weighted": round(rw,4), "f1_weighted": round(fw,4),
        "bleu4_mean": round(sum(bleu_s)/n,4), "meteor_mean": round(sum(meteor_s)/n,4),
        "rougeL_mean": round(sum(rouge_s)/n,4),
        "cider_mean": round(cider_score,4) if cider_score is not None else "install pycocoevalcap",
        "json_parse_failure_rate": round(parse_failures/n,4),
        "hallucination_rate": round(hallucinations/n,4),
    }

    csv_path = os.path.join(ADAPTER_PATH, "eval_per_event_predictions.csv")
    pd.DataFrame(per_event_rows).to_csv(csv_path, index=False)
    results_path = os.path.join(ADAPTER_PATH, "eval_results.json")
    with open(results_path,"w") as f: json.dump(results, f, indent=2)

    print("\n===== EVALUATION SUMMARY (TRAIN-SET) =====")
    for k,v in results.items():
        if k in ("taxonomy","NOTE"): continue
        print(f"{k}: {v}")
    print("WARNING: Train-set metrics — interpret as upper bound, not generalization.")

if __name__ == "__main__":
    main()
