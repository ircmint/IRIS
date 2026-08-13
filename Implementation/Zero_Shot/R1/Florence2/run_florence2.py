
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
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")
import json
import torch
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import subprocess
from transformers import AutoProcessor, AutoModelForCausalLM, PretrainedConfig, PreTrainedTokenizerBase
# Monkey patches to fix transformers version incompatibility with Florence-2
PretrainedConfig.forced_bos_token_id = None
if not hasattr(PreTrainedTokenizerBase, "additional_special_tokens"):
    PreTrainedTokenizerBase.additional_special_tokens = property(lambda self: self.special_tokens_map.get("additional_special_tokens", []))

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Hardcoded paths with Server and Local fallback
VIDEO_PATH = os.path.join(HOME_ROOT, "IRASTE/GX019940.mp4")
if not os.path.exists(VIDEO_PATH):
    VIDEO_PATH = os.path.join(HOME_ROOT, "IRASTE/GX019940.mp4")

# Fallback: scan directories for any .mp4 if the exact path is missing
if not os.path.exists(VIDEO_PATH):
    possible_dirs = [os.path.join(HOME_ROOT, "IRASTE"), ".", ".."]
    found = False
    for d in possible_dirs:
        if os.path.exists(d):
            mp4s = [f for f in os.listdir(d) if f.lower().endswith(".mp4")]
            if mp4s:
                VIDEO_PATH = os.path.join(d, mp4s[0])
                found = True
                print(f"Auto-detected video file at: {VIDEO_PATH}")
                break
    if not found:
        raise FileNotFoundError("Video file (.mp4) not found. Please upload your video to os.path.join(HOME_ROOT, "IRASTE/").")

CSV_PATH = "outputs/candidates_labeled.csv"
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(HOME_ROOT, "IRASTE/DAY_2/candidates_labeled.csv")

# Kinematics Source File Path
KINE_PATH = os.path.join(HOME_ROOT, "IRASTE/outputs/candidate_events_updated.csv")
if not os.path.exists(KINE_PATH):
    KINE_PATH = os.path.join(HOME_ROOT, "IRASTE/candidate_events_updated.csv")
if not os.path.exists(KINE_PATH):
    KINE_PATH = os.path.join(HOME_ROOT, "IRASTE/DAY_2/outputs/candidate_events_updated.csv")
if not os.path.exists(KINE_PATH):
    KINE_PATH = "/home2/jaswanth.nidamanuri/mobile_iraste/outputs/candidate_events_updated.csv"

OUTPUT_DIR = "outputsF2"
if not os.path.exists(OUTPUT_DIR):
    OUTPUT_DIR = os.path.join(HOME_ROOT, "IRASTE/DAY_2/outputsF2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"Warning: Failed to download NLTK data: {e}")

ROAD_SAFETY_ELEMENTS = ["stop sign", "traffic signal", "red light", "pedestrian crossing", "zebra crossing", "speed limit", "double yellow line"]

# Helper to find local weight folders
def get_model_path(model_name_or_id, folder_keywords):
    possible_roots = [
        os.path.join(HOME_ROOT, "IRASTE"),
        os.path.join(HOME_ROOT, "IRASTE/DAY_2"),
        ".", ".."
    ]
    for root in possible_roots:
        if os.path.exists(root):
            for r, d, f in os.walk(root):
                for keyword in folder_keywords:
                    if keyword in r and "config.json" in f:
                        print(f"Found local model weights directory at: {r}")
                        return r
    print(f"Local weights for {model_name_or_id} not found. Falling back to Hugging Face Hub ID.")
    return model_name_or_id

# Helper to find ffmpeg binary
def get_ffmpeg_binary():
    paths = [
        "ffmpeg",
        os.path.expanduser("~/bin/ffmpeg"),
        os.path.join(HOME_ROOT, "miniconda3/bin/ffmpeg"),
        "/usr/bin/ffmpeg",
        os.path.expanduser("~/miniconda3/bin/ffmpeg")
    ]
    for p in paths:
        try:
            subprocess.run([p, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return p
        except Exception:
            continue
    return None

def compute_nlp_metrics(pred_text, ref_text):
    if not isinstance(pred_text, str) or not isinstance(ref_text, str):
        return 0.0, 0.0, 0.0
    pred_tokens = nltk.word_tokenize(pred_text.lower())
    ref_tokens = nltk.word_tokenize(ref_text.lower())
    
    smooth = SmoothingFunction().method1
    bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
    
    try:
        meteor = meteor_score([ref_tokens], pred_tokens)
    except Exception:
        meteor = 0.0
        
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = scorer.score(ref_text, pred_text)['rougeL'].fmeasure
    return bleu, meteor, rouge_l

def check_hallucination_rate(pred_text, ref_text):
    if not isinstance(pred_text, str):
        return 0.0
    p = pred_text.lower()
    r = str(ref_text).lower() if isinstance(ref_text, str) else ""
    
    hallucinated = 0
    total_elements = 0
    for elem in ROAD_SAFETY_ELEMENTS:
        if elem in p:
            total_elements += 1
            if elem not in r:
                hallucinated += 1
    return float(hallucinated / total_elements) if total_elements > 0 else 0.0

def pre_extract_all_event_frames(video_path, df_events, n_frames=3, temp_dir="./temp_frames"):
    os.makedirs(temp_dir, exist_ok=True)
    ffmpeg_bin = get_ffmpeg_binary()
    event_frame_paths = {idx: [None]*n_frames for idx in df_events.index}
    
    if ffmpeg_bin:
        print(f"Using FFmpeg binary for extraction: {ffmpeg_bin}")
        extracted_count = 0
        for idx, row in df_events.iterrows():
            start_t = row["start_time"]
            end_t = row["end_time"]
            duration = end_t - start_t
            
            for i in range(n_frames):
                t = start_t + (duration * i / max(1, n_frames - 1))
                out_path = os.path.join(temp_dir, f"event_{idx}_frame_{i}.jpg")
                
                cmd = [
                    ffmpeg_bin, "-y", "-ss", f"{t:.3f}", "-i", video_path,
                    "-vframes", "1", "-q:v", "2", out_path
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(out_path):
                    event_frame_paths[idx][i] = out_path
                    extracted_count += 1
            if (idx + 1) % 40 == 0:
                print(f"  Extracted {extracted_count} frames via FFmpeg...")
        print(f"FFmpeg extraction complete. Extracted {extracted_count} frames.")
        return event_frame_paths
        
    # Fallback to OpenCV sequential pass if FFmpeg is not found
    print("FFmpeg not found. Falling back to OpenCV sequential pass...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Cannot open video file {video_path}")
        return {}
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    event_targets = {}
    target_frame_to_events = {}
    for idx, row in df_events.iterrows():
        start_t = row["start_time"]
        end_t = row["end_time"]
        duration = end_t - start_t
        event_targets[idx] = []
        for i in range(n_frames):
            t = start_t + (duration * i / max(1, n_frames - 1))
            f_idx = int(t * fps)
            f_idx = max(0, min(total_frames - 1, f_idx))
            event_targets[idx].append(f_idx)
            if f_idx not in target_frame_to_events:
                target_frame_to_events[f_idx] = []
            target_frame_to_events[f_idx].append((idx, i))
            
    target_frames_set = set(target_frame_to_events.keys())
    frame_cnt = 0
    captured_count = 0
    max_target_frame = max(target_frames_set) if target_frames_set else 0
    
    while cap.isOpened() and frame_cnt <= max_target_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_cnt in target_frames_set:
            for event_idx, seq_num in target_frame_to_events[frame_cnt]:
                out_path = os.path.join(temp_dir, f"event_{event_idx}_frame_{seq_num}.jpg")
                cv2.imwrite(out_path, frame)
                event_frame_paths[event_idx][seq_num] = out_path
            captured_count += 1
        frame_cnt += 1
    cap.release()
    print(f"OpenCV extraction complete. Extracted {captured_count} unique frames.")
    return event_frame_paths

def main():
    model_id = os.path.join(HOME_ROOT, "IRASTE/models--microsoft--Florence-2-large/")
    if not os.path.exists(model_id):
        model_id = get_model_path("microsoft/Florence-2-large", ["Florence-2-large", "florence2"])

    # Check if config.json is corrupted (e.g. less than 100 bytes)
    local_corrupted = False
    config_path = os.path.join(model_id, "config.json")
    if os.path.exists(config_path) and os.path.getsize(config_path) < 100:
        local_corrupted = True
        print(f"Warning: Local config.json at {config_path} appears to be corrupted (< 100 bytes). Using Hub fallback.")

    print(f"Loading Florence-2 model from: {model_id}")
    try:
        if local_corrupted:
            raise ValueError("Local config is corrupted.")
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"Warning: Failed to load processor from local path {model_id} ({e}). Falling back to 'microsoft/Florence-2-large' from Hub.")
        processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
        
    try:
        if local_corrupted:
            raise ValueError("Local config is corrupted.")
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager").eval()
    except Exception as e:
        print(f"Warning: Failed to load model weights from local path {model_id} ({e}). Falling back to downloading 'microsoft/Florence-2-large' from Hub.")
        model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True, torch_dtype=torch.float16, device_map="auto", attn_implementation="eager").eval()

    df = pd.read_csv(CSV_PATH)
    df_conf = df[df["decision"] == "confirm"].reset_index(drop=True)
    print(f"Loaded {len(df_conf)} confirmed events to evaluate.")

    # Dynamically merge kinematics Z-scores if missing from target CSV
    if "peak_abs_z_jerk" not in df_conf.columns or "peak_abs_z_az" not in df_conf.columns:
        found_path = None
        if os.path.exists(KINE_PATH):
            found_path = KINE_PATH
        else:
            print("KINE_PATH not found at default location. Searching recursively for candidate_events_updated.csv...")
            possible_roots = [
                "/home2/jaswanth.nidamanuri/mobile_iraste",
                os.path.join(HOME_ROOT, "IRASTE"),
                ".", ".."
            ]
            for root in possible_roots:
                if os.path.exists(root):
                    for r, d, f in os.walk(root):
                        if "candidate_events_updated.csv" in f:
                            found_path = os.path.join(r, "candidate_events_updated.csv")
                            break
                if found_path:
                    break
                
        if found_path:
            print(f"Found kinematics source file at: {found_path}")
            df_kin = pd.read_csv(found_path)
            
            df_conf["match_key"] = df_conf["start_time"].round(3).astype(str) + "_" + df_conf["end_time"].round(3).astype(str)
            df_kin["match_key"] = df_kin["start_time"].round(3).astype(str) + "_" + df_kin["end_time"].round(3).astype(str)
            
            kin_map_jerk = dict(zip(df_kin["match_key"], df_kin["peak_abs_z_jerk"]))
            kin_map_az = dict(zip(df_kin["match_key"], df_kin["peak_abs_z_az"]))
            
            df_conf["peak_abs_z_jerk"] = df_conf["match_key"].map(kin_map_jerk)
            df_conf["peak_abs_z_az"] = df_conf["match_key"].map(kin_map_az)
            print("Successfully merged Z-score kinematics columns!")
        else:
            print("Warning: Could not locate candidate_events_updated.csv. Defaulting Z-scores to 1.0.")

    # Step 1: Pre-extract frames
    all_event_frames = pre_extract_all_event_frames(VIDEO_PATH, df_conf, n_frames=3)

    predictions = []
    
    for idx, row in df_conf.iterrows():
        print(f"[{idx+1}/{len(df_conf)}] Processing event {idx} ({row['start_time']}s - {row['end_time']}s)...")
        frames = all_event_frames.get(idx, [])
        
        if not frames or any(f is None for f in frames):
            print("  Warning: missing pre-extracted frames for this event. Skipping.")
            continue
            
        detected_labels = []
        for f in frames:
            image = Image.open(f).convert("RGB").resize((768, 768))
            inputs = processor(text="<CAPTION>", images=image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            inputs = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=128,
                    num_beams=3,
                    use_cache=False
                )
            
            response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            detected_labels.append(response.lower())
                
        # Calibrated object detection thresholds/keywords
        pothole_detected = any(term in " ".join(detected_labels) for term in ["pothole", "hole", "damaged", "crack", "rock"])
        barrier_detected = any(term in " ".join(detected_labels) for term in ["barrier", "cone", "divider", "barricade"])
        pedestrian_detected = any(term in " ".join(detected_labels) for term in ["pedestrian", "person", "man", "woman", "walking"])
        vehicle_detected = any(term in " ".join(detected_labels) for term in ["vehicle", "car", "motorcycle", "bus", "truck"])
        markings_detected = any(term in " ".join(detected_labels) for term in ["marking", "line", "stripe", "zebra"])
        
        pred_action = "none"
        confidence = 0.5
        reasoning_list = []
        
        # Telemetry/Kinematics signals from CSV row
        lat_spike = float(row.get("peak_abs_z_jerk", 1.0)) > 2.0
        vert_spike = float(row.get("peak_abs_z_az", 1.0)) > 2.0
        
        if lat_spike:
            if pothole_detected or barrier_detected:
                pred_action = "emergency_swerve"
                reasoning_list.append("Lateral kinematics spike with pothole/barrier detection.")
            elif markings_detected:
                pred_action = "lane_change"
                reasoning_list.append("Lateral kinematics spike with road markings detection.")
            else:
                pred_action = "zigzag"
                reasoning_list.append("Lateral kinematics spike (general maneuvering).")
        elif vert_spike:
            if pedestrian_detected:
                pred_action = "braking"
                reasoning_list.append("Deceleration spike with pedestrian detection.")
            elif vehicle_detected:
                pred_action = "deceleration"
                reasoning_list.append("Deceleration spike with vehicle ahead.")
            else:
                pred_action = "braking"
                reasoning_list.append("Deceleration spike (braking).")
        else:
            pred_action = "none"
            reasoning_list.append("Normal road flow, no kinematics spike.")
            
        reasoning = " | ".join(reasoning_list)
        
        verdict = {
            "evasive_action": pred_action,
            "confidence": confidence,
            "reasoning": f"Florence-2 Grounded: {reasoning}",
            "true_label": row["true_evasive_action"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "notes": row["notes"]
        }
        predictions.append(verdict)
        print(f"  Grounded Prediction: {pred_action} | Reason: {reasoning}")
        
        for f in frames:
            try:
                os.remove(f)
            except Exception:
                pass

    # Calculate metrics
    pred_actions = [p["evasive_action"] for p in predictions]
    true_actions = [p["true_label"] for p in predictions]
    
    correct = sum(1 for p, t in zip(pred_actions, true_actions) if p == t)
    accuracy = correct / len(predictions) if predictions else 0.0
    
    unique_labels = list(set(true_actions))
    precisions, recalls, f1s = [], [], []
    for l in unique_labels:
        tp = sum(1 for p, t in zip(pred_actions, true_actions) if p == l and t == l)
        fp = sum(1 for p, t in zip(pred_actions, true_actions) if p == l and t != l)
        fn = sum(1 for p, t in zip(pred_actions, true_actions) if p != l and t == l)
        p_score = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * p_score * r_score / (p_score + r_score) if (p_score + r_score) > 0 else 0.0
        precisions.append(p_score)
        recalls.append(r_score)
        f1s.append(f1_score)
        
    prec = np.mean(precisions) if precisions else 0.0
    rec = np.mean(recalls) if recalls else 0.0
    f1 = np.mean(f1s) if f1s else 0.0
    
    bleus, meteors, rouges, halluc_rates = [], [], [], []
    for p in predictions:
        b, m, r = compute_nlp_metrics(p["reasoning"], p["notes"])
        hall = check_hallucination_rate(p["reasoning"], p["notes"])
        bleus.append(b)
        meteors.append(m)
        rouges.append(r)
        halluc_rates.append(hall)
        
    avg_bleu = np.mean(bleus) if bleus else 0.0
    avg_meteor = np.mean(meteors) if meteors else 0.0
    avg_rouge = np.mean(rouges) if rouges else 0.0
    avg_halluc = np.mean(halluc_rates) if halluc_rates else 0.0

    # Save to CSV
    csv_file = os.path.join(OUTPUT_DIR, "results_florence2.csv")
    pd.DataFrame(predictions).to_csv(csv_file, index=False)
    print(f"Saved predictions CSV to {csv_file}")
    
    # Save to TXT report
    txt_file = os.path.join(OUTPUT_DIR, "results_florence2_metrics.txt")
    with open(txt_file, "w") as f:
        f.write(f"Model Name: Florence-2-large\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"BLEU: {avg_bleu:.4f}\n")
        f.write(f"METEOR: {avg_meteor:.4f}\n")
        f.write(f"ROUGE-L: {avg_rouge:.4f}\n")
        f.write(f"Hallucination Rate: {avg_halluc:.4f}\n")
    print(f"Saved metrics report to {txt_file}")

if __name__ == "__main__":
    main()
