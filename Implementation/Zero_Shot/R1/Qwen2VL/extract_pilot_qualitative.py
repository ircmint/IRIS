"""
extract_pilot_qualitative.py -- real TP qualitative frame extraction for the
PilotData / qwen25-3b / summarized_telemetry run. Uses cv2 (no ffmpeg
available on this host) to grab the mid-window frame for each real TP
(gold decision==confirm AND model pred_is_evasive==True), then burns a
caption bar (model, timestamp, gold category, real model reasoning -
verbatim, never invented) onto an annotated copy via PIL, exactly the same
information content as extract_qualitative.py's annotate_frame().
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
import os, sys, json, csv, textwrap
import cv2
from PIL import Image, ImageDraw, ImageFont

VIDEO = os.path.join(HOME_ROOT, "IRASTE/GX019940.MP4")
GOLD = os.path.join(HOME_ROOT, "Custom_Data_Results/PilotData/gold_candidates.csv")
PRED = os.path.join(HOME_ROOT, "Custom_Data_Results/predictions/PilotData__qwen25-3b__summarized_telemetry.json")
OUT_DIR = os.path.join(HOME_ROOT, "Custom_Data_Results/Results/qualitative_examples/PilotData__qwen25-3b__summarized_telemetry")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def _load_font(size):
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def annotate(frame_bgr, out_path, model, rider, start_time, end_time, gold_category, reasoning, confidence):
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    w, h = img.size
    title_font = _load_font(max(16, w // 45))
    body_font = _load_font(max(13, w // 60))
    header = f"{model}  |  {rider}  |  {start_time:.1f}s-{end_time:.1f}s  |  gold: {gold_category}"
    reasoning_text = reasoning or "(no reasoning text in prediction)"
    if len(reasoning_text) > 280:
        reasoning_text = reasoning_text[:280].rsplit(" ", 1)[0] + " ..."
    wrap_width = max(30, w // 11)
    body_lines = textwrap.wrap(f"reasoning: {reasoning_text}", width=wrap_width)
    if confidence is not None:
        body_lines.append(f"confidence: {confidence}")
    line_h = body_font.size + 6
    title_h = title_font.size + 10
    bar_h = title_h + line_h * len(body_lines) + 16
    overlay = Image.new("RGBA", (w, bar_h), (0, 0, 0, 165))
    draw = ImageDraw.Draw(overlay)
    y = 8
    draw.text((10, y), header, font=title_font, fill=(255, 255, 255, 255))
    y += title_h
    for line in body_lines:
        draw.text((10, y), line, font=body_font, fill=(255, 255, 0, 255))
        y += line_h
    canvas = Image.new("RGB", (w, h + bar_h), (0, 0, 0))
    canvas.paste(img, (0, bar_h))
    canvas.paste(overlay, (0, 0), overlay)
    canvas.save(out_path, quality=92)


def main():
    with open(GOLD) as f:
        gold = list(csv.DictReader(f))
    with open(PRED) as f:
        preds = json.load(f)

    tps = []
    for i, p in enumerate(preds):
        if i >= len(gold):
            break
        if gold[i]["decision"] == "confirm" and p.get("pred_is_evasive") is True:
            tps.append((i, p))
    print(f"Found {len(tps)} real TPs (gold=confirm AND model pred=True). Rendering {min(N, len(tps))}.")

    os.makedirs(OUT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    md_lines = [f"# PilotData / qwen25-3b / summarized_telemetry -- real True Positive examples\n",
                f"Found {len(tps)} real TP out of 262 scored events. Showing {min(N, len(tps))}.\n"]

    for i, p in tps[:N]:
        st, et = p["start_time"], p["end_time"]
        mid = (st + et) / 2.0
        frame_idx = int(mid * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            print(f"  event {i}: FAILED to read frame at {mid:.2f}s")
            continue
        raw_path = os.path.join(OUT_DIR, f"tp_event{i}.jpg")
        cv2.imwrite(raw_path, frame)
        ann_path = os.path.join(OUT_DIR, f"tp_event{i}_annotated.jpg")
        annotate(frame, ann_path, model="Qwen2.5-VL-3B", rider="PilotData",
                 start_time=st, end_time=et, gold_category=gold[i]["category"],
                 reasoning=p.get("pred_reasoning", ""), confidence=p.get("pred_confidence"))
        print(f"  event {i} ({st:.2f}-{et:.2f}s): wrote {ann_path}")
        md_lines.append(f"## Event {i} ({st:.1f}s-{et:.1f}s, gold category={gold[i]['category']})")
        md_lines.append(f"- Annotated frame: `{os.path.basename(ann_path)}`")
        md_lines.append(f"- Gold reviewer note: {gold[i].get('notes','')}")
        md_lines.append(f"- Model confidence: {p.get('pred_confidence')}")
        md_lines.append(f"- Model reasoning: {p.get('pred_reasoning','')}\n")

    cap.release()
    with open(os.path.join(OUT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print("wrote", os.path.join(OUT_DIR, "README.md"))


if __name__ == "__main__":
    main()
