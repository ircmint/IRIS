"""
main_pipeline.py
------------------
End-to-end IRC Compliance Pipeline.

Usage:
    # Full video dataset (2-wheeler dashcam footage)
    python3 main_pipeline.py --video /path/to/ride.mp4 --sample-every 15

    # Single test frame / image folder
    python3 main_pipeline.py --image /path/to/frame.jpg
    python3 main_pipeline.py --image-dir /path/to/frames_folder

Outputs (in outputs/):
    frame_reports.json         - per-frame compliance report (all detections + cited clauses)
    frame_reports.csv          - flattened violation-level table (now includes retrieval_score,
                                  retrieval_rank, query_text, absence_confidence, zone_type,
                                  infrastructure_element, severity — see README)
    retrieval_results.csv      - one row per retrieved clause candidate (for Precision@k)
    verdicts.csv               - one row per compliance verdict (for CHR / RHR)
    irc_knowledge_graph.graphml
    irc_knowledge_graph.json
    irc_knowledge_graph.html   - open in any browser, interactive
    vis/annotated_<frame_id>.jpg  - visual overlays per frame

Every violation is assigned a stable `event_id` of the form
"<frame_id>_v<violation_index>" so that frame_reports.csv, retrieval_results.csv
and verdicts.csv can all be joined on the same key. (This is distinct from the
IMU-derived event_id produced later by merge_events.py / event_compliance.csv,
which represents a physical ride-event window rather than a single violation.)
"""

import argparse
import csv
import json
import os
import cv2

from config import FRAMES_DIR, OUTPUT_DIR, FRAME_SAMPLE_EVERY_N_FRAMES
from irc_kb_builder import build_all
from compliance_engine import load_retrievers, evaluate_frame
from marking_detector import detect_markings
from sign_shape_detector import detect_signs
from object_detector import detect_objects
from visualize import annotate_frame
from knowledge_graph import IRCKnowledgeGraph
import json
import os

GOLD_MAPPING_PATH = os.path.join(os.path.dirname(__file__), "F:\CVIT\IRASTE\knowledgeGraph\irc_compliance_pipeline\gold_clause_mapping.json")

with open(GOLD_MAPPING_PATH, "r") as f:
    GOLD = json.load(f)


def extract_frames_from_video(video_path, sample_every_n):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idx = 0
    saved = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % sample_every_n == 0:
            timestamp = round(idx / fps, 2)
            frame_id = f"f{idx:06d}"
            path = os.path.join(FRAMES_DIR, f"{frame_id}.jpg")
            cv2.imwrite(path, frame)
            saved.append((frame_id, path, timestamp))
        idx += 1
    cap.release()
    return saved, fps


def process_frame(frame_id, frame_path, timestamp, retrievers, kg):
    frame = cv2.imread(frame_path)
    if frame is None:
        return None

    marking_result = detect_markings(frame)
    sign_detections = detect_signs(frame)
    object_detections = detect_objects(frame)

    compliance_report = evaluate_frame(marking_result, sign_detections,
                                        object_detections, retrievers)

    vis_path = annotate_frame(frame, sign_detections, object_detections,
                               compliance_report, frame_id)

    kg.ingest_frame_report(frame_id, timestamp, marking_result, sign_detections,
                            object_detections, compliance_report)

    return {
        "frame_id": frame_id,
        "timestamp_sec": timestamp,
        "source_path": frame_path,
        "annotated_path": vis_path,
        "marking_result": marking_result,
        "sign_detections": sign_detections,
        "object_detections": object_detections,
        "compliance": compliance_report,
    }


def _violation_event_id(frame_id, idx):
    """Stable per-violation identifier shared across frame_reports.csv,
    retrieval_results.csv and verdicts.csv."""
    return f"{frame_id}_v{idx:02d}"


def write_csv_report(reports, path):
    """frame_reports.csv — one row per violation (or one 'compliant' row per
    clean frame). EXTENDED with retrieval_score, retrieval_rank, query_text,
    absence_confidence, zone_type, infrastructure_element, severity."""
    rows = []
    for r in reports:
        if not r:
            continue
        if not r["compliance"]["violations"]:
            rows.append({
                "frame_id": r["frame_id"], "timestamp_sec": r["timestamp_sec"],
                "is_compliant": True, "rule": "", "irc_code": "", "clause_id": "",
                "clause_heading": "", "description": "No violations detected",
                "retrieval_score": "", "retrieval_rank": "", "query_text": "",
                "absence_confidence": "", "zone_type": r["compliance"].get("zone_type", "Unknown"),
                "infrastructure_element": "", "severity": "",
            })
        for idx, v in enumerate(r["compliance"]["violations"]):
            cited = v.get("cited_clause") or {}

            # --- Debug: show exactly what query drove this citation, and
            # the ranked candidate scores, so mismatches are easy to spot. ---
            print("RULE  :", v.get("rule"))
            print("QUERY :", v.get("query_text", v.get("query", "")))
            for cand in v.get("retrieved_candidates", []):
                print(f"  TOP-{cand.get('rank')} [{cand.get('clause_id')}] "
                      f"sim={cand.get('similarity')}  {cand.get('heading')}")
            print("CITED CLAUSE:")
            print(cited)
            print("-" * 50)

            rows.append({
                "frame_id": r["frame_id"], "timestamp_sec": r["timestamp_sec"],
                "is_compliant": False, "rule": v["rule"], "irc_code": v["irc_code"],
                "clause_id": cited.get("clause_id", ""),
                "clause_heading": cited.get("heading", ""),
                "description": v["description"],
                "retrieval_score": v.get("retrieval_score", ""),
                "retrieval_rank": v.get("retrieval_rank", ""),
                "query_text": v.get("query_text", v.get("query", "")),
                "absence_confidence": v.get("absence_confidence", ""),
                "zone_type": v.get("zone_type", "Unknown"),
                "infrastructure_element": v.get("infrastructure_element", ""),
                "severity": v.get("severity", ""),
            })
    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def write_retrieval_results_csv(reports, path):

    """outputs/retrieval_results.csv — one row per retrieved clause candidate
    (top-k per violation query), with a blank `human_relevant` column left for
    manual annotation (used for Precision@k)."""
    fieldnames = ["event_id", "frame_id","rule",  "query_text", "retrieval_rank",
                  "clause_id", "irc_code", "similarity_score",
                  "retrieved_clause_text", "human_relevant"]
    rows = []
    for r in reports:
        if not r:
            continue
        for idx, v in enumerate(r["compliance"]["violations"]):
            event_id = _violation_event_id(r["frame_id"], idx)
                    
            for cand in v.get("retrieved_candidates", []):


                rule = v.get("rule", "")
                expected = GOLD.get(rule, [])

                human_relevant = 1 if str(cand.get("clause_id", "")) in expected else 0

                rows.append({
                    "event_id": event_id,
                    "frame_id": r["frame_id"],
                    "rule": rule,
                    "query_text": v.get("query_text", v.get("query", "")),
                    "retrieval_rank": cand.get("rank", ""),
                    "clause_id": cand.get("clause_id", ""),
                    "irc_code": cand.get("irc_code", ""),
                    "similarity_score": cand.get("similarity", ""),
                    "retrieved_clause_text": cand.get("text_snippet", ""),
                    "human_relevant": human_relevant,
                })                
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path

def write_verdicts_csv(reports, path):
    fieldnames = [
        "event_id",
        "frame_id",
        "rule",
        "verdict",
        "cited_clause",
        "retrieved_clause_text",
        "reasoning",
        "retrieved_clause",
        "retrieval_score"
    ]

    rows = []

    for r in reports:
        if not r:
            continue

        violations = r["compliance"]["violations"]

        # Compliant frame
        if not violations:
            rows.append({
                "event_id": _violation_event_id(r["frame_id"], 0),
                "frame_id": r["frame_id"],
                "rule": "",
                "verdict": "COMPLIANT",
                "cited_clause": "",
                "reasoning": "No violations detected",
                "retrieved_clause": "",
                "retrieval_score": "",
            })
            continue

        # One row per violation
        for idx, v in enumerate(violations):

            cited = v.get("cited_clause") or {}

            rows.append({
                "event_id": _violation_event_id(r["frame_id"], idx),
                "frame_id": r["frame_id"],

                "rule": v.get("rule", ""),          # <-- THIS WAS MISSING

                "verdict": "NON_COMPLIANT",

                "cited_clause": cited.get("clause_id", ""),

                "reasoning": v.get("description", ""),

                "retrieved_clause": cited.get("heading", "") or cited.get("clause_id", ""),

                "retrieved_clause_text": cited.get("text", ""),

                "reasoning": v.get("description", ""),

                "retrieval_score": v.get("retrieval_score", ""),
            })

    
    print("=" * 60)
    print("DEBUG write_verdicts_csv()")
    print("Path:", path)
    print("Rows:", len(rows))
    print("Fieldnames:", fieldnames)
    print("=" * 60)

    try:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print("File written successfully.")
    except Exception as e:
        import traceback
        print("Exception type:", type(e).__name__)
        print("Exception:", e)
        traceback.print_exc()
        raise        
    return path


def run_pipeline(video_path=None, image_path=None, image_dir=None,
                  sample_every=FRAME_SAMPLE_EVERY_N_FRAMES, video_name="video_dataset"):
    print("[1/6] Building IRC knowledge bases (per-PDF, requested sections only)...")
    build_all()
    retrievers = load_retrievers()
    print(f"      Loaded retrievers: {list(retrievers.keys())}")

    kg = IRCKnowledgeGraph(video_name=video_name)

    frame_jobs = []  # (frame_id, path, timestamp)
    if video_path:
        print(f"[2/6] Extracting frames from video: {video_path}")
        frame_jobs, fps = extract_frames_from_video(video_path, sample_every)
        print(f"      Extracted {len(frame_jobs)} frames (source fps={fps:.1f}, "
              f"sampling every {sample_every} frames)")
    elif image_path:
        frame_jobs = [(os.path.splitext(os.path.basename(image_path))[0], image_path, 0.0)]
    elif image_dir:
        files = sorted(f for f in os.listdir(image_dir)
                        if f.lower().endswith((".jpg", ".jpeg", ".png")))
        frame_jobs = [(os.path.splitext(f)[0], os.path.join(image_dir, f), i)
                      for i, f in enumerate(files)]
    else:
        raise ValueError("Provide --video, --image, or --image-dir")

    print(f"[3/6] Running detection + compliance evaluation on {len(frame_jobs)} frame(s)...")
    reports = []
    for frame_id, path, ts in frame_jobs:
        rep = process_frame(frame_id, path, ts, retrievers, kg)
        reports.append(rep)
        if rep:
            verdict = "COMPLIANT" if rep["compliance"]["is_irc_compliant"] else \
                f"NON-COMPLIANT ({rep['compliance']['num_violations']} issue(s))"
            print(f"      {frame_id} @ {ts}s -> {verdict}")

    print("[4/6] Writing reports...")
    json_path = os.path.join(OUTPUT_DIR, "frame_reports.json")
    with open(json_path, "w") as f:
        json.dump(reports, f, indent=2, default=str)
    csv_path = os.path.join(OUTPUT_DIR, "frame_reports.csv")
    write_csv_report(reports, csv_path)

    print("[5/6] Writing retrieval + verdict evaluation exports...")
    retrieval_path = os.path.join(OUTPUT_DIR, "retrieval_results.csv")
    write_retrieval_results_csv(reports, retrieval_path)
    verdicts_path = os.path.join(OUTPUT_DIR, "verdicts.csv")
    write_verdicts_csv(reports, verdicts_path)

    print("[6/6] Exporting knowledge graph...")
    graphml_path = kg.export_graphml()
    kg_json_path = kg.export_json()
    html_path = kg.export_html()
    stats = kg.summary_stats()

    print("\n=== SUMMARY ===")
    total = len([r for r in reports if r])
    non_compliant = len([r for r in reports if r and not r["compliance"]["is_irc_compliant"]])
    print(f"Frames processed : {total}")
    print(f"Non-compliant    : {non_compliant}")
    print(f"Compliant        : {total - non_compliant}")
    print(f"KG nodes/edges   : {stats['total_nodes']} / {stats['total_edges']}")
    print(f"Outputs          : {json_path}\n                   {csv_path}\n"
          f"                   {retrieval_path}\n                   {verdicts_path}\n"
          f"                   {graphml_path}\n                   {kg_json_path}\n"
          f"                   {html_path}")

    return {
        "reports": reports,
        "kg_stats": stats,
        "paths": {
            "frame_reports_json": json_path,
            "frame_reports_csv": csv_path,
            "retrieval_results_csv": retrieval_path,
            "verdicts_csv": verdicts_path,
            "graphml": graphml_path,
            "kg_json": kg_json_path,
            "kg_html": html_path,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IRC 2-Wheeler Video Compliance Pipeline")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file")
    parser.add_argument("--image", type=str, default=None, help="Path to a single test image/frame")
    parser.add_argument("--image-dir", type=str, default=None, help="Path to a folder of frames")
    parser.add_argument("--sample-every", type=int, default=FRAME_SAMPLE_EVERY_N_FRAMES,
                         help="Sample every Nth video frame")
    parser.add_argument("--video-name", type=str, default="video_dataset",
                         help="Label for the Video node in the knowledge graph")
    args = parser.parse_args()

    run_pipeline(video_path=args.video, image_path=args.image, image_dir=args.image_dir,
                 sample_every=args.sample_every, video_name=args.video_name)