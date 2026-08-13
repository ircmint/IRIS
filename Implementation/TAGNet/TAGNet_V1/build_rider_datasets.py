"""
Build per-rider event datasets for ICVJP pipeline.
Reads evasive_actions CSV + imu_processed CSV per rider.
Groups consecutive labeled frames into events, picks 30 balanced events,
extracts ~5-second video clips with ffmpeg, and writes a dataset JSON
compatible with the existing Ada pipeline.

Usage:
    python build_rider_datasets.py [--rider R1|R2|R3|R4|ALL] [--dry-run]

Output per rider:
    G:/ZS-Compliance-Pipeline/{Rider}/clips/    <- extracted mp4 clips
    G:/ZS-Compliance-Pipeline/{Rider}/{Rider}_dataset.json
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

import os, sys, csv, json, re, subprocess, argparse
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────

RIDERS = {
    "R1": {
        "name":    "Rider1_NJ",
        "base":    r"G:\Driver_Behaviour\pipeline\Custom_Data\Rider1_NJ",
        "video":   r"G:\Driver_Behaviour\pipeline\Custom_Data\Rider1_NJ\Rider1_NJ_720p.mp4",
        "actions": r"G:\Driver_Behaviour\pipeline\Custom_Data\Rider1_NJ\Rider1_NJ_evasive_actions.csv",
        "imu":     r"G:\Driver_Behaviour\pipeline\Custom_Data\Rider1_NJ\Rider1_NJ_imu_processed.csv",
        "out_dir": r"G:\ZS-Compliance-Pipeline\R1",
    },
    "R2": {
        "name":    "Rider2_AZ",
        "base":    r"G:\Driver_Behaviour\pipeline\Custom_Data\Rider2_AZ",
        "video":   r"G:\Driver_Behaviour\pipeline\Custom_Data\Rider2_AZ\Rider2_AZ_720p.mp4",
        "actions": r"G:\Driver_Behaviour\pipeline\Custom_Data\Rider2_AZ\Rider2_AZ_evasive_actions.csv",
        "imu":     r"G:\Driver_Behaviour\pipeline\Custom_Data\Rider2_AZ\Rider2_AZ_imu_processed.csv",
        "out_dir": r"G:\ZS-Compliance-Pipeline\R2",
    },
    "R3": {
        "name":    "Rider3_VA",
        "base":    r"G:\Driver_Behaviour\pipeline\Custom_Data\Rider3_VA",
        "video":   r"G:\Driver_Behaviour\pipeline\Custom_Data\Rider3_VA\Rider3_VA_720p.mp4",
        "actions": r"G:\Driver_Behaviour\pipeline\Custom_Data\Rider3_VA\Rider3_VA_evasive_actions.csv",
        "imu":     r"G:\Driver_Behaviour\pipeline\Custom_Data\Rider3_VA\Rider3_VA_imu_processed.csv",
        "out_dir": r"G:\ZS-Compliance-Pipeline\R3",
    },
    "R4": {
        "name":    "Rider4_UC",
        "base":    r"G:\Driver_Behaviour\pipeline\Custom_Data\Rider4_UC",
        "video":   r"G:\Driver_Behaviour\pipeline\Custom_Data\Rider4_UC\Rider4_UC_720p.mp4",
        "actions": r"G:\Driver_Behaviour\pipeline\Custom_Data\Rider4_UC\Rider4_UC_evasive_actions.csv",
        "imu":     r"G:\Driver_Behaviour\pipeline\Custom_Data\Rider4_UC\Rider4_UC_imu_processed.csv",
        "out_dir": r"G:\ZS-Compliance-Pipeline\R4",
    },
}

# Map raw CSV labels → pipeline action labels
ACTION_MAP = {
    "Acceleration":     "Acceleration",
    "Deceleration":     "Deceleration",
    "Lane_Change":      "Lane_Change",
    "Braking":          "Hard_Braking",
    "Emergency_Swerve": "Hard_Braking",
    # Zigzag excluded — not in pipeline
}

TARGET_CATEGORIES = ["Acceleration", "Deceleration", "Hard_Braking", "Lane_Change"]
EVENTS_PER_RIDER  = 30          # total events per rider
EVENTS_PER_CAT    = 8           # max per category (4 × 8 = 32; capped at 30)
MIN_EVENT_FRAMES  = 1           # minimum frames to qualify as an event (IMU grouping)
GAP_FRAMES        = 15          # gap in frames to split events (≈0.5s at 30fps)
CLIP_PAD_S        = 1.5         # seconds of padding before/after event
ADA_SCRATCH       = SCRATCH_ROOT  # Ada target root

# GPS (same for all riders — Hyderabad study area)
GPS_LAT   = 17.538
GPS_LON   = 78.237
GPS_LOCATION = "Outer Ring Road, Hyderabad, Telangana, India"

# Gold category → pipeline action mapping
GOLD_ACTION_MAP = {
    "acceleration": "Acceleration",
    "deceleration": "Deceleration",
    "lane_change":  "Lane_Change",
    "braking":      "Hard_Braking",
    "hard_braking": "Hard_Braking",
    # zigzag / unknown excluded
}

# ── HELPERS ──────────────────────────────────────────────────────────────────

def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def load_gold_events(gold_path):
    """Load confirmed gold events, sorted by category then duration (longest first)."""
    rows = read_csv(gold_path)
    events = []
    for r in rows:
        if r.get("decision","").strip().lower() != "confirm":
            continue
        cat_raw = r.get("category","").strip().lower()
        action  = GOLD_ACTION_MAP.get(cat_raw)
        if action is None:
            continue
        start = float(r["start_time"])
        end   = float(r["end_time"])
        events.append({
            "action":       action,
            "raw_action":   cat_raw,
            "source":       "gold",
            "start_time_s": start,
            "end_time_s":   end,
            "mid_time_s":   (start + end) / 2,
            "duration":     end - start,
            "notes":        r.get("notes",""),
        })
    # Sort by duration descending (longest/richest events first)
    events.sort(key=lambda e: e["duration"], reverse=True)
    return events

def group_events(rows):
    """Group consecutive frames of the same (mapped) action into event windows.

    Normal frames within GAP_FRAMES of the last seen action frame are bridged
    over — only a different non-Normal action or a gap larger than GAP_FRAMES
    ends the current event.
    """
    events = []
    i = 0
    while i < len(rows):
        row = rows[i]
        raw_action = row.get("evasive_action", "Normal")
        action = ACTION_MAP.get(raw_action)
        if action is None:
            i += 1
            continue
        # Start of a new event
        start_i    = i
        start_frame = int(row["video_frame"])
        last_action_frame = start_frame
        last_action_i     = i
        j = i + 1
        while j < len(rows):
            rj = rows[j]
            raw_j = rj.get("evasive_action", "Normal")
            act_j = ACTION_MAP.get(raw_j)   # None for Normal/Zigzag
            cur_frame = int(rj["video_frame"])
            gap = cur_frame - last_action_frame
            if act_j == action and gap <= GAP_FRAMES:
                # Continuation of same action (even with bridged Normal frames)
                last_action_frame = cur_frame
                last_action_i     = j
                j += 1
            elif act_j is None and gap <= GAP_FRAMES:
                # Normal/Zigzag frame within gap — bridge over it
                j += 1
            else:
                # Different action or gap too large — end event
                break

        # Event spans from start_i to last_action_i (ignore trailing Normal frames)
        end_i     = last_action_i
        end_frame = int(rows[end_i]["video_frame"])
        n_action  = sum(1 for r in rows[start_i:end_i+1]
                        if ACTION_MAP.get(r.get("evasive_action","Normal")) == action)
        if n_action >= MIN_EVENT_FRAMES:
            mid_frame = (start_frame + end_frame) // 2
            mid_row = min(rows[start_i:end_i+1],
                          key=lambda r: abs(int(r["video_frame"]) - mid_frame))
            # Signal strength from action frames only
            action_rows_w = [r for r in rows[start_i:end_i+1]
                             if ACTION_MAP.get(r.get("evasive_action","Normal")) == action]
            events.append({
                "action":       action,
                "raw_action":   raw_action,
                "start_frame":  start_frame,
                "end_frame":    end_frame,
                "mid_frame":    int(mid_row["video_frame"]),
                "start_time_s": float(rows[start_i]["video_time_s"]),
                "end_time_s":   float(rows[end_i]["video_time_s"]),
                "mid_time_s":   float(mid_row["video_time_s"]),
                "n_frames":     n_action,
                "lay_bp_max":   max(abs(float(r.get("lay_bp_mean","0") or 0)) for r in action_rows_w),
                "jerk_mean":    sum(float(r.get("jerk_mean","0") or 0) for r in action_rows_w) / len(action_rows_w),
            })
        i = j
    return events

def select_events(events):
    """Select balanced events per category, prefer longer/stronger events."""
    by_cat = defaultdict(list)
    for e in events:
        by_cat[e["action"]].append(e)
    # Sort each category: prefer events with stronger signal and longer duration
    for cat in by_cat:
        by_cat[cat].sort(key=lambda e: (e["lay_bp_max"] + abs(e["jerk_mean"]/1000), e["n_frames"]), reverse=True)
    selected = []
    for cat in TARGET_CATEGORIES:
        evs = by_cat.get(cat, [])
        selected.extend(evs[:EVENTS_PER_CAT])
    # Trim to EVENTS_PER_RIDER
    selected = selected[:EVENTS_PER_RIDER]
    return selected

def get_telemetry(imu_rows, start_frame, end_frame):
    """Extract mean telemetry for event frame window from imu_processed CSV."""
    window = [r for r in imu_rows if start_frame <= int(r.get("frame", 0)) <= end_frame]
    if not window:
        return {"ay": 0.0, "ax": 0.0, "az": 0.0, "jerk": 0.0}
    def mean_col(col):
        vals = []
        for r in window:
            v = r.get(col, "")
            try: vals.append(float(v))
            except: pass
        return sum(vals) / len(vals) if vals else 0.0
    return {
        "ay":   round(mean_col("lay_bp"), 4),
        "ax":   round(mean_col("lax_bp"), 4),
        "az":   round(mean_col("laz_bp"), 4),
        "jerk": round(mean_col("jerk"),   4),
    }

def extract_clip(video_path, start_s, end_s, out_path, dry_run=False):
    """Extract a clip from the video using ffmpeg."""
    duration = max(0.5, end_s - start_s)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_s:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-an",  # no audio (dashcam)
        out_path
    ]
    if dry_run:
        print(f"  [DRY] {' '.join(cmd)}")
        return True
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [FFMPEG ERROR] {result.stderr[-300:]}")
        return False
    return True

# ── MAIN ─────────────────────────────────────────────────────────────────────

def build_rider(rider_key, dry_run=False):
    cfg = RIDERS[rider_key]
    print(f"\n{'='*60}")
    print(f"Building dataset for {cfg['name']} ({rider_key})")
    print(f"{'='*60}")

    out_dir   = cfg["out_dir"]
    clips_dir = os.path.join(out_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    # 1. Load gold confirmed events (primary source)
    gold_path = os.path.join(cfg["base"], "gold_candidates.csv")
    gold_events = load_gold_events(gold_path) if os.path.exists(gold_path) else []
    from collections import Counter
    gold_counts = Counter(e["action"] for e in gold_events)
    print(f"Gold confirmed: {len(gold_events)} events")
    for cat, n in sorted(gold_counts.items()): print(f"  {cat}: {n}")

    # 2. Load IMU for telemetry lookup
    imu_rows = read_csv(cfg["imu"])

    # 3. Load evasive_actions for IMU-based fallback events
    action_rows = read_csv(cfg["actions"])
    imu_events_raw = group_events(action_rows)
    imu_events     = select_events(imu_events_raw)
    # Convert IMU events to same format as gold events
    imu_converted = []
    for ev in imu_events:
        imu_converted.append({
            "action":       ev["action"],
            "raw_action":   ev["raw_action"],
            "source":       "imu",
            "start_time_s": ev["start_time_s"],
            "end_time_s":   ev["end_time_s"],
            "mid_time_s":   ev["mid_time_s"],
            "duration":     ev["end_time_s"] - ev["start_time_s"],
            "notes":        "",
        })

    # 4. Merge: gold first, then IMU to fill up to 30
    by_cat = defaultdict(list)
    for ev in gold_events:   by_cat[ev["action"]].append(ev)
    for ev in imu_converted: by_cat[ev["action"]].append(ev)  # IMU appended after gold

    selected = []
    for cat in TARGET_CATEGORIES:
        evs = by_cat.get(cat, [])
        # gold events already sorted by duration; IMU by signal
        # deduplicate by time proximity (>2s apart)
        kept = []
        for ev in evs:
            overlap = any(abs(ev["mid_time_s"] - k["mid_time_s"]) < 2.0 for k in kept)
            if not overlap:
                kept.append(ev)
            if len(kept) >= EVENTS_PER_CAT:
                break
        selected.extend(kept)
    selected = selected[:EVENTS_PER_RIDER]

    cat_sel = Counter(e["action"] for e in selected)
    print(f"\nSelected {len(selected)} events (gold+imu):")
    for cat, n in sorted(cat_sel.items()): print(f"  {cat}: {n}")

    # 5. Extract clips and build dataset JSON
    dataset  = []
    event_id = 1
    skipped  = 0

    for ev in selected:
        clip_start = max(0, ev["start_time_s"] - CLIP_PAD_S)
        clip_end   = ev["end_time_s"] + CLIP_PAD_S
        src_tag    = "G" if ev["source"]=="gold" else "I"
        clip_fname = f"event_{event_id:03d}_{ev['action']}.mp4"
        clip_local = os.path.join(clips_dir, clip_fname)
        clip_ada   = f"{ADA_SCRATCH}/{cfg['name']}/clips/{clip_fname}"

        print(f"  [{event_id:2d}][{src_tag}] {ev['action']:<15}"
              f"  t={ev['start_time_s']:.1f}-{ev['end_time_s']:.1f}s"
              f"  → {clip_fname}")

        ok = extract_clip(cfg["video"], clip_start, clip_end, clip_local, dry_run=dry_run)
        if not ok and not dry_run:
            print(f"       [SKIP] clip extraction failed")
            skipped += 1; continue

        # Telemetry: find IMU rows near the event mid time
        # IMU video_time_s matches evasive_actions video_time_s
        mid_t = ev["mid_time_s"]
        window_imu = [r for r in imu_rows
                      if abs(float(r.get("video_time_s","0") or 0) - mid_t) <= 3.0]
        tel = {
            "ay":   round(sum(float(r.get("lay_bp","0") or 0) for r in window_imu)/max(1,len(window_imu)), 4),
            "ax":   round(sum(float(r.get("lax_bp","0") or 0) for r in window_imu)/max(1,len(window_imu)), 4),
            "az":   round(sum(float(r.get("laz_bp","0") or 0) for r in window_imu)/max(1,len(window_imu)), 4),
            "jerk": round(sum(float(r.get("jerk","0")   or 0) for r in window_imu)/max(1,len(window_imu)), 4),
        }

        gps = {
            "lat":       GPS_LAT,
            "lon":       GPS_LON,
            "location":  GPS_LOCATION,
            "speed_kmh": 30.0,
            "zone_type": "urban",
        }

        dataset.append({
            "event_id":      event_id,
            "rider":         cfg["name"],
            "location":      GPS_LOCATION,
            "evasive_action": ev["action"],
            "raw_action":    ev["raw_action"],
            "source":        ev["source"],
            "clip_path":     clip_ada,
            "clip_local":    clip_local,
            "mid_time_s":    mid_t,
            "start_time_s":  ev["start_time_s"],
            "end_time_s":    ev["end_time_s"],
            "notes":         ev.get("notes",""),
            "telemetry":     tel,
            "gps":           gps,
        })
        event_id += 1

    out_json = os.path.join(out_dir, f"{cfg['name']}_dataset.json")
    if not dry_run:
        with open(out_json, "w") as f:
            json.dump(dataset, f, indent=2)

    print(f"\nDataset: {len(dataset)} events  (skipped: {skipped})")
    print(f"JSON   → {out_json}")
    print(f"Clips  → {clips_dir}")
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rider", default="ALL", help="R1|R2|R3|R4|ALL")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    keys = list(RIDERS.keys()) if args.rider == "ALL" else [args.rider.upper()]
    for k in keys:
        if k not in RIDERS:
            print(f"Unknown rider: {k}"); continue
        build_rider(k, dry_run=args.dry_run)

    print("\nDone.")
