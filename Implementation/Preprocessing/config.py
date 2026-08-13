# config.py
# ---------------------------------------------------------------------------
# EDIT THIS FILE ONLY. Every other script imports paths/settings from here,
# so you never have to touch argparse flags or hunt through multiple files.
# ---------------------------------------------------------------------------
import os

# ---- Folders -----------------------------------------------------------
BASE_DIR = r"E:\IIITH\Pilot_2\DAY_1"

RAW_TELEMETRY_DIR = os.path.join(BASE_DIR, "raw_telemetry")   # your 1-3 raw CSVs go here, one per clip
OUT_DIR            = os.path.join(BASE_DIR, "day1_outputs")
DATA_DIR           = os.path.join(BASE_DIR, "data")

# ---- Files produced/consumed by the pipeline ----------------------------
CANDIDATE_EVENTS_CSV   = os.path.join(OUT_DIR, "candidate_events.csv")     # step 1 output -> you review this by hand
GOLD_LABELS_CSV         = os.path.join(DATA_DIR, "gold_labels_v1_FROZEN.csv")  # step 2 output (after your review)

FILTER_ABLATION_OUT    = os.path.join(OUT_DIR, "filter_ablation_results.csv")
THRESHOLD_ABLATION_OUT = os.path.join(OUT_DIR, "threshold_ablation_results.csv")
KSWEEP_OUT              = os.path.join(OUT_DIR, "threshold_ksweep_results.csv")
HITL_ABLATION_OUT      = os.path.join(OUT_DIR, "hitl_ablation_results.csv")
HITL_OFFSET_OUT         = os.path.join(OUT_DIR, "hitl_offset_results.csv")

# Optional: a second annotator's independently-reviewed copy of candidate_events.csv,
# same columns, filled in independently. Leave as None if you don't have one --
# the dual-annotator / kappa rows will just print as NA instead of being invented.
SECOND_ANNOTATOR_CSV = None   # e.g. os.path.join(OUT_DIR, "candidate_events_annotator2.csv")

# ---- Column / detection settings ----------------------------------------
TIME_COL = "cts"          # continuous seconds-float timestamp column
GAP_TOL = 1.0              # seconds: gaps <= this are merged into one event
TIOU_THRESHOLD = 0.5       # tIoU match threshold for scoring predicted vs gold events

# Sample rate is auto-estimated per file from median(diff(cts)), but if your
# telemetry has irregular sampling, set this to override the estimate.
FORCE_SAMPLE_RATE_HZ = None   # e.g. 20  -- leave None to auto-estimate

RANDOM_SEED = 0

# ---------------------------------------------------------------------------
# STEP 0 ONLY: your raw data currently comes as 3 separate GoPro-style
# streams per clip (accel / gyro / gps CSVs, cts in MILLISECONDS) instead of
# one merged CSV. step0_merge_raw_streams.py reads these, aligns them, and
# writes a merged {clip_id}.csv into RAW_TELEMETRY_DIR in the exact schema
# steps 1-5 already expect. Add one entry per clip below.
# ---------------------------------------------------------------------------
RAW_STREAM_CLIPS = {
    "GX019940": dict(
        accel=r"E:\IIITH\Pilot_2\DAY_1\accel_data.csv",
        gyro=r"E:\IIITH\Pilot_2\DAY_1\gyro_data.csv",
        gps=r"E:\IIITH\Pilot_2\DAY_1\gps_data.csv",
    ),
    # "GX019941": dict(accel=r"...", gyro=r"...", gps=r"..."),  # add more clips here
}

# Provisional filter + detector used ONLY to produce a first-pass evasive_flag
# for you to review in step 1. This is NOT the final threshold config --
# step4_threshold_ablation.py sweeps real alternatives later. Reasonable
# Day-1 defaults for ~200Hz automotive IMU data:
STEP0_FILTER = dict(low=0.3, high=15, order=4)
STEP0_DETECT_K = 2.0
STEP0_DETECT_WINDOW_SEC = 1.0     # rolling window for the provisional z-score, in seconds
STEP0_MIN_FUSION_CHANNELS = 2     # co-occurrence across jerk_X/Y/Z axes
MOVING_SPEED_THRESHOLD_KMH = 2.0  # below this, is_moving = FALSE
GPS_MERGE_TOLERANCE_SEC = 0.5     # how far a GPS fix can be from an IMU sample and still be used
