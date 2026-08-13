# pipeline_utils.py
# ---------------------------------------------------------------------------
# Shared functions: loading telemetry, filtering, jerk computation, event
# detection strategies, and tIoU-based scoring against gold labels.
# No hardcoded paths live here -- see config.py.
# ---------------------------------------------------------------------------
import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

import config


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_telemetry_file(path):
    """Load one raw telemetry file, auto-detecting comma vs tab delimiter."""
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    required = ["cts", "ax_raw", "ay_raw", "az_raw"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}. "
                          f"Found columns: {list(df.columns)}")
    df = df.sort_values(config.TIME_COL).reset_index(drop=True)
    return df


def load_all_raw_telemetry(raw_dir=None):
    """Returns dict {clip_id: dataframe} for every CSV in raw_dir.

    NOTE: this pipeline is accelerometer-only. load_telemetry_file requires
    ax_raw/ay_raw/az_raw and never reads gyroscope columns -- any table row
    or config named "accel+gyro" is a labeling error unless gyro columns
    are explicitly added to `required` in load_telemetry_file() and wired
    into apply_filter_config()/the detectors below. Do not rename a row to
    say "gyro" without actually adding gyro data to the computation -- that
    would misrepresent what was measured.
    """
    raw_dir = raw_dir or config.RAW_TELEMETRY_DIR
    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"No .csv files found in {raw_dir}. "
            f"Put your per-clip raw telemetry CSVs there first."
        )
    out = {}
    for f in files:
        clip_id = os.path.splitext(os.path.basename(f))[0]
        out[clip_id] = load_telemetry_file(f)
    return out


def load_gold_labels(path=None):
    """Returns dict {clip_id: [(start,end), ...]}"""
    path = path or config.GOLD_LABELS_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Gold labels file not found at {path}.\n"
            f"Run step2_build_gold.py after you've reviewed candidate_events.csv."
        )
    df = pd.read_csv(path)
    required = ["clip_id", "start_time", "end_time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    gold = {}
    for clip_id, grp in df.groupby("clip_id"):
        gold[str(clip_id)] = list(zip(grp["start_time"].astype(float),
                                       grp["end_time"].astype(float)))
    return gold


# ---------------------------------------------------------------------------
# Sample-rate estimation
# ---------------------------------------------------------------------------
def estimate_sample_rate(df):
    if config.FORCE_SAMPLE_RATE_HZ:
        return config.FORCE_SAMPLE_RATE_HZ
    t = df[config.TIME_COL].values
    dt = np.median(np.diff(t))
    if dt <= 0:
        raise ValueError("Non-increasing timestamps; check your cts column / file sort order.")
    return 1.0 / dt


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def butter_filter(signal, fs, low=None, high=None, order=4):
    """
    Zero-phase Butterworth filter via sosfiltfilt (second-order-sections form).
    SOS is used instead of the transfer-function (b,a) form because ba-form
    Butterworth filters above ~order 4 become numerically unstable (their
    coefficients can produce all-NaN output even though the SOS-equivalent
    filter is perfectly well-behaved) -- this bit real order-6/8 runs before.
    low=None,high=X  -> lowpass at X
    low=X,high=None  -> highpass at X
    low=X,high=Y     -> bandpass X-Y
    low=None,high=None -> passthrough (no filtering, i.e. "raw" config)
    """
    nyq = fs / 2.0
    if low is None and high is None:
        return signal.copy()

    def clamp(f):
        safe = min(f, nyq * 0.98)
        if safe != f:
            print(f"  [warn] cutoff {f} Hz >= Nyquist ({nyq:.2f} Hz at fs={fs:.2f} Hz); "
                  f"clamped to {safe:.3f} Hz. Your real sample rate may be higher than this "
                  f"test data's -- check FORCE_SAMPLE_RATE_HZ / estimated fs if this looks wrong.")
        return safe

    if low is not None and high is not None:
        sos = butter(order, [clamp(low) / nyq, clamp(high) / nyq], btype="band", output="sos")
    elif high is not None:
        sos = butter(order, clamp(high) / nyq, btype="low", output="sos")
    else:
        sos = butter(order, clamp(low) / nyq, btype="high", output="sos")

    min_len = 3 * (2 * len(sos) + 1)  # sosfiltfilt's minimum signal length requirement
    if len(signal) <= min_len:
        return signal.copy()
    out = sosfiltfilt(sos, signal)

    # Integrity guard: a correctly-designed SOS filter should never emit
    # NaN/Inf. If it does, that's a genuine numerical bug (e.g. unstable
    # coefficients), not legitimate signal content -- fail loudly instead of
    # silently returning bad data. This does NOT flag negative values --
    # acceleration/jerk are signed quantities and negative values are normal,
    # physically meaningful signal content, not an error condition.
    n_bad = int(np.sum(~np.isfinite(out)))
    if n_bad > 0:
        raise FloatingPointError(
            f"butter_filter produced {n_bad} NaN/Inf sample(s) "
            f"(low={low}, high={high}, order={order}, fs={fs:.2f}Hz). "
            f"This indicates unstable filter coefficients, not valid signal "
            f"content -- do not use this config's output. Try a lower order "
            f"or check that cutoffs are well inside the Nyquist range."
        )
    return out


def apply_filter_config(df, low, high, order):
    """Returns a copy of df with ax_f, ay_f, az_f, jerk_x/y/z, total_jerk, z_jerk added,
    using the given filter config applied to the RAW axes."""
    fs = estimate_sample_rate(df)
    out = df.copy()
    for axis in ["ax", "ay", "az"]:
        raw_col = f"{axis}_raw"
        out[f"{axis}_f"] = butter_filter(out[raw_col].values, fs, low, high, order)

    dt = 1.0 / fs
    for axis in ["ax", "ay", "az"]:
        out[f"jerk_{axis}"] = np.gradient(out[f"{axis}_f"].values, dt)

    out["total_jerk"] = np.sqrt(
        out["jerk_ax"] ** 2 + out["jerk_ay"] ** 2 + out["jerk_az"] ** 2
    )
    mu, sigma = out["total_jerk"].mean(), out["total_jerk"].std()
    out["z_jerk"] = (out["total_jerk"] - mu) / sigma if sigma > 0 else 0.0
    return out


def snr_improvement_db(df_raw_col, df_filtered_col):
    """
    Rough SNR-improvement proxy: ratio of variance reduction in the filtered
    signal vs raw signal, expressed in dB. This is a relative, in-sample proxy
    (no separate noise-only reference channel exists in this telemetry), so
    report it as such in the paper -- do not present as a calibrated SNR meter.
    """
    raw = np.asarray(df_raw_col)
    filt = np.asarray(df_filtered_col)
    var_raw = np.var(raw)
    var_filt = np.var(filt)
    if var_raw <= 0 or var_filt <= 0:
        return float("nan")
    # Convention: a filter that reduces high-freq noise variance while
    # preserving event-band energy shows as a variance reduction here.
    return 10 * np.log10(var_raw / var_filt)


# ---------------------------------------------------------------------------
# Event detection strategies (grouping flagged samples into intervals)
# ---------------------------------------------------------------------------
def flags_to_events(df, flag_col, time_col=None, gap_tol=None):
    """Collapse consecutive flag==True/1 rows into (start,end) events,
    merging events separated by <= gap_tol seconds."""
    time_col = time_col or config.TIME_COL
    gap_tol = config.GAP_TOL if gap_tol is None else gap_tol

    flagged = df[df[flag_col].astype(bool)]
    if flagged.empty:
        return []

    times = flagged[time_col].values
    events = []
    start = times[0]
    prev = times[0]
    for t in times[1:]:
        if t - prev > gap_tol:
            events.append((start, prev))
            start = t
        prev = t
    events.append((start, prev))
    return events


def detect_fixed_global(df, z_col="z_jerk", k=3.0):
    flag = df[z_col].abs() > k
    return flags_to_events(df.assign(_flag=flag), "_flag")


def detect_adaptive_single(df, z_col="total_jerk", window_sec=1.5, k=3.0):
    fs = estimate_sample_rate(df)
    window = max(5, int(window_sec * fs))
    roll_mean = df[z_col].rolling(window, min_periods=1, center=True).mean()
    roll_std = df[z_col].rolling(window, min_periods=1, center=True).std().replace(0, np.nan)
    z = (df[z_col] - roll_mean) / roll_std
    flag = z.abs() > k
    flag = flag.fillna(False)
    return flags_to_events(df.assign(_flag=flag), "_flag")


def detect_adaptive_multi_no_fusion(df, cols=("jerk_ax", "jerk_ay", "jerk_az"),
                                     window_sec=1.5, k=3.0):
    """Flag if ANY channel exceeds threshold (no co-occurrence requirement)."""
    fs = estimate_sample_rate(df)
    window = max(5, int(window_sec * fs))
    flag_any = pd.Series(False, index=df.index)
    for c in cols:
        roll_mean = df[c].rolling(window, min_periods=1, center=True).mean()
        roll_std = df[c].rolling(window, min_periods=1, center=True).std().replace(0, np.nan)
        z = (df[c] - roll_mean) / roll_std
        flag_any = flag_any | (z.abs() > k).fillna(False)
    return flags_to_events(df.assign(_flag=flag_any), "_flag")


def detect_adaptive_fusion(df, cols=("jerk_ax", "jerk_ay", "jerk_az"),
                            window_sec=1.5, k=3.0, min_channels=2):
    """Flag only if >= min_channels exceed threshold simultaneously (co-occurrence)."""
    fs = estimate_sample_rate(df)
    window = max(5, int(window_sec * fs))
    votes = pd.Series(0, index=df.index)
    for c in cols:
        roll_mean = df[c].rolling(window, min_periods=1, center=True).mean()
        roll_std = df[c].rolling(window, min_periods=1, center=True).std().replace(0, np.nan)
        z = (df[c] - roll_mean) / roll_std
        votes = votes + (z.abs() > k).fillna(False).astype(int)
    flag = votes >= min_channels
    return flags_to_events(df.assign(_flag=flag), "_flag")


def merge_nms(events, iou_thresh=0.3):
    """Simple NMS-style merge: sort by start time, merge overlapping/adjacent
    events instead of allowing near-duplicates."""
    if not events:
        return events
    events = sorted(events, key=lambda e: e[0])
    merged = [events[0]]
    for s, e in events[1:]:
        ls, le = merged[-1]
        if s <= le:  # overlap -> merge
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def detect_adaptive_fusion_nms(df, **kwargs):
    events = detect_adaptive_fusion(df, **kwargs)
    return merge_nms(events)


def detect_adaptive_fusion_nms_roadscaled(df, speed_col="speed_kmh", **kwargs):
    """
    Proxy for 'road-class-scaled k': this telemetry has no road-class label,
    so k is scaled by a speed-bin proxy (low/med/high speed -> looser/tighter
    threshold). Documented as an approximation, not a true road-class signal.
    """
    base_k = kwargs.pop("k", 3.0)
    speed = df[speed_col] if speed_col in df.columns else pd.Series(0, index=df.index)
    # crude 3-bin scaling: low speed -> more sensitive (lower k), high speed -> stricter
    k_series = pd.cut(speed, bins=[-1, 20, 60, 1e9],
                       labels=[base_k * 0.8, base_k * 1.0, base_k * 1.3]).astype(float)
    # detect_adaptive_fusion doesn't support per-row k natively; approximate by
    # running once with the median effective k across the clip.
    eff_k = float(k_series.median()) if k_series.notna().any() else base_k
    events = detect_adaptive_fusion(df, k=eff_k, **kwargs)
    return merge_nms(events)


THRESHOLD_STRATEGIES = {
    "fixed_global": detect_fixed_global,
    "adaptive_single": detect_adaptive_single,
    "adaptive_multi_no_fusion": detect_adaptive_multi_no_fusion,
    "adaptive_fusion": detect_adaptive_fusion,
    "adaptive_fusion_nms": detect_adaptive_fusion_nms,
    "adaptive_fusion_nms_roadscaled": detect_adaptive_fusion_nms_roadscaled,
}


# ---------------------------------------------------------------------------
# Scoring: tIoU-based matching of predicted events vs gold events
# ---------------------------------------------------------------------------
def tiou(a, b):
    s1, e1 = a
    s2, e2 = b
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    if union <= 0:
        return 0.0
    return inter / union


def score_events(pred_events, gold_events, thresh=None):
    """Greedy one-to-one matching by tIoU >= thresh. Returns dict with
    precision, recall, f1, tp, fp, fn."""
    thresh = config.TIOU_THRESHOLD if thresh is None else thresh
    gold_used = [False] * len(gold_events)
    tp = 0
    for p in pred_events:
        best_iou, best_idx = 0.0, -1
        for i, g in enumerate(gold_events):
            if gold_used[i]:
                continue
            iou = tiou(p, g)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= thresh and best_idx >= 0:
            gold_used[best_idx] = True
            tp += 1
    fp = len(pred_events) - tp
    fn = len(gold_events) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if np.isnan(precision) or np.isnan(recall):
        f1 = float("nan")
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return dict(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn,
                n_pred=len(pred_events), n_gold=len(gold_events))


def score_events_multi_clip(pred_by_clip, gold_by_clip, thresh=None):
    """Aggregate scoring across all clips (micro-average: pool tp/fp/fn)."""
    thresh = config.TIOU_THRESHOLD if thresh is None else thresh
    tot_tp = tot_fp = tot_fn = tot_pred = tot_gold = 0
    for clip_id, gold_events in gold_by_clip.items():
        pred_events = pred_by_clip.get(clip_id, [])
        r = score_events(pred_events, gold_events, thresh)
        tot_tp += r["tp"]; tot_fp += r["fp"]; tot_fn += r["fn"]
        tot_pred += r["n_pred"]; tot_gold += r["n_gold"]
    precision = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else float("nan")
    recall = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else float("nan")
    if np.isnan(precision) or np.isnan(recall):
        f1 = float("nan")
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    fp_rate = tot_fp / tot_pred if tot_pred > 0 else float("nan")
    return dict(precision=precision, recall=recall, f1=f1,
                tp=tot_tp, fp=tot_fp, fn=tot_fn,
                n_pred=tot_pred, n_gold=tot_gold, fp_rate=fp_rate)


# ---------------------------------------------------------------------------
# Scoring (NEW): tolerance-window / point-detection matching
# ---------------------------------------------------------------------------
# Rationale: tIoU@0.5 assumes events have enough duration that a 50% overlap
# is a meaningful bar. Many of this pilot's gold events are near-instantaneous
# (single-digit sample counts), where tIoU is punishingly strict even for a
# detector that fires at essentially the right moment. Point/tolerance-window
# matching -- "does the predicted event's midpoint land within +/-N seconds
# of the true event's midpoint" -- is standard for exactly this situation in
# point-detection literature (e.g. seizure/anomaly onset spotting), and is
# reported here as a SECOND metric alongside tIoU@0.5, not a replacement.
#
# Midpoint (not start-time) is used as each event's representative instant,
# so that a detector whose window is systematically wider/narrower than gold
# -- but centered on the same physical event -- isn't penalized for duration
# mismatch alone.
def _midpoint(ev):
    s, e = ev
    return (s + e) / 2.0


def score_events_tolerance(pred_events, gold_events, tolerance_sec=None):
    """Greedy one-to-one matching by |pred_midpoint - gold_midpoint| <= tolerance_sec."""
    tolerance_sec = getattr(config, "TOLERANCE_SEC", 0.5) if tolerance_sec is None else tolerance_sec
    gold_used = [False] * len(gold_events)
    tp = 0
    for p in pred_events:
        p_mid = _midpoint(p)
        best_dist, best_idx = float("inf"), -1
        for i, g in enumerate(gold_events):
            if gold_used[i]:
                continue
            dist = abs(p_mid - _midpoint(g))
            if dist < best_dist:
                best_dist, best_idx = dist, i
        if best_idx >= 0 and best_dist <= tolerance_sec:
            gold_used[best_idx] = True
            tp += 1
    fp = len(pred_events) - tp
    fn = len(gold_events) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if np.isnan(precision) or np.isnan(recall):
        f1 = float("nan")
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return dict(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn,
                n_pred=len(pred_events), n_gold=len(gold_events))


def score_events_tolerance_multi_clip(pred_by_clip, gold_by_clip, tolerance_sec=None):
    """Aggregate tolerance-window scoring across all clips (micro-average)."""
    tolerance_sec = getattr(config, "TOLERANCE_SEC", 0.5) if tolerance_sec is None else tolerance_sec
    tot_tp = tot_fp = tot_fn = tot_pred = tot_gold = 0
    for clip_id, gold_events in gold_by_clip.items():
        pred_events = pred_by_clip.get(clip_id, [])
        r = score_events_tolerance(pred_events, gold_events, tolerance_sec)
        tot_tp += r["tp"]; tot_fp += r["fp"]; tot_fn += r["fn"]
        tot_pred += r["n_pred"]; tot_gold += r["n_gold"]
    precision = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) > 0 else float("nan")
    recall = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) > 0 else float("nan")
    if np.isnan(precision) or np.isnan(recall):
        f1 = float("nan")
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    fp_rate = tot_fp / tot_pred if tot_pred > 0 else float("nan")
    return dict(precision=precision, recall=recall, f1=f1,
                tp=tot_tp, fp=tot_fp, fn=tot_fn,
                n_pred=tot_pred, n_gold=tot_gold, fp_rate=fp_rate)


# ---------------------------------------------------------------------------
# Event categorization (NEW): keyword-based classification of reviewed
# events by evasive-action type, using the free-text notes written during
# HITL review (typically VLM-generated captions).
#
# IMPORTANT: this is a DESCRIPTIVE/diagnostic tool, not a per-category
# detection threshold. Categories are only known AFTER an event has already
# been detected and reviewed -- you cannot apply a category-specific
# threshold at detection time because the category doesn't exist yet at
# that point in the pipeline. Use this to characterize what magnitude/
# duration ranges each category actually spans in your reviewed data, and
# to see plainly when a category has zero examples (do not fabricate a
# range for an empty category -- report it as no data, as done below).
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS = [
    # (category_name, list of substrings to match in lowercased notes text)
    # Order matters: first match wins, so put more specific phrases first.
    ("likely_noise",         ["camera vibration", "sensor noise", "no visible change"]),
    ("braking_deceleration", ["brak", "decelerat", "slowing", "slow down", "stopp"]),
    ("near_miss",            ["near miss", "near-miss", "collision", "hazard"]),
    ("emergency_swerve",     ["swerv", "avoid"]),
    ("sharp_turn",           ["sharp turn"]),
    ("acceleration",         ["accelerat"]),
    ("lane_change",          ["lane"]),
]


def categorize_event_notes(note):
    """Classify one event's free-text review note into a category using
    CATEGORY_KEYWORDS (first matching keyword wins). Returns 'other' if
    nothing matches. This is a simple substring heuristic, not an NLP
    model -- spot-check a sample of each category's notes before reporting
    category-level statistics in a paper."""
    n = str(note).lower()
    for category, keywords in CATEGORY_KEYWORDS:
        if any(kw in n for kw in keywords):
            return category
    return "other"