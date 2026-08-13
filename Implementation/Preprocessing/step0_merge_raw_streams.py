# step0_merge_raw_streams.py
# ---------------------------------------------------------------------------
# Your raw exports come as 3 separate files per clip (accel, gyro, gps), with
# cts in MILLISECONDS and unlabeled axis columns ("1", "2"). This script:
#   1. Loads and renames columns properly
#   2. Converts cts from ms -> seconds (the unit steps 1-5 assume)
#   3. Merges GPS (lower rate, ~10 Hz) onto the IMU timeline (~200 Hz)
#   4. Bandpass-filters accel, computes jerk, total_jerk, rolling z-scores
#   5. Runs a PROVISIONAL fusion detector to produce a first-pass evasive_flag
#      -- this is a Day-1 default, not tuned. You will review/correct these
#      in step 1, and step4_threshold_ablation.py later tests real alternatives.
#   6. Writes E:\evasive_project\raw_telemetry\{clip_id}.csv, ready for step1.
#
# Run:  python step0_merge_raw_streams.py
# ---------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import config
import pipeline_utils as pu


def load_accel(path):
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Accelerometer [m/s²]": "ax_raw",
        "1": "ay_raw",
        "2": "az_raw",
    })
    return df[["cts", "date", "ax_raw", "ay_raw", "az_raw"]]


def load_gyro(path):
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Gyroscope [rad/s]": "gx_raw",
        "1": "gy_raw",
        "2": "gz_raw",
    })
    return df[["cts", "gx_raw", "gy_raw", "gz_raw"]]


def load_gps(path):
    df = pd.read_csv(path)
    df = df.rename(columns={
        "GPS (Lat.) [deg]": "lat",
        "GPS (Long.) [deg]": "lon",
        "GPS (2D speed) [m/s]": "speed_ms",
    })
    df["speed_kmh"] = df["speed_ms"] * 3.6
    return df[["cts", "lat", "lon", "speed_kmh"]]


def format_video_ts(seconds_from_start):
    m = int(seconds_from_start // 60)
    s = seconds_from_start - m * 60
    return f"{m:02d}:{s:04.1f}"


def merge_one_clip(clip_id, paths):
    accel = load_accel(paths["accel"])
    gyro = load_gyro(paths["gyro"])
    gps = load_gps(paths["gps"])

    if not np.allclose(accel["cts"].values, gyro["cts"].values):
        raise ValueError(
            f"{clip_id}: accel and gyro cts timelines don't match exactly. "
            f"They're normally identical for GoPro IMU exports -- check these "
            f"are really the same recording."
        )

    df = accel.copy()
    df["gx_raw"] = gyro["gx_raw"].values
    df["gy_raw"] = gyro["gy_raw"].values
    df["gz_raw"] = gyro["gz_raw"].values

    # cts arrives in milliseconds -> convert to seconds (what steps 1-5 assume)
    df["cts"] = df["cts"] / 1000.0
    gps = gps.copy()
    gps["cts"] = gps["cts"] / 1000.0

    df = df.sort_values("cts").reset_index(drop=True)
    gps = gps.sort_values("cts").reset_index(drop=True)

    # merge_asof: attach nearest GPS fix within tolerance to each IMU sample
    merged = pd.merge_asof(
        df, gps, on="cts", direction="nearest",
        tolerance=config.GPS_MERGE_TOLERANCE_SEC,
    )
    merged["lat"] = merged["lat"].ffill().bfill()
    merged["lon"] = merged["lon"].ffill().bfill()
    merged["speed_kmh"] = merged["speed_kmh"].ffill().bfill()

    t0 = merged["cts"].iloc[0]
    merged["video_ts"] = (merged["cts"] - t0).apply(format_video_ts)

    # ---- filtering + jerk (provisional Day-1 config, see config.STEP0_FILTER) ----
    fs = pu.estimate_sample_rate(merged)
    low, high, order = config.STEP0_FILTER["low"], config.STEP0_FILTER["high"], config.STEP0_FILTER["order"]
    for axis in ["ax", "ay", "az"]:
        merged[f"{axis}_bpf"] = pu.butter_filter(merged[f"{axis}_raw"].values, fs, low, high, order)

    dt = 1.0 / fs
    merged["jerk_X"] = np.gradient(merged["ax_bpf"].values, dt)
    merged["jerk_Y"] = np.gradient(merged["ay_bpf"].values, dt)
    merged["jerk_Z"] = np.gradient(merged["az_bpf"].values, dt)
    merged["total_jerk"] = np.sqrt(merged["jerk_X"] ** 2 + merged["jerk_Y"] ** 2 + merged["jerk_Z"] ** 2)

    win = max(3, int(config.STEP0_DETECT_WINDOW_SEC * fs))
    roll_mean_az = merged["az_bpf"].rolling(win, min_periods=1, center=True).mean()
    roll_std_az = merged["az_bpf"].rolling(win, min_periods=1, center=True).std().replace(0, np.nan)
    merged["z_az"] = ((merged["az_bpf"] - roll_mean_az) / roll_std_az).fillna(0)

    roll_mean_j = merged["total_jerk"].rolling(win, min_periods=1, center=True).mean()
    roll_std_j = merged["total_jerk"].rolling(win, min_periods=1, center=True).std().replace(0, np.nan)
    merged["z_jerk"] = ((merged["total_jerk"] - roll_mean_j) / roll_std_j).fillna(0)

    merged["is_moving"] = merged["speed_kmh"] > config.MOVING_SPEED_THRESHOLD_KMH

    # ---- provisional evasive_flag: co-occurrence across jerk axes ----
    votes = pd.Series(0, index=merged.index)
    for c in ["jerk_X", "jerk_Y", "jerk_Z"]:
        rm = merged[c].rolling(win, min_periods=1, center=True).mean()
        rs = merged[c].rolling(win, min_periods=1, center=True).std().replace(0, np.nan)
        z = ((merged[c] - rm) / rs).fillna(0)
        votes = votes + (z.abs() > config.STEP0_DETECT_K).astype(int)
    merged["evasive_flag"] = (votes >= config.STEP0_MIN_FUSION_CHANNELS).astype(int)
    merged["evasive_type"] = np.where(merged["evasive_flag"] == 1, "candidate", "None")
    merged["confidence"] = np.clip(merged["z_jerk"].abs() / 6.0, 0, 1).where(merged["evasive_flag"] == 1, 0.0)

    out = merged[[
        "cts", "date", "video_ts", "ax_raw", "ay_raw", "az_raw",
        "ax_bpf", "ay_bpf", "az_bpf", "jerk_X", "jerk_Y", "jerk_Z", "total_jerk",
        "z_az", "z_jerk", "lat", "lon", "speed_kmh", "is_moving",
        "evasive_flag", "evasive_type", "confidence",
    ]]
    return out


def main():
    os.makedirs(config.RAW_TELEMETRY_DIR, exist_ok=True)
    for clip_id, paths in config.RAW_STREAM_CLIPS.items():
        missing = [k for k, p in paths.items() if not os.path.exists(p)]
        if missing:
            print(f"[skip] {clip_id}: missing/unreadable file(s) for {missing} -- "
                  f"check the paths in config.RAW_STREAM_CLIPS.")
            continue
        merged = merge_one_clip(clip_id, paths)
        out_path = os.path.join(config.RAW_TELEMETRY_DIR, f"{clip_id}.csv")
        merged.to_csv(out_path, index=False)
        n_events = int(merged["evasive_flag"].sum())
        print(f"{clip_id}: {len(merged)} samples, {merged['cts'].iloc[-1]-merged['cts'].iloc[0]:.1f}s, "
              f"{n_events} provisional flagged samples -> {out_path}")

    print("\nNext: python step1_build_candidates.py")
    print("(This will group the provisional evasive_flag into candidate events for your review.)")


if __name__ == "__main__":
    main()
