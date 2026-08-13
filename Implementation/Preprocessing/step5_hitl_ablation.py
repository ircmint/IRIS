# step5_hitl_ablation.py
# ---------------------------------------------------------------------------
# Computes the HITL-validation table directly from your reviewed
# candidate_events.csv (decision + adjusted_start/adjusted_end filled in)
# and gold_labels_v1_FROZEN.csv.
#
# What's REAL here:
#   - "Threshold labels, no HITL check" row: raw auto-detected candidates
#     (start_time/end_time) scored against gold (confirmed+adjusted) events.
#   - "+ Single annotator" row: the confirmed events themselves scored
#     against gold (trivially near-perfect since gold IS the confirmed set --
#     see note printed below) and the real mean |adjusted - detected| offset,
#     which is your genuine video-vs-IMU correction.
#   - Offset sub-table before/after: "after" applies a single global
#     constant-offset correction (median signed offset across all confirmed
#     events, treated as an estimated clock skew) and re-measures how many
#     events still exceed 0.5s/1s error after that correction.
#
# What's NOT computable without more data (left as NA, clearly labeled):
#   - Dual-annotator rows + kappa: need config.SECOND_ANNOTATOR_CSV, a second
#     independently-reviewed copy of candidate_events.csv.
#   - FP-cause feedback-loop row: requires actually re-running threshold
#     ablation after manually categorizing false-positive causes -- this is
#     an iterative step, not a single automated pass. Do it as a real Day-2
#     step: categorize FPs from step4 output, adjust k/fusion rules, rerun
#     step4, and fill this row in by hand from the new numbers.
#
# Run:  python step5_hitl_ablation.py
# ---------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import config
import pipeline_utils as pu


def load_reviewed():
    if not os.path.exists(config.CANDIDATE_EVENTS_CSV):
        raise FileNotFoundError(f"{config.CANDIDATE_EVENTS_CSV} not found. Run step1 first.")
    df = pd.read_csv(config.CANDIDATE_EVENTS_CSV)
    df["decision"] = df["decision"].astype(str).str.strip().str.lower()
    return df


def get_adjusted(row):
    s = row["adjusted_start"]
    e = row["adjusted_end"]
    s = float(s) if pd.notna(s) and str(s).strip() != "" else float(row["start_time"])
    e = float(e) if pd.notna(e) and str(e).strip() != "" else float(row["end_time"])
    return s, e


def main():
    df = load_reviewed()
    confirmed = df[df["decision"] == "confirm"].copy()
    if confirmed.empty:
        raise ValueError("No confirmed rows in candidate_events.csv -- finish your review first.")

    gold = pu.load_gold_labels()

    # ---- raw (pre-HITL) detected events, grouped by clip ----
    raw_pred_by_clip = {}
    for clip_id, grp in df.groupby("clip_id"):
        raw_pred_by_clip[str(clip_id)] = list(zip(grp["start_time"].astype(float),
                                                    grp["end_time"].astype(float)))
    no_hitl_scores = pu.score_events_multi_clip(raw_pred_by_clip, gold)

    # ---- offsets: |adjusted - detected| per confirmed event ----
    offsets = []
    for _, row in confirmed.iterrows():
        adj_s, adj_e = get_adjusted(row)
        offsets.append(adj_s - float(row["start_time"]))
    offsets = np.array(offsets, dtype=float)
    mean_offset_uncorrected = float(np.mean(np.abs(offsets))) if len(offsets) else float("nan")

    # single-annotator "post-HITL" events = confirmed set with adjusted times
    single_pred_by_clip = {}
    for clip_id, grp in confirmed.groupby("clip_id"):
        evs = [get_adjusted(r) for _, r in grp.iterrows()]
        single_pred_by_clip[str(clip_id)] = evs
    single_scores = pu.score_events_multi_clip(single_pred_by_clip, gold)
    mean_offset_corrected = 0.0  # by construction: gold WAS built from these adjusted times

    # ---- dual annotator (only if a second file is configured) ----
    dual_scores = None
    kappa = None
    dual_mean_offset = None
    if config.SECOND_ANNOTATOR_CSV and os.path.exists(config.SECOND_ANNOTATOR_CSV):
        df2 = pd.read_csv(config.SECOND_ANNOTATOR_CSV)
        df2["decision"] = df2["decision"].astype(str).str.strip().str.lower()
        # Cohen's kappa on confirm/reject agreement, aligned by row order
        # (candidate_events.csv rows must be identical/same order in both files)
        d1 = df["decision"].reindex(range(len(df))).values
        d2 = df2["decision"].reindex(range(len(df2))).values
        n = min(len(d1), len(d2))
        d1, d2 = d1[:n], d2[:n]
        po = np.mean(d1 == d2)
        cats = sorted(set(d1) | set(d2))
        pe = sum((np.mean(d1 == c)) * (np.mean(d2 == c)) for c in cats)
        kappa = (po - pe) / (1 - pe) if pe != 1 else float("nan")

        confirmed2 = df2[df2["decision"] == "confirm"].copy()
        dual_pred_by_clip = {}
        for clip_id, grp in confirmed2.groupby("clip_id"):
            evs = [get_adjusted(r) for _, r in grp.iterrows()]
            dual_pred_by_clip[str(clip_id)] = evs
        dual_scores = pu.score_events_multi_clip(dual_pred_by_clip, gold)

        offsets2 = []
        for _, row in confirmed2.iterrows():
            adj_s, _ = get_adjusted(row)
            offsets2.append(adj_s - float(row["start_time"]))
        dual_mean_offset = float(np.mean(np.abs(offsets + np.array(offsets2[:len(offsets)]))) / 2) \
            if len(offsets2) else float("nan")

    # ---- main HITL table ----
    rows = [
        {
            "Config": "Threshold labels, no HITL check",
            "Precision": round(no_hitl_scores["precision"], 3) if no_hitl_scores["precision"] == no_hitl_scores["precision"] else "NA",
            "Recall": round(no_hitl_scores["recall"], 3) if no_hitl_scores["recall"] == no_hitl_scores["recall"] else "NA",
            "F1 (tIoU@0.5)": round(no_hitl_scores["f1"], 3) if no_hitl_scores["f1"] == no_hitl_scores["f1"] else "NA",
            "kappa (agreement)": "—",
            "Mean Video-IMU Offset (s)": round(mean_offset_uncorrected, 3),
        },
        {
            "Config": "+ Single annotator, video-timestamp cross-check",
            "Precision": round(single_scores["precision"], 3) if single_scores["precision"] == single_scores["precision"] else "NA",
            "Recall": round(single_scores["recall"], 3) if single_scores["recall"] == single_scores["recall"] else "NA",
            "F1 (tIoU@0.5)": round(single_scores["f1"], 3) if single_scores["f1"] == single_scores["f1"] else "NA",
            "kappa (agreement)": "—",
            "Mean Video-IMU Offset (s)": round(mean_offset_corrected, 3),
        },
    ]
    if dual_scores is not None:
        rows.append({
            "Config": "+ Dual annotator, video-timestamp cross-check",
            "Precision": round(dual_scores["precision"], 3) if dual_scores["precision"] == dual_scores["precision"] else "NA",
            "Recall": round(dual_scores["recall"], 3) if dual_scores["recall"] == dual_scores["recall"] else "NA",
            "F1 (tIoU@0.5)": round(dual_scores["f1"], 3) if dual_scores["f1"] == dual_scores["f1"] else "NA",
            "kappa (agreement)": round(kappa, 3) if kappa == kappa else "NA",
            "Mean Video-IMU Offset (s)": round(dual_mean_offset, 3) if dual_mean_offset == dual_mean_offset else "NA",
        })
    else:
        rows.append({
            "Config": "+ Dual annotator, video-timestamp cross-check",
            "Precision": "NA", "Recall": "NA", "F1 (tIoU@0.5)": "NA",
            "kappa (agreement)": "NA (no second annotator file configured)",
            "Mean Video-IMU Offset (s)": "NA",
        })
    rows.append({
        "Config": "+ Dual annotator + FP-cause feedback loop into thresholding",
        "Precision": "NA", "Recall": "NA", "F1 (tIoU@0.5)": "NA",
        "kappa (agreement)": "NA (requires manual iteration, see script docstring)",
        "Mean Video-IMU Offset (s)": "NA",
    })
    main_df = pd.DataFrame(rows)

    # ---- offset sub-table: before vs after global constant-offset correction ----
    pct_gt_1s_before = float(np.mean(np.abs(offsets) > 1.0) * 100) if len(offsets) else float("nan")
    pct_gt_05s_before = float(np.mean(np.abs(offsets) > 0.5) * 100) if len(offsets) else float("nan")
    mean_before = float(np.mean(np.abs(offsets))) if len(offsets) else float("nan")

    if len(offsets):
        skew_est = float(np.median(offsets))  # estimated constant clock skew
        corrected = offsets - skew_est
        pct_gt_1s_after = float(np.mean(np.abs(corrected) > 1.0) * 100)
        pct_gt_05s_after = float(np.mean(np.abs(corrected) > 0.5) * 100)
        mean_after = float(np.mean(np.abs(corrected)))
    else:
        pct_gt_1s_after = pct_gt_05s_after = mean_after = float("nan")

    offset_df = pd.DataFrame([
        {"Config": "Before timestamp calibration", "% Offset > 1s": round(pct_gt_1s_before, 1),
         "% Offset > 0.5s": round(pct_gt_05s_before, 1), "Mean Offset (s)": round(mean_before, 3)},
        {"Config": "After clock-offset correction (median-skew)", "% Offset > 1s": round(pct_gt_1s_after, 1),
         "% Offset > 0.5s": round(pct_gt_05s_after, 1), "Mean Offset (s)": round(mean_after, 3)},
    ])

    os.makedirs(config.OUT_DIR, exist_ok=True)
    main_df.to_csv(config.HITL_ABLATION_OUT, index=False)
    offset_df.to_csv(config.HITL_OFFSET_OUT, index=False)

    print("\n=== HITL validation ablation ===\n")
    print(main_df.to_string(index=False))
    print("\n--- Video-IMU offset calibration ---\n")
    print(offset_df.to_string(index=False))
    print(f"\nSaved:\n  {config.HITL_ABLATION_OUT}\n  {config.HITL_OFFSET_OUT}")
    print("\nIMPORTANT CAVEATS to carry into your writeup:")
    print(" - 'Single annotator' precision/recall are near-1.0 by construction, since gold")
    print("   labels ARE the confirmed set from this same review pass. That row is really")
    print("   demonstrating internal consistency, not held-out accuracy -- say so explicitly,")
    print("   or hold out a few clips as a second, independent check if you have time.")
    print(" - Dual-annotator / kappa / feedback-loop rows need real additional data collection")
    print("   (see script docstring) -- don't fill placeholder numbers into those cells.")
    print(f" - N is small ({len(confirmed)} confirmed events) -- report this as a pilot,")
    print("   not a statistically powered result.")


if __name__ == "__main__":
    main()
