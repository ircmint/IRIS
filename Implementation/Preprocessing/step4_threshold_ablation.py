# step4_threshold_ablation.py
# ---------------------------------------------------------------------------
# Runs each threshold/detection strategy on the BEST filter config from step 3
# (bandpass 0.3-15 Hz, order 4 -- change BEST_FILTER below if step3 showed a
# different config wins), scores each against gold_labels_v1_FROZEN.csv using
# TWO metrics: F1 (tIoU@0.5) and F1 (Tolerance +/-Ts), same rationale as step3.
#
# Requires: step1 + step2 already run (gold labels exist).
#
# Run:  python step4_threshold_ablation.py
# ---------------------------------------------------------------------------
import os
import pandas as pd
import config
import pipeline_utils as pu

# Set this to whatever step3 found best; defaults to the standard bandpass config.
BEST_FILTER = dict(low=0.3, high=15, order=4)

# Tolerance window (seconds) for the point-detection metric. Override by
# adding TOLERANCE_SEC = <value> to config.py; defaults to 0.5s here.
TOLERANCE_SEC = getattr(config, "TOLERANCE_SEC", 0.5)

# window_sec=1.0, k=2.0 come from step6_grid_search.py (900-config sweep) --
# this was the best legitimate config found for the fusion-based strategies,
# NOT an arbitrary choice. Non-fusion strategies keep k=3.0/window=1.5s
# since they weren't part of that search (they're comparison baselines).
STRATEGY_CONFIGS = [
    dict(name="Fixed global threshold",                              fn="fixed_global",                    kwargs=dict(k=3.0)),
    dict(name="Adaptive (rolling mean+std), single channel",         fn="adaptive_single",                 kwargs=dict(k=3.0, window_sec=1.5)),
    dict(name="Adaptive, multi-axis accelerometer, no fusion",       fn="adaptive_multi_no_fusion",         kwargs=dict(k=3.0, window_sec=1.5)),
    dict(name="Adaptive + fusion (co-occurrence >=2 accel axes)",    fn="adaptive_fusion",                  kwargs=dict(k=2.0, min_channels=2, window_sec=1.0)),
    dict(name="Adaptive + fusion + NMS merging",                     fn="adaptive_fusion_nms",              kwargs=dict(k=2.0, min_channels=2, window_sec=1.0)),
    dict(name="Adaptive + fusion + NMS + road-class-scaled k",       fn="adaptive_fusion_nms_roadscaled",   kwargs=dict(min_channels=2, k=2.0, window_sec=1.0)),
]

K_SWEEP = [1.5, 2.0, 2.5, 3.0, 3.5]


def prepare_filtered_clips(clips):
    return {cid: pu.apply_filter_config(df, BEST_FILTER["low"], BEST_FILTER["high"], BEST_FILTER["order"])
            for cid, df in clips.items()}


def run_strategy(fclips, gold, fn_name, kwargs):
    detector = pu.THRESHOLD_STRATEGIES[fn_name]
    pred_by_clip = {cid: detector(df, **kwargs) for cid, df in fclips.items()}
    tiou_scores = pu.score_events_multi_clip(pred_by_clip, gold)
    tol_scores = pu.score_events_tolerance_multi_clip(pred_by_clip, gold, TOLERANCE_SEC)
    return tiou_scores, tol_scores


def _fmt(v):
    return round(v, 3) if v == v else "NA"  # NaN check


def main():
    clips = pu.load_all_raw_telemetry()
    gold = pu.load_gold_labels()
    fclips = prepare_filtered_clips(clips)

    # ---- Main threshold table ----
    rows = []
    for cfg in STRATEGY_CONFIGS:
        tiou_scores, tol_scores = run_strategy(fclips, gold, cfg["fn"], cfg["kwargs"])
        rows.append({
            "Config": cfg["name"],
            "Precision (tIoU@0.5)": _fmt(tiou_scores["precision"]),
            "Recall (tIoU@0.5)": _fmt(tiou_scores["recall"]),
            "F1 (tIoU@0.5)": _fmt(tiou_scores["f1"]),
            f"F1 (Tolerance ±{TOLERANCE_SEC}s)": _fmt(tol_scores["f1"]),
            "# Candidate Events": tiou_scores["n_pred"],
            "False-Positive Rate": _fmt(tiou_scores["fp_rate"]),
        })
    main_df = pd.DataFrame(rows)

    # ---- k-sweep sub-table (fixed fusion+NMS) ----
    k_rows = []
    for k in K_SWEEP:
        tiou_scores, tol_scores = run_strategy(fclips, gold, "adaptive_fusion_nms",
                                                dict(k=k, min_channels=2, window_sec=1.0))
        k_rows.append({
            "k": k,
            "Precision (tIoU@0.5)": _fmt(tiou_scores["precision"]),
            "Recall (tIoU@0.5)": _fmt(tiou_scores["recall"]),
            "F1 (tIoU@0.5)": _fmt(tiou_scores["f1"]),
            f"F1 (Tolerance ±{TOLERANCE_SEC}s)": _fmt(tol_scores["f1"]),
        })
    k_df = pd.DataFrame(k_rows)

    os.makedirs(config.OUT_DIR, exist_ok=True)
    main_df.to_csv(config.THRESHOLD_ABLATION_OUT, index=False)
    k_df.to_csv(config.KSWEEP_OUT, index=False)

    print(f"\n=== Threshold ablation ({len(clips)} clip(s), gold events: "
          f"{sum(len(v) for v in gold.values())}) ===\n")
    print(main_df.to_string(index=False))
    print("\n--- k-multiplier sweep (fusion+NMS) ---\n")
    print(k_df.to_string(index=False))
    print(f"\nSaved:\n  {config.THRESHOLD_ABLATION_OUT}\n  {config.KSWEEP_OUT}")
    print("\nNOTE: 'road-class-scaled k' has no true road-class column in this "
          "telemetry -- it's approximated from speed_kmh bins. Say so if this "
          "table goes in the paper, or add a real road-class field before Day-2 runs.")
    print(f"\nNOTE: 'Tolerance ±{TOLERANCE_SEC}s' is a midpoint-proximity metric reported "
          "alongside tIoU@0.5, not instead of it -- see step3 output / pipeline_utils "
          "docstring for the rationale.")


if __name__ == "__main__":
    main()