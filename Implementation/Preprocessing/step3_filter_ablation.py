# step3_filter_ablation.py
# ---------------------------------------------------------------------------
# Runs each filtering config on your RAW accel columns, re-detects events
# with a FIXED detector (so only the filtering stage varies), and scores
# against gold_labels_v1_FROZEN.csv using TWO metrics:
#   - F1 (tIoU@0.5): strict interval-overlap matching
#   - F1 (Tolerance +/-Ts): point/midpoint matching within a time tolerance
#     -- more appropriate for near-instantaneous events; see pipeline_utils
#     .score_events_tolerance for the rationale. Reported alongside, not
#     instead of, tIoU@0.5.
#
# Requires: step1 + step2 already run (gold labels exist).
#
# Run:  python step3_filter_ablation.py
# ---------------------------------------------------------------------------
import os
import pandas as pd
import config
import pipeline_utils as pu

# Fixed detector used across ALL filter configs, so F1 differences are
# attributable to filtering quality, not detector choice.
# k=2.0, window_sec=1.0 come from step6_grid_search.py (900-config sweep
# over the fusion detector) -- the best legitimate config found, not an
# arbitrary pick. Note this uses detect_adaptive_single (single-channel),
# not the fusion detector the grid search optimized, so treat this table's
# absolute F1s as a lower bound relative to step4's fusion results.
FIXED_DETECTOR_K = 3.0
FIXED_DETECTOR_WINDOW_SEC = 1.5

# Tolerance window (seconds) for the point-detection metric. Override by
# adding TOLERANCE_SEC = <value> to config.py; defaults to 0.5s here.
TOLERANCE_SEC = getattr(config, "TOLERANCE_SEC", 0.5)

FILTER_CONFIGS = [
    dict(name="No filtering (raw IMU)",        low=None, high=None, order=None),
    dict(name="Low-pass only (drift removal)", low=None, high=0.3,  order=4),
    dict(name="High-pass only (noise removal)",low=15,   high=None, order=4),
    dict(name="Bandpass (low+high combined)",  low=0.3,  high=15,   order=4),
    dict(name="Bandpass, higher order",        low=0.3,  high=15,   order=6),
    dict(name="Bandpass, narrower band",       low=0.5,  high=10,   order=4),
]

ORDER_SWEEP = [2, 4, 6, 8]  # fixed band 0.3-15 Hz


def run_config(clips, gold, low, high, order):
    """Apply filter config to all clips, detect events with fixed detector,
    return (tiou_scores dict, tolerance_scores dict, mean_snr_db)."""
    pred_by_clip = {}
    snrs = []
    for clip_id, df in clips.items():
        # butter_filter treats low=None,high=None as passthrough regardless
        # of order, so order=None (the "no filtering" config) is safe here.
        fdf = pu.apply_filter_config(df, low, high, order if order is not None else 4)
        events = pu.detect_adaptive_single(fdf, z_col="total_jerk",
                                            window_sec=FIXED_DETECTOR_WINDOW_SEC,
                                            k=FIXED_DETECTOR_K)
        pred_by_clip[clip_id] = events

        # SNR proxy: raw axis vs filtered axis, averaged over the 3 axes
        for axis in ["ax", "ay", "az"]:
            snr = pu.snr_improvement_db(df[f"{axis}_raw"], fdf[f"{axis}_f"])
            if snr == snr:  # not NaN
                snrs.append(snr)

    tiou_scores = pu.score_events_multi_clip(pred_by_clip, gold)
    tol_scores = pu.score_events_tolerance_multi_clip(pred_by_clip, gold, TOLERANCE_SEC)
    mean_snr = sum(snrs) / len(snrs) if snrs else float("nan")
    return tiou_scores, tol_scores, mean_snr


def _fmt(v):
    return round(v, 3) if v == v else "NA"  # NaN check


def main():
    clips = pu.load_all_raw_telemetry()
    gold = pu.load_gold_labels()

    # ---- Main filter table ----
    rows = []
    for cfg in FILTER_CONFIGS:
        tiou_scores, tol_scores, snr = run_config(clips, gold, cfg["low"], cfg["high"], cfg["order"])
        band = "—" if cfg["low"] is None and cfg["high"] is None else \
               f"{cfg['low'] or 0}–{cfg['high'] or '∞'}"
        row = dict(
            Config=cfg["name"],
            **{"Bandpass Range (Hz)": band},
            **{"Filter Order": cfg["order"] if cfg["order"] is not None else "—"},
            **{"SNR Improvement (dB, vs raw)": ("baseline" if cfg["low"] is None and cfg["high"] is None
                                                 else round(snr, 2))},
            **{"Event F1 (tIoU@0.5)": _fmt(tiou_scores["f1"])},
            **{f"Event F1 (Tolerance ±{TOLERANCE_SEC}s)": _fmt(tol_scores["f1"])},
        )
        rows.append(row)

    main_df = pd.DataFrame(rows)

    # ---- Order-sensitivity sub-table (fixed band 0.3-15 Hz) ----
    order_rows = []
    for order in ORDER_SWEEP:
        tiou_scores, tol_scores, snr = run_config(clips, gold, 0.3, 15, order)
        order_rows.append(dict(
            **{"Filter Order": order},
            **{"F1 (tIoU@0.5)": _fmt(tiou_scores["f1"])},
            **{f"F1 (Tolerance ±{TOLERANCE_SEC}s)": _fmt(tol_scores["f1"])},
            **{"Mean SNR (dB)": round(snr, 2)},
        ))
    order_df = pd.DataFrame(order_rows)

    os.makedirs(config.OUT_DIR, exist_ok=True)
    main_df.to_csv(config.FILTER_ABLATION_OUT, index=False)
    order_out = config.FILTER_ABLATION_OUT.replace(".csv", "_order_sweep.csv")
    order_df.to_csv(order_out, index=False)

    print(f"\n=== Filter ablation ({len(clips)} clip(s), gold events: "
          f"{sum(len(v) for v in gold.values())}) ===\n")
    print(main_df.to_string(index=False))
    print("\n--- Order sensitivity (band 0.3-15 Hz) ---\n")
    print(order_df.to_string(index=False))
    print(f"\nSaved:\n  {config.FILTER_ABLATION_OUT}\n  {order_out}")
    print("\nNOTE: SNR-improvement is an in-sample variance-reduction proxy "
          "(no isolated noise-only reference channel exists in this telemetry) -- "
          "caption/footnote this in the paper rather than presenting it as a "
          "calibrated SNR meter. With only "
          f"{len(clips)} clip(s) and {sum(len(v) for v in gold.values())} gold events, "
          "treat F1 differences between configs as directional, not statistically significant.")
    print(f"\nNOTE: 'Tolerance ±{TOLERANCE_SEC}s' is a midpoint-proximity metric reported "
          "alongside tIoU@0.5, not instead of it -- appropriate given many gold events "
          "are near-instantaneous, where interval-overlap matching is disproportionately "
          "strict. Cite both in your table; state the tolerance metric's definition in "
          "a footnote if it goes in the paper.")


if __name__ == "__main__":
    main()
