# check_data_quality.py
# ---------------------------------------------------------------------------
# One-off diagnostic: confirms whether NaN/Inf values in your pipeline are
# genuine bugs or legitimate signal content, and explains every NA you saw
# in the step3/step4 output.
#
# Run:  python check_data_quality.py
# ---------------------------------------------------------------------------
import numpy as np
import config
import pipeline_utils as pu

FILTER_CONFIGS_TO_CHECK = [
    ("No filtering (raw IMU)",        None, None, None),
    ("Low-pass only (drift removal)", None, 0.3,  4),
    ("High-pass only (noise removal)",15,   None, 4),
    ("Bandpass (low+high combined)",  0.3,  15,   4),
    ("Bandpass, higher order",        0.3,  15,   6),
    ("Bandpass, narrower band",       0.5,  10,   4),
]


def report(name, arr):
    arr = np.asarray(arr, dtype=float)
    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    n_neg = int((arr < 0).sum())
    print(f"  {name:12s}  NaN={n_nan:6d}  Inf={n_inf:6d}  "
          f"negative(legit,signed)={n_neg:6d}  "
          f"min={np.nanmin(arr):9.4f}  max={np.nanmax(arr):9.4f}")


def main():
    clips = pu.load_all_raw_telemetry()
    print(f"Loaded {len(clips)} clip(s)\n")

    for clip_id, df in clips.items():
        print(f"=== {clip_id} ({len(df)} samples) ===")
        print("-- raw columns --")
        for axis in ["ax", "ay", "az"]:
            report(f"{axis}_raw", df[f"{axis}_raw"])

        for name, low, high, order in FILTER_CONFIGS_TO_CHECK:
            print(f"\n-- {name} --")
            try:
                fdf = pu.apply_filter_config(df, low, high, order if order is not None else 4)
            except FloatingPointError as e:
                print(f"  *** GENUINE BUG CAUGHT: {e}")
                continue
            for axis in ["ax", "ay", "az"]:
                report(f"{axis}_f", fdf[f"{axis}_f"])
            report("total_jerk", fdf["total_jerk"])

            # Explain why a config might yield zero detections (the source of
            # the "NA" rows in step3/step4 -- not a NaN-from-filtering bug).
            std = fdf["total_jerk"].std()
            print(f"  total_jerk std = {std:.6f}  "
                  f"{'<-- near-zero: rolling z-scores will be NaN -> 0 events -> NA row (expected for aggressive low-pass, not a filtering bug)' if std < 1e-6 else ''}")
        print()

    print("Summary:")
    print(" - 'negative' counts above are EXPECTED and are not errors -- accel/jerk are")
    print("   signed physical quantities. Only NaN/Inf counts indicate an actual problem.")
    print(" - If any config prints 'GENUINE BUG CAUGHT', that's a real numerical")
    print("   instability -- send me that exact message and I'll fix the filter design.")
    print(" - If every NaN/Inf count above is 0, the filtering stage is clean, and the")
    print("   'NA' row you saw in step3/step4 output is legitimate zero-variance behavior")
    print("   from the low-pass config, not a bug in the filter.")


if __name__ == "__main__":
    main()