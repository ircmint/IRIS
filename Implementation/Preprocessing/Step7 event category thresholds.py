# step7_event_category_thresholds.py
# ---------------------------------------------------------------------------
# Computes REAL min/max/median duration and peak-jerk ranges per evasive-
# action category, from your actual reviewed candidate_events.csv.
#
# This is a diagnostic/descriptive table, NOT a per-category detection
# threshold -- category is only known after HITL review, which happens
# after detection. The pipeline's actual detector (step4) uses ONE global
# k/window for all events regardless of category; this script exists to
# check whether that's actually a reasonable simplification (do different
# categories need different magnitudes?) using your real labeled data.
#
# Categories with ZERO examples are reported as such, explicitly -- this
# script will never invent a min/max range for a category with no data in
# your reviewed set. If a category you need (e.g. braking, near-miss) shows
# 0 examples, that means this clip's HITL review never described an event
# that way -- not that the category doesn't exist in principle. Options:
#   1. Review additional clips that may contain those event types
#   2. Manually inspect a few of your 42 REJECTED candidates -- some may be
#      real braking/near-miss events that got rejected for other reasons;
#      re-labeling any of those would give this script real data to work with
#   3. Use published biomechanics/IMU literature values as an explicitly-
#      labeled ASSUMPTION (not a measured result) if you need a placeholder
#      for Day-1 -- but flag it as an assumption in the paper, not a finding
#
# Run:  python step7_event_category_thresholds.py
# ---------------------------------------------------------------------------
import pandas as pd
import config
import pipeline_utils as pu


def main():
    df = pd.read_csv(config.CANDIDATE_EVENTS_CSV)
    df["decision"] = df["decision"].astype(str).str.strip().str.lower()
    confirmed = df[df["decision"] == "confirm"].copy()
    rejected = df[df["decision"] == "reject"].copy()

    confirmed["category"] = confirmed["notes"].apply(pu.categorize_event_notes)

    print(f"=== Event category analysis: {len(confirmed)} confirmed, "
          f"{len(rejected)} rejected candidates ===\n")

    all_categories = [c for c, _ in pu.CATEGORY_KEYWORDS] + ["other"]
    rows = []
    for cat in all_categories:
        sub = confirmed[confirmed["category"] == cat]
        if len(sub) == 0:
            rows.append(dict(
                Category=cat, N=0,
                **{"Peak z-jerk range": "NO DATA -- 0 examples in reviewed set"},
                **{"Duration range (s)": "NO DATA -- 0 examples in reviewed set"},
            ))
            continue
        rows.append(dict(
            Category=cat, N=len(sub),
            **{"Peak z-jerk range": f"{sub['peak_abs_z_jerk'].min():.3f} - {sub['peak_abs_z_jerk'].max():.3f}"},
            **{"Duration range (s)": f"{sub['duration'].min():.3f} - {sub['duration'].max():.3f}"},
        ))
    result_df = pd.DataFrame(rows)

    print(result_df.to_string(index=False))

    empty_cats = result_df[result_df["N"] == 0]["Category"].tolist()
    if empty_cats:
        print(f"\n*** Categories with ZERO real examples in this clip's reviewed data: "
              f"{', '.join(empty_cats)} ***")
        print("Do NOT report threshold ranges for these categories as measured results.")
        print("See script docstring for legitimate options to address this for Day 2.")

    out_path = config.OUT_DIR + "/event_category_thresholds.csv"
    import os
    os.makedirs(config.OUT_DIR, exist_ok=True)
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()