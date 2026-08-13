# step1_build_candidates.py
# ---------------------------------------------------------------------------
# Reads every CSV in config.RAW_TELEMETRY_DIR, groups consecutive
# evasive_flag=1 rows into candidate events, writes candidate_events.csv
# for you to review by hand (fill in decision / adjusted_start / adjusted_end).
#
# Run:  python step1_build_candidates.py
# ---------------------------------------------------------------------------
import os
import pandas as pd
import config
import pipeline_utils as pu


def build_candidates():
    os.makedirs(config.OUT_DIR, exist_ok=True)
    clips = pu.load_all_raw_telemetry()

    rows = []
    for clip_id, df in clips.items():
        if "evasive_flag" not in df.columns:
            raise ValueError(f"{clip_id}: no evasive_flag column found.")
        events = pu.flags_to_events(df, "evasive_flag")
        for (start, end) in events:
            window = df[(df[config.TIME_COL] >= start) & (df[config.TIME_COL] <= end)]
            peak_conf = window["confidence"].max() if "confidence" in window else float("nan")
            peak_zjerk = window["z_jerk"].abs().max() if "z_jerk" in window else float("nan")
            peak_zaz = window["z_az"].abs().max() if "z_az" in window else float("nan")
            evasive_type = (window["evasive_type"].mode().iloc[0]
                             if "evasive_type" in window and not window["evasive_type"].mode().empty
                             else "")
            rows.append(dict(
                clip_id=clip_id,
                start_time=round(float(start), 3),
                end_time=round(float(end), 3),
                duration=round(float(end - start), 3),
                n_samples=len(window),
                peak_confidence=peak_conf,
                peak_abs_z_jerk=peak_zjerk,
                peak_abs_z_az=peak_zaz,
                evasive_type=evasive_type,
                decision="",       # <-- fill in: confirm / reject
                adjusted_start="", # <-- optional correction
                adjusted_end="",   # <-- optional correction
                category="",       # <-- optional free-text
                notes="",          # <-- optional free-text
            ))

    out = pd.DataFrame(rows)
    out.to_csv(config.CANDIDATE_EVENTS_CSV, index=False)
    print(f"Wrote {len(out)} candidate events from {len(clips)} clip(s) to:")
    print(f"  {config.CANDIDATE_EVENTS_CSV}")
    print("Next: open that file, watch each clip around [start_time, end_time],")
    print("      and set decision to 'confirm' or 'reject' (adjust times if needed).")


if __name__ == "__main__":
    build_candidates()
