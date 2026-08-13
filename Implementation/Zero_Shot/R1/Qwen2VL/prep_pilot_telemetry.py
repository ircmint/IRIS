"""
prep_pilot_telemetry.py -- one-time adapter converting the pilot dataset's
raw accel_data.csv / gyro_data.csv (schema: cts,date,Accelerometer [m/s^2],1,2,
temperature [deg C]) into the Accelerometer.csv / Gyroscope.csv format that
zero_shot_infer.py's load_raw_slice() expects (a time-like column it can
filter on, in the SAME units as gold_candidates.csv start_time/end_time,
i.e. seconds elapsed since video start).

Verified mapping (2026-07-25, against ~/IRASTE/DAY_2/candidate_events_updated.csv):
  - accel_data.csv row 1: cts=38.697   -> cts/1000 = 0.038697s
    candidate_events_updated.csv row 1: start_time=0.039s   (match)
  - accel_data.csv last row: cts=709741.4359 -> cts/1000 = 709.7414s
    candidate_events_updated.csv last row: end_time=709.741s (match)
  So `cts` is elapsed milliseconds since video start already synced to the
  video timeline; dividing by 1000 gives seconds directly comparable to
  gold_candidates.csv start_time/end_time. The `date` (wall-clock ISO
  timestamp) column is NOT used as the time axis -- its ~11ms granularity is
  coarser than the ~5ms sample spacing implied by cts and was not needed
  once the cts/1000 match above was confirmed.

Axis-name assumption (documented, not verified against a spec): the header
"Accelerometer [m/s^2],1,2" is a mangled 3-axis export where only the first
axis kept its intended column name. We label them X, Y, Z in that column
order. This labeling does not affect scoring (which never reads axis
identity) -- it only affects the cosmetic column names shown to the VLM in
the raw_telemetry condition's prompt text.
"""
import sys
import pandas as pd

SRC_ACCEL = sys.argv[1] if len(sys.argv) > 1 else "accel_data.csv"
SRC_GYRO = sys.argv[2] if len(sys.argv) > 2 else "gyro_data.csv"
DST_ACCEL = sys.argv[3] if len(sys.argv) > 3 else "Accelerometer.csv"
DST_GYRO = sys.argv[4] if len(sys.argv) > 4 else "Gyroscope.csv"


def convert(src, dst, axis_prefix, unit):
    df = pd.read_csv(src)
    cols = list(df.columns)
    # cols: ['cts', 'date', '<label> [unit]', '1', '2', 'temperature [°C]']
    time_s = df[cols[0]].astype(float) / 1000.0
    out = pd.DataFrame({
        "Time (s)": time_s,
        f"{axis_prefix}_X ({unit})": df[cols[2]],
        f"{axis_prefix}_Y ({unit})": df[cols[3]],
        f"{axis_prefix}_Z ({unit})": df[cols[4]],
    })
    out.to_csv(dst, index=False)
    print(f"Wrote {dst}: {len(out)} rows, time range "
          f"[{out['Time (s)'].min():.3f}, {out['Time (s)'].max():.3f}]s")


convert(SRC_ACCEL, DST_ACCEL, "Accel", "m/s^2")
convert(SRC_GYRO, DST_GYRO, "Gyro", "rad/s")
