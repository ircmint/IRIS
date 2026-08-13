# Filtering and Thresholding

IMU-based pipeline to extract, annotate, and validate evasive event candidates from raw GoPro telemetry.

## Overview

Raw GoPro exports provide 3 separate CSV streams (accelerometer, gyroscope, GPS) at different sample rates. This pipeline:
1. Merges and aligns the streams on a common time axis
2. Bandpass-filters the accelerometer to isolate evasive motion signatures
3. Detects candidate events using rolling z-score thresholding
4. Supports human-in-the-loop (HITL) annotation to build gold labels
5. Runs ablation studies on filter parameters and detection thresholds

## Files

| Script | Step | Purpose |
|--------|------|---------|
| `config.py` | — | **Edit this first.** All paths, thresholds, and column names |
| `pipeline_utils.py` | — | Shared utilities: bandpass filter, z-score detector, tIoU scorer |
| `step0_merge_raw_streams.py` | 0 | Merge accel/gyro/GPS CSVs → single merged telemetry CSV |
| `step1_build_candidates.py` | 1 | Run fusion detector → `candidate_events.csv` for human review |
| `step2_build_gold.py` | 2 | After HITL annotation → produce frozen `gold_labels.csv` |
| `step3_filter_ablation.py` | 3 | Ablate bandpass filter parameters (low/high cutoff, order) |
| `step4_threshold_ablation.py` | 4 | Sweep z-score thresholds and window sizes |
| `step5_hitl_ablation.py` | 5 | Measure inter-annotator agreement and offset sensitivity |
| `Step6 grid search.py` | 6 | Joint grid search over filter + threshold parameter space |
| `Step7 event category thresholds.py` | 7 | Per-category threshold tuning (braking vs lane-change vs accel) |
| `Check data quality.py` | — | Visualize signal quality and flag problematic segments |

## Quick Start

```bash
# 1. Edit config.py to point to your raw data files
# 2. Merge raw streams
python step0_merge_raw_streams.py

# 3. Extract candidates → review candidate_events.csv and fill in the label column
python step1_build_candidates.py

# 4. Build frozen gold labels after annotation
python step2_build_gold.py

# 5. Run ablation studies
python step3_filter_ablation.py
python step4_threshold_ablation.py
python step5_hitl_ablation.py
```

## Signal Processing Details

- **Bandpass filter:** Butterworth (default: 0.3–15 Hz, order 4) to isolate body-motion frequencies
- **Jerk computation:** First-order difference of filtered acceleration
- **Fusion detector:** Requires co-occurrence across ≥2 axes (jerk_X/Y/Z) within a 1-second window
- **tIoU:** Temporal Intersection-over-Union used to match predicted vs gold events (threshold: 0.5)

## Output Files

| File | Description |
|------|-------------|
| `candidate_events.csv` | All detected events with provisional labels (for human review) |
| `gold_labels_v1_FROZEN.csv` | Human-verified final labels — **do not edit** |
| `filter_ablation_results.csv` | Filter sweep results |
| `threshold_ablation_results.csv` | Threshold sweep results |
| `grid_search_all_results.csv` | Joint filter+threshold grid search |
