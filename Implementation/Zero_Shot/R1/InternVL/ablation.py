
# ============================================================
# PATHS — edit or set environment variables before running
#   IRASTE_SCRATCH : root of your scratch/working directory
#   IRASTE_HOME    : your home directory on the compute node
#   HF_HOME        : HuggingFace cache directory
# ============================================================
import os as _os
SCRATCH_ROOT = _os.environ.get("IRASTE_SCRATCH", "/scratch/<your_username>")
HOME_ROOT    = _os.environ.get("IRASTE_HOME",    "/home/<your_username>")
HF_CACHE     = _os.environ.get("HF_HOME",        _os.path.join(SCRATCH_ROOT, "hf_cache"))
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(HOME_ROOT, "IRASTE/DAY_3/local_models_outputs")
RELABELED_CSV = os.path.join(HOME_ROOT, "IRASTE/DAY_2/candidate_events_updated_relabeled.csv")

# ----------------------------------------------------------------------
# Build a lookup: (clip_id, start_time, end_time) -> corrected true label
# ----------------------------------------------------------------------
df_true = pd.read_csv(RELABELED_CSV)

# Round times to avoid float-matching mismatches across files written by
# different scripts/runs
df_true["_start_r"] = df_true["start_time"].round(3)
df_true["_end_r"] = df_true["end_time"].round(3)

true_lookup = {}
for _, row in df_true.iterrows():
    key = (str(row["clip_id"]), row["_start_r"], row["_end_r"])
    true_lookup[key] = str(row["true_evasive_action"]).lower().strip()

print(f"Loaded {len(true_lookup)} corrected ground-truth labels from {RELABELED_CSV}")
print("Label distribution in corrected ground truth:")
print(df_true["true_evasive_action"].value_counts())
print()

# ----------------------------------------------------------------------
# Known model file-prefix -> display-name mapping (same as your main script)
# ----------------------------------------------------------------------
KNOWN_MODEL_PATTERNS = [
    ("gdino",        "GroundingDino"),
    ("owlv2",        "OwlViTv2"),
    ("florence2",    "Florence-2"),
    ("internvl2_2b", "InternVL2-2B"),
    ("internvl2_8b", "InternVL2-8B"),
]

def _display_config_name(config_key: str) -> str:
    mapping = {
        "video_only": "Video-only zero-shot",
        "raw_telemetry": "Video + raw telemetry",
        "summarized_telemetry": "Video + summarized telemetry",
        "telemetry_taxonomy": "Video + summarized telemetry + taxonomy",
        "telemetry": "Video + telemetry (rule-based)",
    }
    return mapping.get(config_key, config_key)

def calculate_evaluation_metrics(pred_actions, true_actions):
    unique_labels = list(set(true_actions))
    precisions, recalls, f1s = [], [], []
    for l in unique_labels:
        tp = sum(1 for p, t in zip(pred_actions, true_actions) if p == l and t == l)
        fp = sum(1 for p, t in zip(pred_actions, true_actions) if p == l and t != l)
        fn = sum(1 for p, t in zip(pred_actions, true_actions) if p != l and t == l)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p); recalls.append(r); f1s.append(f1)
    prec = np.mean(precisions) if precisions else 0.0
    rec = np.mean(recalls) if recalls else 0.0
    f1 = np.mean(f1s) if f1s else 0.0
    total_correct = sum(1 for p, t in zip(pred_actions, true_actions) if p == t)
    acc = total_correct / len(pred_actions) if len(pred_actions) > 0 else 0.0
    return {
        "Accuracy": round(acc, 3),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1": round(f1, 3),
    }

# ----------------------------------------------------------------------
# Rescore every results_*.csv against the corrected labels
# ----------------------------------------------------------------------
metrics_rows = []
skipped_files = []

for fname in sorted(os.listdir(OUTPUT_DIR)):
    if not (fname.startswith("results_") and fname.endswith(".csv")):
        continue
    if fname.endswith(".bak"):
        continue

    matched_model = None
    config_key = None
    for key, display in KNOWN_MODEL_PATTERNS:
        if f"results_{key}_" in fname:
            matched_model = display
            config_key = fname.replace(f"results_{key}_", "").replace(".csv", "")
            break
    if matched_model is None:
        continue

    file_path = os.path.join(OUTPUT_DIR, fname)
    if not (os.path.exists(file_path) and os.path.getsize(file_path) > 0):
        continue

    try:
        df_preds = pd.read_csv(file_path)
    except Exception as e:
        print(f"  Error reading {fname}: {e}")
        continue

    required_cols = {"pred_label", "clip_id", "start_time", "end_time"}
    if not required_cols.issubset(df_preds.columns):
        print(f"  Skipping {fname}: missing one of {required_cols} (has {list(df_preds.columns)})")
        skipped_files.append(fname)
        continue

    corrected_true = []
    unmatched = 0
    for _, row in df_preds.iterrows():
        key = (str(row["clip_id"]), round(float(row["start_time"]), 3), round(float(row["end_time"]), 3))
        if key in true_lookup:
            corrected_true.append(true_lookup[key])
        else:
            corrected_true.append(None)
            unmatched += 1

    if unmatched > 0:
        print(f"  WARNING: {fname} -- {unmatched}/{len(df_preds)} rows had no matching "
              f"(clip_id, start_time, end_time) key in the relabeled CSV. These rows are excluded from scoring.")

    df_preds["corrected_true_label"] = corrected_true
    df_scored = df_preds.dropna(subset=["corrected_true_label"]).copy()
    df_scored["pred_label"] = df_scored["pred_label"].astype(str).str.lower().str.strip()

    pred_actions = df_scored["pred_label"].tolist()
    true_actions = df_scored["corrected_true_label"].tolist()

    metrics = calculate_evaluation_metrics(pred_actions, true_actions)
    metrics_rows.append({
        "Model": matched_model,
        "Config": _display_config_name(config_key),
        "N": len(df_scored),
        "Accuracy": metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "F1": metrics["F1"],
    })

    # Save the rescored predictions file (does not overwrite the original)
    rescored_path = os.path.join(OUTPUT_DIR, fname.replace(".csv", "_rescored.csv"))
    df_preds.to_csv(rescored_path, index=False)

if not metrics_rows:
    print("No results files could be rescored -- check column names / paths above.")
else:
    df_metrics = pd.DataFrame(metrics_rows)
    metrics_csv = os.path.join(OUTPUT_DIR, "ablation_metrics_CORRECTED.csv")
    df_metrics.to_csv(metrics_csv, index=False)

    report_content = "# VLM & Detection Models Ablation Report (corrected ground truth)\n\n"
    report_content += "| Model | Config | N | Accuracy | Precision | Recall | F1 Score |\n"
    report_content += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n"
    for _, row in df_metrics.iterrows():
        report_content += (f"| {row['Model']} | {row['Config']} | {row['N']} | "
                            f"{row['Accuracy']:.3f} | {row['Precision']:.3f} | "
                            f"{row['Recall']:.3f} | {row['F1']:.3f} |\n")

    report_md_path = os.path.join(OUTPUT_DIR, "ablation_report_CORRECTED.md")
    with open(report_md_path, "w") as f:
        f.write(report_content)

    print("\n" + report_content)
    print(f"Ablation report saved to: {report_md_path}")
    print(f"Metrics CSV saved to: {metrics_csv}")

    try:
        plt.figure(figsize=(14, 6))
        model_colors = {
            "GroundingDino": "#475569",
            "OwlViTv2": "#64748b",
            "Florence-2": "#e11d48",
            "InternVL2-2B": "#d97706",
            "InternVL2-8B": "#059669",
        }
        colors = [model_colors.get(row["Model"], "#000000") for _, row in df_metrics.iterrows()]
        bars = plt.bar(range(len(df_metrics)), df_metrics["F1"], color=colors, edgecolor="#0f172a", width=0.5, zorder=3)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        plt.ylabel("Macro F1 Score", fontsize=11, fontweight="bold")
        plt.title("Model Performance Comparison (F1 Score) -- Corrected Ground Truth", fontsize=13, fontweight="bold")
        labels = [f"{row['Model']}\n({row['Config']})" for _, row in df_metrics.iterrows()]
        plt.xticks(range(len(df_metrics)), labels, rotation=30, ha="right", fontsize=8)
        plt.ylim(0.0, 1.1)
        plt.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "ablation_plot_CORRECTED.png"), dpi=150)
        plt.close()
        print(f"Comparison plot saved to: {OUTPUT_DIR}/ablation_plot_CORRECTED.png")
    except Exception as e:
        print(f"Failed to generate plot: {e}")

if skipped_files:
    print(f"\nSkipped files missing required columns: {skipped_files}")