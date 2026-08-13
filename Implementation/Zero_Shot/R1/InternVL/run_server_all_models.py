
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
# -*- coding: utf-8 -*-
"""
Final ablation table compiler.

CHANGES from the earlier version:
  1. Florence-2 rows are EXCLUDED from the main table by default. Reason:
     `microsoft/Florence-2-large` (the base checkpoint used here) was not
     trained with a <VQA> task token -- Microsoft's own docs confirm the
     released models don't include VQA capability. Feeding it <VQA> +
     free-text produced out-of-distribution garbage (100% JSON parse
     failure, incoherent text), not a real classification signal. See
     FLORENCE2_MODE below if you later redo Florence-2 as
     "<MORE_DETAILED_CAPTION> + rule-based keyword classifier" (a
     legitimate but methodologically different approach) -- flip the flag
     and it will show up labeled distinctly, not compared as apples-to-
     apples with the VLM-direct-classification rows.
  2. Adds an N column and a footnote block explaining exclusions, so the
     table is self-documenting instead of silently dropping a model.
  3. true_label is read as-is from the results CSVs -- these must already
     be the label-corrected versions (produced by
     fix_labels_and_recompile.py), not the original buggy ones.

FIX (this version): line ~150 referenced `INCLUDE_FLORENCE2`, which is
never defined -- the actual flag is `INCLUDE_FLORENCE2_VQA_BROKEN`. That
mismatch is what caused the NameError in the sbatch run. All references
below now consistently use `INCLUDE_FLORENCE2_VQA_BROKEN`.

Usage:
    python final_ablation_table.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(HOME_ROOT, "IRASTE/DAY_3/local_models_outputs")

# Set True only after Florence-2 has been re-run with a real task token
# (e.g. <MORE_DETAILED_CAPTION>) + a downstream rule-based classifier over
# the caption text. Leave False while Florence-2 CSVs still reflect the
# broken <VQA> run.
INCLUDE_FLORENCE2_VQA_BROKEN = True

configs = [
    {"model": "GroundingDino", "config": "video_only", "display_name": "Video-only object-level", "file_name": "results_gdino_video_only.csv"},
    {"model": "GroundingDino", "config": "telemetry", "display_name": "Video + telemetry (rule-based)", "file_name": "results_gdino_telemetry.csv"},
    {"model": "OwlViTv2", "config": "video_only", "display_name": "Video-only object-level", "file_name": "results_owlv2_video_only.csv"},
    {"model": "OwlViTv2", "config": "telemetry", "display_name": "Video + telemetry (rule-based)", "file_name": "results_owlv2_telemetry.csv"},
    {"model": "Florence-2 (caption+rules)", "config": "video_only", "display_name": "Caption-only, rule-classified", "file_name": "results_florence2_caption_rules_video_only.csv"},
    {"model": "Florence-2 (caption+rules)", "config": "raw_telemetry", "display_name": "Caption + telemetry thresholds", "file_name": "results_florence2_caption_rules_raw_telemetry.csv"},
    {"model": "Florence-2 (caption+rules)", "config": "summarized_telemetry", "display_name": "Caption + telemetry thresholds", "file_name": "results_florence2_caption_rules_summarized_telemetry.csv"},
    {"model": "Florence-2 (caption+rules)", "config": "telemetry_taxonomy", "display_name": "Caption + telemetry thresholds", "file_name": "results_florence2_caption_rules_telemetry_taxonomy.csv"},
    {"model": "InternVL2-2B", "config": "video_only", "display_name": "Video-only zero-shot", "file_name": "results_internvl2_video_only.csv"},
    {"model": "InternVL2-2B", "config": "raw_telemetry", "display_name": "Video + raw telemetry", "file_name": "results_internvl2_raw_telemetry.csv"},
    {"model": "InternVL2-2B", "config": "summarized_telemetry", "display_name": "Video + summarized telemetry", "file_name": "results_internvl2_summarized_telemetry.csv"},
    {"model": "InternVL2-2B", "config": "telemetry_taxonomy", "display_name": "Video + summarized telemetry + taxonomy", "file_name": "results_internvl2_telemetry_taxonomy.csv"},
]

florence_vqa_broken_configs = [
    {"model": "Florence-2", "config": "video_only", "display_name": "Video-only zero-shot", "file_name": "results_florence2_video_only.csv"},
    {"model": "Florence-2", "config": "raw_telemetry", "display_name": "Video + raw telemetry", "file_name": "results_florence2_raw_telemetry.csv"},
    {"model": "Florence-2", "config": "summarized_telemetry", "display_name": "Video + summarized telemetry", "file_name": "results_florence2_summarized_telemetry.csv"},
    {"model": "Florence-2", "config": "telemetry_taxonomy", "display_name": "Video + summarized telemetry + taxonomy", "file_name": "results_florence2_telemetry_taxonomy.csv"},
]

if INCLUDE_FLORENCE2_VQA_BROKEN:
    configs = configs + florence_vqa_broken_configs


def calculate_evaluation_metrics(predictions):
    pred_actions = [p["pred_label"] for p in predictions]
    true_actions = [p["true_label"] for p in predictions]

    unique_labels = list(set(true_actions))
    precisions, recalls, f1s = [], [], []

    for l in unique_labels:
        tp = sum(1 for p, t in zip(pred_actions, true_actions) if p == l and t == l)
        fp = sum(1 for p, t in zip(pred_actions, true_actions) if p == l and t != l)
        fn = sum(1 for p, t in zip(pred_actions, true_actions) if p != l and t == l)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    prec = np.mean(precisions) if precisions else 0.0
    rec = np.mean(recalls) if recalls else 0.0
    f1 = np.mean(f1s) if f1s else 0.0

    total_correct = sum(1 for p, t in zip(pred_actions, true_actions) if p == t)
    acc = total_correct / len(predictions) if len(predictions) > 0 else 0.0

    return {
        "Accuracy": round(acc, 3),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1": round(f1, 3)
    }


def compile_and_print_ablation_table():
    print("\n" + "=" * 80)
    print("FINAL ABLATION TABLE")
    print("=" * 80)

    metrics_rows = []
    skipped = []
    for cfg in configs:
        file_path = os.path.join(OUTPUT_DIR, cfg["file_name"])
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            try:
                df_preds = pd.read_csv(file_path)
                predictions = [
                    {"pred_label": str(r["pred_label"]).lower(), "true_label": str(r["true_label"]).lower()}
                    for _, r in df_preds.iterrows()
                ]
                metrics = calculate_evaluation_metrics(predictions)
                metrics_rows.append({
                    "Model": cfg["model"],
                    "Config": cfg["display_name"],
                    "N": len(predictions),
                    "Accuracy": metrics["Accuracy"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"],
                    "F1": metrics["F1"]
                })
            except Exception as e:
                print(f"  Error reading {cfg['file_name']}: {e}")
        else:
            skipped.append(cfg["file_name"])

    if skipped:
        print("Not yet available, skipped: " + ", ".join(skipped))

    if not metrics_rows:
        print("No completed results files found to compile yet.")
        return

    df_metrics = pd.DataFrame(metrics_rows)
    metrics_csv = os.path.join(OUTPUT_DIR, "local_ablation_metrics.csv")
    df_metrics.to_csv(metrics_csv, index=False)

    report_content = "# Local VLM & Detection Models Ablation Report\n\n"
    report_content += "| Model | Config | N | Accuracy | Precision | Recall | F1 Score |\n"
    report_content += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n"
    for _, row in df_metrics.iterrows():
        report_content += f"| {row['Model']} | {row['Config']} | {row['N']} | {row['Accuracy']:.3f} | {row['Precision']:.3f} | {row['Recall']:.3f} | {row['F1']:.3f} |\n"

    if not INCLUDE_FLORENCE2_VQA_BROKEN:
        report_content += (
            "\n**Note:** Florence-2 (`microsoft/Florence-2-large`) is excluded from this table. "
            "The base checkpoint was not trained with a VQA/free-form-classification task token -- "
            "Microsoft's own model documentation states the released Florence-2 models do not "
            "include VQA capability. Prompting it with an unsupported `<VQA>` task token produced "
            "out-of-distribution, incoherent output (0% valid responses), not a real accuracy signal. "
            "A methodologically valid Florence-2 entry would use a supported task token (e.g. "
            "`<MORE_DETAILED_CAPTION>`) followed by a separate rule-based text classifier over the "
            "caption -- a different method from direct VLM classification, and not yet run.\n"
        )

    report_md_path = os.path.join(OUTPUT_DIR, "local_ablation_report.md")
    with open(report_md_path, "w") as f:
        f.write(report_content)

    print("\n" + report_content)
    print(f"Ablation report saved to: {report_md_path}")
    print(f"Metrics CSV saved to: {metrics_csv}")

    try:
        plt.figure(figsize=(12, 6))
        model_colors = {
            "GroundingDino": "#475569",
            "OwlViTv2": "#64748b",
            "Florence-2": "#e11d48",
            "Florence-2 (caption+rules)": "#e11d48",
            "InternVL2-2B": "#d97706"
        }
        colors = [model_colors.get(row["Model"], "#000000") for _, row in df_metrics.iterrows()]
        bars = plt.bar(range(len(df_metrics)), df_metrics["F1"], color=colors, edgecolor="#0f172a", width=0.5, zorder=3)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f"{yval:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        plt.ylabel("Macro F1 Score", fontsize=11, fontweight="bold")
        plt.title("Model Performance Comparison (F1 Score)", fontsize=13, fontweight="bold")
        labels = [f"{row['Model']}\n({row['Config'][:28]})" for _, row in df_metrics.iterrows()]
        plt.xticks(range(len(df_metrics)), labels, rotation=25, ha="right", fontsize=9)
        plt.ylim(0.0, 1.1)
        plt.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "local_ablation_plot.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Plot saved to: {plot_path}")
    except Exception as e:
        print(f"Failed to generate plot: {e}")


if __name__ == "__main__":
    compile_and_print_ablation_table()