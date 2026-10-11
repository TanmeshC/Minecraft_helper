import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FIGURES_DIR = os.path.join("docs", "figures")
PROCESSED_DIR = os.path.join("data", "processed")

def generate_charts():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    summary_path = os.path.join(PROCESSED_DIR, "pipeline_summary.json")
    if not os.path.exists(summary_path):
        print(f"Summary file not found at {summary_path}")
        return

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # 1. Intent Category Distribution
    intent_dist = summary.get("intent_distribution", {})
    if intent_dist:
        plt.figure(figsize=(10, 6))
        sorted_intents = sorted(intent_dist.items(), key=lambda x: x[1], reverse=True)
        keys, vals = zip(*sorted_intents)
        bars = plt.barh(keys, vals, color="#2e7d32")
        plt.xlabel("Number of Questions")
        plt.title("MineMentor Clean Dataset: Questions by Intent Category")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        intent_fig_path = os.path.join(FIGURES_DIR, "intents_distribution.png")
        plt.savefig(intent_fig_path, dpi=200)
        plt.close()
        print(f"Saved intent chart to {intent_fig_path}")

    # 2. Difficulty Distribution
    diff_dist = summary.get("difficulty_distribution", {})
    if diff_dist:
        plt.figure(figsize=(7, 5))
        d_keys = list(diff_dist.keys())
        d_vals = list(diff_dist.values())
        colors = ["#4caf50", "#2196f3", "#ff9800", "#f44336"]
        plt.bar(d_keys, d_vals, color=colors[:len(d_keys)])
        plt.ylabel("Number of Questions")
        plt.title("Distribution of Questions by Player Difficulty Tier")
        for i, v in enumerate(d_vals):
            plt.text(i, v + (max(d_vals)*0.01), f"{v:,}", ha="center", fontweight="bold")
        plt.tight_layout()
        diff_fig_path = os.path.join(FIGURES_DIR, "difficulty_distribution.png")
        plt.savefig(diff_fig_path, dpi=200)
        plt.close()
        print(f"Saved difficulty chart to {diff_fig_path}")

    # 3. Model Performance Comparison
    intent_metrics_path = os.path.join(PROCESSED_DIR, "intent_metrics.json")
    diff_metrics_path = os.path.join(PROCESSED_DIR, "difficulty_metrics.json")

    if os.path.exists(intent_metrics_path) and os.path.exists(diff_metrics_path):
        with open(intent_metrics_path, "r", encoding="utf-8") as f:
            im = json.load(f)
        with open(diff_metrics_path, "r", encoding="utf-8") as f:
            dm = json.load(f)

        models = ["Intent Classifier\n(16 Classes)", "Difficulty Classifier\n(4 Classes)"]
        accuracies = [im.get("accuracy", 0) * 100, dm.get("accuracy", 0) * 100]
        f1_macros = [im.get("macro_f1", 0) * 100, dm.get("macro_f1", 0) * 100]
        f1_weighteds = [im.get("weighted_f1", 0) * 100, dm.get("weighted_f1", 0) * 100]

        x = np.arange(len(models))
        width = 0.25

        plt.figure(figsize=(9, 5))
        plt.bar(x - width, accuracies, width, label="Accuracy (%)", color="#1976d2")
        plt.bar(x, f1_weighteds, width, label="Weighted F1 (%)", color="#388e3c")
        plt.bar(x + width, f1_macros, width, label="Macro F1 (%)", color="#f57c00")

        plt.ylabel("Score (%)")
        plt.title("ML Models Evaluation Performance on Held-Out Test Set (26,426 samples)")
        plt.xticks(x, models)
        plt.ylim(0, 105)
        plt.legend()
        plt.tight_layout()
        perf_fig_path = os.path.join(FIGURES_DIR, "model_performance.png")
        plt.savefig(perf_fig_path, dpi=200)
        plt.close()
        print(f"Saved model performance chart to {perf_fig_path}")

if __name__ == "__main__":
    generate_charts()
