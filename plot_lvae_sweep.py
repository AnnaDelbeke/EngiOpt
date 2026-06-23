"""
Live-updating metrics chart for the lambda_lv sweep.
Parses all eval txt files, maps to version + lambda, plots key metrics vs lambda.
Run directly: python plot_sweep_metrics.py
"""

import re
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

EVAL_DIR = "results/lvae_3d_evaluation"
OUT_PATH = os.path.join(EVAL_DIR, "sweep_metrics.png")

# version → lambda_lv mapping (v15 excluded: trained without spectral norm)
VERSION_LAMBDA = {
    "v16": 1e-4,
    "v17": 1e-4,   # same lambda as v16, different frob; keep both
    "v18": 1e-3,
    "v19": 1e-2,
    "v20": 3e-2,
    "v21": 5e-2,
    "v22": 1e-1,
    "v23": 2e-3,
    "v24": 5e-3,
    "v25": 8e-3,
    "v26": 7e-5,
    "v27": 1e-5,
    "v28": 5e-4,
    "v29": 5e-1,
    "v30": 1.0,
}

def parse_eval_txt(path):
    metrics = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r"shape_mse\s*:\s*([\d.e+-]+)", line)
            if m: metrics["shape_mse"] = float(m.group(1))
            m = re.match(r"shape_r2\s*:\s*([\d.e+-]+)", line)
            if m: metrics["shape_r2"] = float(m.group(1))
            m = re.match(r"pressure_mse\s*:\s*([\d.e+-]+)", line)
            if m: metrics["pressure_mse"] = float(m.group(1))
            m = re.match(r"pressure_r2\s*:\s*([\d.e+-]+)", line)
            if m: metrics["pressure_r2"] = float(m.group(1))
            m = re.match(r"aoa_mse\s*:\s*([\d.e+-]+)", line)
            if m: metrics["aoa_mse"] = float(m.group(1))
            m = re.match(r"latent_lvae_mmd\s*:\s*([\d.e+-]+)", line)
            if m: metrics["latent_mmd"] = float(m.group(1))
            m = re.match(r"Checkpoint:\s+results/lvae_3d/(lvae_3d_(v\w+))_best", line)
            if m: metrics["version"] = m.group(2)
    return metrics if "version" in metrics and "pressure_mse" in metrics else None


def collect():
    rows = {}
    for path in sorted(glob.glob(os.path.join(EVAL_DIR, "eval_*.txt")), key=os.path.getmtime):
        result = parse_eval_txt(path)
        if result is None:
            continue
        ver = result["version"]
        lam = VERSION_LAMBDA.get(ver)
        if lam is None:
            continue
        # keep the latest eval for each version (files are sorted by timestamp)
        rows[ver] = {"lambda": lam, **result}
    return list(rows.values())


def plot(rows):
    if not rows:
        print("No eval results found yet.")
        return

    rows.sort(key=lambda r: r["lambda"])

    p_mse   = [r["pressure_mse"] for r in rows]
    p_r2    = [r["pressure_r2"]  for r in rows]
    s_mse   = [r["shape_mse"]    for r in rows]
    aoa     = [r["aoa_mse"]      for r in rows]

    xs = np.arange(len(rows))
    xlabels = [f"{r['lambda']:.0e}" for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    def bar_plot(ax, ys, title, ylabel, color, log=False):
        bars = ax.bar(xs, ys, color=color, alpha=0.8, edgecolor="k", linewidth=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(xs)
        ax.set_xticklabels(xlabels, fontsize=7)
        if log:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        pass

    bar_plot(axes[0, 0], p_mse, "Pressure MSE ↓", "MSE", "#e07b54", log=True)
    bar_plot(axes[0, 1], p_r2,  "Pressure R² ↑",  "R²",  "#5ba85b")
    bar_plot(axes[1, 0], s_mse, "Shape MSE ↓",    "MSE", "#5b8dc8", log=True)
    bar_plot(axes[1, 1], aoa,   "AoA MSE ↓",      "MSE", "#c8a85b", log=True)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}  ({len(rows)} models: {[r['version'] for r in rows]})")


if __name__ == "__main__":
    rows = collect()
    print(f"Found {len(rows)} evaluated models: {[r['version'] for r in rows]}")
    plot(rows)
