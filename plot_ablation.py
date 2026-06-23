"""
Data ablation plots for all pipeline components.

Usage:
    .venv/bin/python plot_ablation.py --component bae
    .venv/bin/python plot_ablation.py --component lvae
    .venv/bin/python plot_ablation.py --component ddmw
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"text.usetex": False, "font.family": "DejaVu Sans"})
import matplotlib.pyplot as plt
import numpy as np

OUT = "thesis/figures"
os.makedirs(OUT, exist_ok=True)

# ── shared helpers ────────────────────────────────────────────────────────────

def load_eval_jsons(ablation_dir):
    by_n = defaultdict(list)
    for n_dir in sorted(glob.glob(os.path.join(ablation_dir, "n*_s*"))):
        m = re.search(r"n(\d+)_s(\d+)", os.path.basename(n_dir))
        if not m:
            continue
        n = int(m.group(1))
        jsons = sorted(glob.glob(os.path.join(n_dir, "eval_*.json")))
        if not jsons:
            continue
        with open(jsons[-1]) as f:
            by_n[n].append(json.load(f))
    return by_n


def get_series(by_n, metric):
    sizes, means, stds = [], [], []
    for n in sorted(by_n):
        vals = [r[metric] for r in by_n[n] if metric in r]
        if vals:
            sizes.append(n)
            means.append(np.mean(vals))
            stds.append(np.std(vals))
    return np.array(sizes), np.array(means), np.array(stds)


# ── BAE ablation ──────────────────────────────────────────────────────────────

def plot_bae():
    import torch
    sys.path.insert(0, ".")
    from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
    from engiopt.bezier_ae.train_bezier_ae_3d import WingsBezierDataset3D
    from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

    BASE      = "results/bezier_ae_3d_ablation"
    N_SAMPLES = [50, 100, 150, 200, 300, 400, 500, 600, 700, 767]
    SEEDS     = [0, 1]
    SHOW_NS   = [50, 200, 400, 767]
    WING_IDX  = 8
    MIDSPAN   = 7

    results = {}
    for n in N_SAMPLES:
        vals = []
        paths = ([f"{BASE}/n767_s0/eval_metrics.json"] if n == 767
                 else [f"{BASE}/n{n}_s{s}/eval_metrics.json" for s in SEEDS])
        for p in paths:
            if os.path.exists(p):
                with open(p) as f:
                    vals.append(json.load(f)["mse_mean"])
        if vals:
            results[n] = vals

    ns    = sorted(results.keys())
    means = [np.mean(results[n]) for n in ns]
    stds  = [np.std(results[n])  for n in ns]

    # (a) MSE curve
    fig_a, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ns, means, color="#4477AA", marker="o", linewidth=1.5, markersize=5)
    ax.fill_between(ns,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color="#4477AA", alpha=0.2)
    ax.set_ylabel("Test MSE", fontsize=10)
    ax.set_xlabel("Number of training wings", fontsize=10)
    ax.set_xticks(ns)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(axis="y", direction="out", which="both", right=False)
    ax.tick_params(axis="x", which="both", direction="out", top=False)
    fig_a.tight_layout()
    for ext in ("pdf", "png"):
        fig_a.savefig(f"{OUT}/ablation_bae3d_curve.{ext}", dpi=150, bbox_inches="tight")
        print(f"Saved: {OUT}/ablation_bae3d_curve.{ext}")
    plt.close(fig_a)

    # (b) Airfoil strip
    ds = NewWingsDataset("Wing_TL/data/processed/new_dataset_slices.pkl",
                         "Wing_TL/data/processed/new_dataset_scalars.pkl", seed=0)
    test_items = [it for it in list(ds["test"]) if it["final"] == 1]
    test_ds = WingsBezierDataset3D(test_items)
    x_gt = test_ds[WING_IDX][MIDSPAN].numpy()

    def load_model(n):
        if n == 767:
            ckpt_path = "results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt"
        else:
            ckpt_path = f"{BASE}/n{n}_s0/models/bae_3d_ablation_n{n}_s0_best.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = BezierAutoencoder3D(
            n_spans=15, n_control_points=32, n_data_points=192,
            slice_hidden_dims=ckpt.get("slice_hidden_dims", [256, 256, 256]),
            span_hidden_dims=ckpt.get("span_hidden_dims", [256, 256]),
            latent_dim=ckpt.get("latent_dim", 128),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model

    show_ns = [n for n in SHOW_NS if n in results]
    fig_b, axes_b = plt.subplots(2, 2, figsize=(5, 2.2),
                                  gridspec_kw={"hspace": 0.35, "wspace": 0.05})
    for idx, n in enumerate(show_ns[:4]):
        r, c = divmod(idx, 2)
        ax_b = axes_b[r, c]
        model = load_model(n)
        x_in = test_ds[WING_IDX].unsqueeze(0)
        with torch.no_grad():
            y, _ = model(x_in)
        y_np = y.squeeze(0)[MIDSPAN].numpy()
        ax_b.plot(x_gt[0], x_gt[1], color="#4477AA", lw=1.5)
        ax_b.plot(y_np[0], y_np[1], color="#EE6677", lw=1.5, linestyle="--")
        ax_b.set_xlim(-0.05, 1.05)
        ax_b.set_ylim(-0.10, 0.16)
        ax_b.set_aspect("equal")
        ax_b.axis("off")
        ax_b.set_title(f"n = {n}", fontsize=9, pad=3)
    for ext in ("pdf", "png"):
        fig_b.savefig(f"{OUT}/ablation_bae3d_airfoils.{ext}", dpi=150, bbox_inches="tight")
        print(f"Saved: {OUT}/ablation_bae3d_airfoils.{ext}")
    plt.close(fig_b)


# ── LVAE ablation ─────────────────────────────────────────────────────────────

def plot_lvae():
    JOINT_DIR = "results/lvae_3d_ablation"
    GEOM_DIR  = "results/lvae_3d_ablation_geom_only"

    METRICS = {
        "shape_mse":    "Shape MSE",
        "pressure_mse": "Pressure MSE",
        "aoa_mse":      "AoA MSE",
        "mmd":          "Coord MMD",
        "mmd_w":        "Latent MMD",
    }
    C_JOINT = "steelblue"
    C_GEOM  = "darkorange"

    joint = load_eval_jsons(JOINT_DIR)
    geom  = load_eval_jsons(GEOM_DIR)

    metrics_to_plot = [k for k in METRICS
                       if any(k in r for runs in joint.values() for r in runs)]

    n_cols = 3
    n_rows = (len(metrics_to_plot) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 4.2, n_rows * 3.2))
    axes = np.array(axes).flatten()

    for ax, key in zip(axes, metrics_to_plot):
        for data, color, label, skip in [
            (joint, C_JOINT, "Joint (geom + pressure)", set()),
            (geom,  C_GEOM,  "Geom-only",              {"pressure_mse"}),
        ]:
            if key in skip:
                continue
            sizes, means, stds = get_series(data, key)
            if not len(sizes):
                continue
            ax.plot(sizes, means, marker="o", linewidth=1.8, color=color, label=label)
            ax.fill_between(sizes, means - stds, means + stds, alpha=0.20, color=color)
        ax.set_xlabel("Training dataset size", fontsize=9)
        ax.set_title(METRICS[key], fontsize=10)
        ax.set_xticks(sorted(joint.keys()))
        ax.tick_params(axis="x", rotation=45, labelsize=9)

    empty = [ax for ax in axes[len(metrics_to_plot):]]
    for ax in empty:
        ax.set_visible(False)
    if empty:
        empty[0].set_visible(True)
        empty[0].set_axis_off()
        handles, labels = axes[0].get_legend_handles_labels()
        empty[0].legend(handles, labels, fontsize=10, loc="center", frameon=False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/ablation_lvae_metrics.{ext}", dpi=150, bbox_inches="tight")
        print(f"Saved: {OUT}/ablation_lvae_metrics.{ext}")
    plt.close()


# ── DDM-W ablation ────────────────────────────────────────────────────────────

def plot_ddmw():
    METRICS = {
        "shape_mse":             "Shape MSE",
        "aoa_mse":               "AoA MSE",
        "vendi":                 "Vendi Score",
        "pressure_mse":          "Pressure MSE",
        "mmd_w":                 "Latent Coverage MMD",
        "volume_constraint_sat": "Volume Constraint Sat.",
    }

    # Per-slice DDM (baseline) ablation — 4 individual panels
    BASELINE_DIR = "results/evaluation"
    BASELINE_METRICS = [
        ("mmd",       "MMD (Wing Shape)"),
        ("aoa_mse",   "MSE (Angle of Attack)"),
        ("vendi",     "Vendi Score"),
        ("shape_mse", "Shape MSE"),
    ]

    by_n_baseline = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(BASELINE_DIR, "eval_n*_s*.json"))):
        m = re.search(r"eval_n(\d+)_s(\d+)\.json", os.path.basename(path))
        if not m:
            continue
        n = int(m.group(1))
        with open(path) as f:
            d = json.load(f)
        by_n_baseline[n].append(d["results"])

    if by_n_baseline:
        for metric_key, ylabel in BASELINE_METRICS:
            sizes, means, stds = [], [], []
            for n in sorted(by_n_baseline):
                vals = [run[metric_key]["mean"] for run in by_n_baseline[n]
                        if metric_key in run]
                if vals:
                    sizes.append(n)
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
            if not sizes:
                continue
            means = np.array(means)
            stds  = np.array(stds)
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            ax.plot(sizes, means, marker="o", linewidth=1.8, color="steelblue")
            ax.fill_between(sizes, means - stds, means + stds, alpha=0.25, color="steelblue")
            ax.set_xlabel("Training dataset size", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xticks(sizes)
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            fig.tight_layout()
            fname = f"ddm_ablation_{metric_key}"
            fig.savefig(f"{OUT}/{fname}.pdf", dpi=150, bbox_inches="tight")
            print(f"Saved: {OUT}/{fname}.pdf")
            plt.close()

    # DDM-W joint vs geom-only — 6-panel comparison
    JOINT_DIR = "results/ddm_w_3d_ablation_v29"
    GEOM_DIR  = "results/ddm_w_3d_ablation_geom_only"
    C_JOINT   = "steelblue"
    C_GEOM    = "darkorange"
    FONT_LABEL  = 13
    FONT_TICK   = 11
    FONT_LEGEND = 12
    FONT_XLABEL = 13

    by_n_joint = load_eval_jsons(JOINT_DIR)
    by_n_geom  = load_eval_jsons(GEOM_DIR)

    fig, axes = plt.subplots(2, 3, figsize=(3 * 4.8, 2 * 3.8))
    axes = axes.flatten()

    for ax, metric_key in zip(axes, METRICS.keys()):
        plotted = False

        sizes_j, means_j, stds_j = get_series(by_n_joint, metric_key)
        if len(sizes_j):
            ax.plot(sizes_j, means_j, marker="o", linewidth=1.8,
                    color=C_JOINT, label="Joint (geometry + pressure)")
            ax.fill_between(sizes_j, means_j - stds_j, means_j + stds_j,
                             alpha=0.25, color=C_JOINT)
            plotted = True

        if metric_key != "pressure_mse":
            sizes_g, means_g, stds_g = get_series(by_n_geom, metric_key)
            if len(sizes_g):
                ax.plot(sizes_g, means_g, marker="o", linewidth=1.8,
                        color=C_GEOM, label="Geometry-only")
                ax.fill_between(sizes_g, means_g - stds_g, means_g + stds_g,
                                 alpha=0.25, color=C_GEOM)
                plotted = True

        if not plotted:
            ax.set_visible(False)
            continue

        all_sizes = sorted(set(sizes_j.tolist()) | (
            set(sizes_g.tolist()) if metric_key != "pressure_mse" else set()
        ))
        ax.set_ylabel(METRICS[metric_key], fontsize=FONT_LABEL)
        ax.set_xticks(all_sizes)
        ax.tick_params(axis="x", rotation=45, labelsize=FONT_TICK)
        ax.tick_params(axis="y", labelsize=FONT_TICK)

    fig.supxlabel("Training dataset size", fontsize=FONT_XLABEL, y=0.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               fontsize=FONT_LEGEND, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    fig.savefig(f"{OUT}/ablation_compare_metrics.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}/ablation_compare_metrics.png")
    plt.close()

    # DDM-W ablation wing surfaces (ground truth + generated at selected n)
    # Requires GPU/model — run separately if needed.
    # See plot_ablation_wing3d_surface logic (now integrated here as a note).


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--component", required=True, choices=["bae", "lvae", "ddmw"],
                   help="Which ablation to plot: bae | lvae | ddmw")
    args = p.parse_args()

    if args.component == "bae":
        plot_bae()
    elif args.component == "lvae":
        plot_lvae()
    elif args.component == "ddmw":
        plot_ddmw()


if __name__ == "__main__":
    main()
