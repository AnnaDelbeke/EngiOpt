"""Generate sensitivity plots from saved JSON files (no GPU/model needed)."""
import json, sys, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

def plot_sensitivity(results, save_path):
    labels   = [f"case {r['case_num']}\nM={r['mach']:.2f}" for r in results]
    mses     = [r['pairwise_shape_mse'] for r in results]
    aoa_stds = [r['aoa_std'] for r in results]
    colors   = []
    for r in results:
        m = r['mach']
        if m < 0.8:   colors.append('steelblue')
        elif m < 1.0: colors.append('darkorange')
        else:         colors.append('firebrick')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.bar(labels, mses, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel("Mean pairwise shape MSE", fontsize=10)
    ax.set_title("Initialization sensitivity - shape", fontsize=11)
    ax.tick_params(axis='x', labelsize=7)
    legend_elements = [
        Patch(facecolor='steelblue',  label='Subsonic (M<0.8)'),
        Patch(facecolor='darkorange', label='Transonic (0.8-1.0)'),
        Patch(facecolor='firebrick',  label='Supersonic (M>=1.0)'),
    ]
    ax.legend(handles=legend_elements, fontsize=8)

    ax = axes[1]
    ax.bar(labels, aoa_stds, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel("AoA std across initializations (deg)", fontsize=10)
    ax.set_title("Initialization sensitivity - AoA", fontsize=11)
    ax.tick_params(axis='x', labelsize=7)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.92, wspace=0.3)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  saved {save_path}")


def plot_sensitivity_by_regime(results, save_path):
    regimes = {
        'Subsonic (M<0.8)':    [r['pairwise_shape_mse'] for r in results if r['mach'] < 0.8],
        'Transonic (0.8-1.0)': [r['pairwise_shape_mse'] for r in results if 0.8 <= r['mach'] < 1.0],
        'Supersonic (M>=1.0)': [r['pairwise_shape_mse'] for r in results if r['mach'] >= 1.0],
    }
    fig, ax = plt.subplots(figsize=(7, 5))
    data   = [v for v in regimes.values() if v]
    labels = [k for k, v in regimes.items() if v]
    colors = ['steelblue', 'darkorange', 'firebrick'][:len(data)]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Mean pairwise shape MSE", fontsize=10)
    ax.set_title("Sensitivity by flow regime", fontsize=11)
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.92)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  saved {save_path}")


json_files = [
    "results/sensitivity_3d/sensitivity_ddm_3d_20260527_074813.json",
    "results/sensitivity_3d/sensitivity_ddm_w_20260527_074815.json",
]

for jf in json_files:
    if not os.path.exists(jf):
        print(f"Missing: {jf}")
        continue
    with open(jf) as f:
        data = json.load(f)
    results = data["per_anchor"]
    stem = jf.replace(".json", "")
    print(f"\nProcessing {jf}  ({len(results)} anchors, model={data['model']})")
    plot_sensitivity(results,           stem + "_bars.png")
    plot_sensitivity_by_regime(results, stem + "_regime.png")

print("\nDone.")
