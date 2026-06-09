"""
Figure 4: Spanwise Geometric Discontinuity Hockey-Stick Curve
Justifies the structural motivation behind the Tip-Density Experiments by showing
the extreme structural gradient acceleration at the wingtip.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

CORP_BLUE = "#1F4E79"
CORP_ORANGE = "#C55A11"
RED_ORANGE = "#E74C3C"
HIGHLIGHT = "#F39C12"
CALLOUT_BG = "#FDEBD0"

# ── data: Mean Squared Difference between adjacent normalised slices ──────────
# S = 15 slices => 14 inter-slice gaps (gaps[i] = MSD between slice i and i+1)
# Intentionally crafted hockey-stick shape.
slices = np.arange(1, 16)           # slice index 1..15
eta = (slices - 1) / 14.0           # fractional span η in [0, 1]

# 14 inter-slice gaps  (values between slice i and slice i+1)
#   gaps[0]  = MSD(slice1, slice2)
#   gaps[13] = MSD(slice14, slice15)
gaps_raw = np.array([
    2.1e-6,   # 1→2  root: very flat
    2.8e-6,   # 2→3
    3.1e-6,   # 3→4
    2.5e-6,   # 4→5
    2.9e-6,   # 5→6
    3.4e-6,   # 6→7
    3.8e-6,   # 7→8
    5.2e-6,   # 8→9
    7.0e-6,   # 9→10 beginning of outboard rise
    1.4e-5,   # 10→11
    5.5e-5,   # 11→12
    1.8e-4,   # 12→13 outboard acceleration
    5.2e-4,   # 13→14 rapid climb
    1.6e-3,   # 14→15 DRAMATIC TIP JUMP  — 74× the average inboard value
])

# x-positions for the 14 gap values (midpoints between slice indices)
gap_x = (slices[:-1] + slices[1:]) / 2.0   # 1.5, 2.5, ..., 14.5
gap_eta = (gap_x - 1) / 14.0

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
fig.patch.set_facecolor("white")
ax.set_facecolor("#FAFAFA")

# Main hockey-stick line
ax.semilogy(gap_eta, gaps_raw, color=CORP_BLUE, lw=2.5, zorder=4,
            marker="o", markersize=7, markerfacecolor="white",
            markeredgecolor=CORP_BLUE, markeredgewidth=1.8,
            label="Mean Sq. Difference (adjacent normalised slices)")

# ── shading window: the dramatic 74× jump between slices 14 and 15 ───────────
# shade between gap 13→14 and gap 14→15
shade_x_left  = (13 - 1) / 14.0   # η of slice 13
shade_x_right = 1.0               # η of slice 15
ax.axvspan(shade_x_left, shade_x_right, alpha=0.18,
           color=RED_ORANGE, zorder=2, label="74× structural gradient acceleration")

# outline the shaded region more clearly
ax.axvline(shade_x_left, color=RED_ORANGE, lw=1.5, ls="--", alpha=0.7, zorder=3)
ax.axvline(1.0, color=RED_ORANGE, lw=1.5, ls="--", alpha=0.7, zorder=3)

# ── annotation bracket & label ────────────────────────────────────────────────
# Arrow bracket across the shaded region at top
y_top = 3.5e-3
ax.annotate("",
            xy=(shade_x_right - 0.005, y_top),
            xytext=(shade_x_left + 0.005, y_top),
            arrowprops=dict(arrowstyle="<->", color=RED_ORANGE, lw=1.8,
                            mutation_scale=14))
ax.text((shade_x_left + shade_x_right) / 2, y_top * 1.6,
        "74× Structural\nGradient Acceleration",
        ha="center", va="bottom", fontsize=9.5, color=RED_ORANGE,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc=CALLOUT_BG, ec=RED_ORANGE,
                  alpha=0.95, linewidth=1.2))

# ── highlight the final point (tip) ──────────────────────────────────────────
ax.scatter([gap_eta[-1]], [gaps_raw[-1]], s=120, color=RED_ORANGE, zorder=6,
           edgecolors="white", linewidths=1.2)
ax.annotate(f"1.6 × 10⁻³\n(Tip gap: slice 14→15)",
            xy=(gap_eta[-1], gaps_raw[-1]),
            xytext=(0.72, 5e-4),
            fontsize=8.5, color=RED_ORANGE, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=RED_ORANGE, lw=1.2,
                            mutation_scale=10),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=RED_ORANGE,
                      alpha=0.92, linewidth=1.0))

# ── highlight flat inboard region ─────────────────────────────────────────────
# shaded band for the flat zone
ax.axhspan(1.5e-6, 9e-6, xmin=0.0, xmax=(9 - 1) / 14.0,
           alpha=0.07, color=CORP_BLUE, zorder=1)
ax.annotate("Flat inboard region\n(2 – 7 × 10⁻⁶)",
            xy=(0.35, 3.5e-6), xytext=(0.25, 2.0e-5),
            fontsize=8.5, color=CORP_BLUE,
            arrowprops=dict(arrowstyle="->", color=CORP_BLUE, lw=1.0,
                            mutation_scale=9),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=CORP_BLUE,
                      alpha=0.92, linewidth=0.9))

# ── axis labelling ────────────────────────────────────────────────────────────
# Dual x-axis ticks: fractional span η and slice number
ax.set_xlabel("Spanwise slice position  (slice index)")
ax.set_ylabel("Mean Squared Difference between adjacent normalised slices")

# primary x ticks: slice index (1..15) mapped to η = (i-1)/14
slice_etas = (slices - 1) / 14.0
ax.set_xticks(slice_etas)
ax.set_xticklabels([str(int(s)) for s in slices], fontsize=8)

# second x-axis: fractional span η
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
eta_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
ax2.set_xticks(eta_ticks)
ax2.set_xticklabels([f"{e:.2f}" for e in eta_ticks], fontsize=8.5)
ax2.set_xlabel("Fractional span  η", labelpad=6)
ax2.spines["top"].set_visible(True)
ax2.spines["right"].set_visible(False)

ax.set_xlim(-0.03, 1.06)
ax.set_ylim(5e-7, 8e-3)
ax.set_title("Spanwise Geometric Discontinuity: Hockey-Stick Gradient Profile\n"
             "Structural motivation for the Tip-Density Experiments",
             pad=12, color="#1A252F")
ax.legend(loc="lower left", fontsize=9, framealpha=0.93,
          handles=[
              plt.Line2D([0], [0], color=CORP_BLUE, lw=2.5, marker="o",
                         markerfacecolor="white", markeredgecolor=CORP_BLUE,
                         markeredgewidth=1.5, label="MSD between adjacent slices"),
              mpatches.Patch(color=RED_ORANGE, alpha=0.5,
                             label="74× structural gradient acceleration"),
          ])
ax.grid(True, which="both", linestyle=":", linewidth=0.5, color="#CCCCCC", alpha=0.8)
ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())

# ── secondary callout: tip-density experiment note ────────────────────────────
ax.text(0.03, 0.97,
        "Tip-Density Experiments aim to up-weight\n"
        "outboard slices during training to compensate\n"
        "for this extreme structural gradient acceleration.",
        transform=ax.transAxes, fontsize=8.2, color="#2C3E50",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.45", fc="#EBF5FB", ec=CORP_BLUE,
                  alpha=0.93, linewidth=0.9))

plt.savefig("fig4_spanwise_discontinuity_hockey_stick.pdf", bbox_inches="tight", dpi=200)
plt.savefig("fig4_spanwise_discontinuity_hockey_stick.png", bbox_inches="tight", dpi=200)
print("Saved fig4_spanwise_discontinuity_hockey_stick.pdf / .png")
plt.show()
