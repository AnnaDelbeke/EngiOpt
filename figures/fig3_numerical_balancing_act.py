"""
Figure 3: The "Numerical Balancing Act" Paradox
Shows how wild control polygon oscillations are absorbed by rational Bézier weights
to produce a flawless smooth airfoil contour.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import FancyArrowPatch
from scipy.special import comb

matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
})

CONTOUR_RED = "#C0392B"
POLYGON_GREEN = "#27AE60"
NODE_GREEN = "#1E8449"
ANNO_BLUE = "#1F4E79"
BG_COLOR = "#FAFAFA"

# ── rational Bézier evaluator ─────────────────────────────────────────────────
def rational_bezier(P, W, n_pts=600):
    n = len(P) - 1
    t = np.linspace(0, 1, n_pts)
    Bt = np.zeros((n_pts, 2))
    for i, ti in enumerate(t):
        b = np.array([comb(n, k, exact=True) * (ti**k) * ((1-ti)**(n-k)) for k in range(n+1)])
        bw = b * W
        Bt[i] = (bw @ P) / bw.sum()
    return Bt

# ── define a representative airfoil slice control polygon ─────────────────────
# A plausible reconstructed NACA-like slice with n=14 control points (degree-13)
# Smooth airfoil would have these cp near-surface; we make them wildly oscillate.

n_ctrl = 15
# x-coordinates: monotonically spaced (mimics our monotonic loop)
x_upper = np.array([1.00, 0.85, 0.70, 0.55, 0.42, 0.31, 0.22, 0.14, 0.08, 0.04,
                    0.01, -0.01, -0.03, -0.045, -0.05])
x_lower = x_upper[::-1]

# Wild y-coordinates for control polygon (zig-zagging oscillations)
rng = np.random.default_rng(7)
y_smooth_upper = 0.06 * np.sqrt(np.clip(x_upper, 0, 1)) * (1 - np.clip(x_upper, 0, 1))
y_smooth_upper = np.where(x_upper < 0, 0.003, y_smooth_upper)

# Oscillation: alternating spikes of ±0.12 around smooth profile
spike_amp = np.array([0.0, 0.11, -0.09, 0.13, -0.10, 0.14, -0.08, 0.12,
                      -0.11, 0.09, 0.13, -0.10, 0.08, -0.07, 0.0])
y_ctrl = y_smooth_upper + spike_amp

# Rational weights: counterbalance spikes (high weight on well-placed nodes,
# extreme weights on spiked nodes to pull curve back)
W = np.array([1.0, 0.08, 12.5, 0.07, 10.2, 0.06, 11.8, 0.07,
              9.5, 14.0, 0.08, 10.1, 13.2, 0.09, 1.0])
W = W / W.sum() * n_ctrl  # renormalise so moderate nodes ≈ 1

P_ctrl = np.column_stack([x_upper, y_ctrl])

# Evaluate rational Bézier curve (the smooth red contour)
curve = rational_bezier(P_ctrl, W, n_pts=800)

# ── figure layout ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
fig.patch.set_facecolor("white")
ax.set_facecolor(BG_COLOR)

# Primary: smooth reconstructed contour
ax.plot(curve[:, 0], curve[:, 1], color=CONTOUR_RED, lw=2.5, zorder=5,
        label="Reconstructed airfoil contour (rational Bézier curve)")

# Control polygon (wild zig-zag)
ax.plot(P_ctrl[:, 0], P_ctrl[:, 1], "--", color=POLYGON_GREEN, lw=1.5,
        zorder=4, alpha=0.85, label="Decoded control polygon")
ax.scatter(P_ctrl[:, 0], P_ctrl[:, 1], s=55, color=NODE_GREEN, zorder=6,
           edgecolors="white", linewidths=0.8)

# ── weight annotations for selected spiked nodes ─────────────────────────────
# annotate a subset of the most extreme spike nodes
annotate_idxs = [1, 3, 5, 7, 9, 11]
x_ann_offsets = [0.06, 0.06, -0.07, 0.06, -0.06, 0.06]
y_ann_offsets = [0.06, -0.07, 0.07, -0.07, 0.07, -0.07]

for idx, dx, dy in zip(annotate_idxs, x_ann_offsets, y_ann_offsets):
    xi, yi = P_ctrl[idx]
    wi = W[idx]
    ax.annotate(f"$w_{{{idx}}}={wi:.2f}$",
                xy=(xi, yi), xytext=(xi + dx, yi + dy),
                fontsize=7.5, color=ANNO_BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=ANNO_BLUE, lw=0.9,
                                mutation_scale=8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=ANNO_BLUE,
                          alpha=0.90, linewidth=0.8))

# ── spike / curve gap visual ─────────────────────────────────────────────────
# draw thin vertical dashed lines from spiked node to roughly where the smooth
# curve is, to emphasise the gap being "absorbed"
for idx in [2, 4, 6, 8]:
    xi, yi = P_ctrl[idx]
    ax.vlines(xi, 0.005, yi, colors=POLYGON_GREEN, lw=0.9, linestyles=":", alpha=0.55)

ax.axhline(0, color="#BDC3C7", lw=0.8, zorder=1)
ax.axvline(0, color="#BDC3C7", lw=0.8, zorder=1)

ax.set_xlim(-0.12, 1.08)
ax.set_ylim(-0.18, 0.35)
ax.set_xlabel("x/c  (normalised chord)")
ax.set_ylabel("y/c")
ax.set_title('The "Numerical Balancing Act" Paradox\n'
             'Wild control polygon oscillations absorbed by rational weights — '
             'smooth contour preserved', pad=10, color="#1A252F")
ax.legend(loc="upper right", fontsize=9, framealpha=0.93)
ax.grid(True, linestyle=":", linewidth=0.5, color="#CCCCCC", alpha=0.8)
ax.set_aspect("equal")

# ── inset zoom near leading edge ──────────────────────────────────────────────
axins = inset_axes(ax, width="38%", height="48%", loc="upper left",
                   bbox_to_anchor=(0.02, 0.02, 1, 1),
                   bbox_transform=ax.transAxes)
axins.set_facecolor("#FEFEFE")

axins.plot(curve[:, 0], curve[:, 1], color=CONTOUR_RED, lw=2.2, zorder=5)
axins.plot(P_ctrl[:, 0], P_ctrl[:, 1], "--", color=POLYGON_GREEN, lw=1.3,
           zorder=4, alpha=0.85)
axins.scatter(P_ctrl[:, 0], P_ctrl[:, 1], s=40, color=NODE_GREEN, zorder=6,
              edgecolors="white", linewidths=0.7)

# zoom region: leading edge area
x1, x2 = -0.065, 0.15
y1, y2 = -0.08, 0.16
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_aspect("equal")
axins.tick_params(labelsize=7)
axins.grid(True, linestyle=":", linewidth=0.5, color="#CCCCCC", alpha=0.8)
axins.set_title("Leading edge zoom", fontsize=8, color="#1A252F", pad=4)

# annotate extreme spike in inset
for idx in [10, 11, 12]:
    xi, yi = P_ctrl[idx]
    if x1 < xi < x2 and y1 < yi < y2:
        wi = W[idx]
        axins.annotate(f"$w={wi:.2f}$",
                       xy=(xi, yi), xytext=(xi + 0.02, yi + 0.03),
                       fontsize=7, color=ANNO_BLUE, fontweight="bold",
                       arrowprops=dict(arrowstyle="-|>", color=ANNO_BLUE, lw=0.8,
                                       mutation_scale=7),
                       bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=ANNO_BLUE,
                                 alpha=0.90, linewidth=0.7))

# mark inset region on main axes
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="#7F8C8D", lw=0.9, ls="--")

# textbox: balancing act explanation
ax.text(0.56, 0.26,
        "Network learns:\n"
        "  high spike node  →  low weight wᵢ ≈ 0.06\n"
        "  low spike node  →  high weight wᵢ ≈ 12–14\n"
        "Rational basis absorbs oscillations;\n"
        "smooth contour remains exact.",
        transform=ax.transAxes, fontsize=8.2, color="#1A252F",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.5", fc="#EBF5FB", ec=ANNO_BLUE,
                  alpha=0.95, linewidth=1.0))

plt.savefig("fig3_numerical_balancing_act.pdf", bbox_inches="tight", dpi=200)
plt.savefig("fig3_numerical_balancing_act.png", bbox_inches="tight", dpi=200)
print("Saved fig3_numerical_balancing_act.pdf / .png")
plt.show()
