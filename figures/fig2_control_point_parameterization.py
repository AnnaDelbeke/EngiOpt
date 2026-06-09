"""
Figure 2: Control Point Parameterization and Monotonicity Loop
Side-by-side comparison of unconstrained baseline vs. proposed monotonic 3D parameterization.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

CORP_BLUE = "#1F4E79"
CORP_ORANGE = "#C55A11"
UPPER_GREEN = "#2ECC71"
LOWER_BLUE = "#2980B9"
ANCHOR_RED = "#E74C3C"
POLY_GRAY = "#7F8C8D"

# ── shared NACA-0012-ish airfoil outline ──────────────────────────────────────
def naca_0012(n=300):
    t = 0.12
    x = np.linspace(0, 1, n)
    yt = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)
    xu = np.concatenate([x[::-1], x[1:]])
    yu = np.concatenate([yt[::-1], -yt[1:]])
    return xu, yu


xu, yu = naca_0012()

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
fig.patch.set_facecolor("white")

# ═══════════════════════════════════════════════════════════════════════════════
# SUBPLOT A  —  Unconstrained Baseline 2D
# ═══════════════════════════════════════════════════════════════════════════════
ax = axes[0]
ax.set_facecolor("#FAFAFA")

ax.plot(xu, yu, color=CORP_BLUE, lw=2.0, zorder=3, label="Airfoil contour")
ax.fill(xu, yu, alpha=0.07, color=CORP_BLUE, zorder=2)

# 32 control points – intentionally erratic / zig-zaggy
rng = np.random.default_rng(42)

# upper surface control points with zig-zags
x_upper_ctrl = np.sort(rng.uniform(0.02, 0.98, 16))
# enforce some crossovers/zig-zags in y
y_upper_base = 0.05 + 0.10 * np.sqrt(x_upper_ctrl) * (1 - x_upper_ctrl)
perturbation = rng.uniform(-0.04, 0.04, 16)
# make a few points deliberately cross the airfoil surface to look "erratic"
perturbation[[3, 7, 11]] = [0.06, -0.05, 0.07]
y_upper_ctrl = y_upper_base + perturbation

# lower surface control points with zig-zags
x_lower_ctrl = np.sort(rng.uniform(0.02, 0.98, 16))
y_lower_base = -(0.04 + 0.09 * np.sqrt(x_lower_ctrl) * (1 - x_lower_ctrl))
perturbation2 = rng.uniform(-0.04, 0.04, 16)
perturbation2[[4, 9, 13]] = [-0.06, 0.05, -0.07]
y_lower_ctrl = y_lower_base + perturbation2

# plot dashed bounding polygon (upper)
ax.plot(x_upper_ctrl, y_upper_ctrl, "--", color=POLY_GRAY, lw=1.0, zorder=4, alpha=0.8)
ax.plot(x_lower_ctrl, y_lower_ctrl, "--", color=POLY_GRAY, lw=1.0, zorder=4, alpha=0.8)

ax.scatter(x_upper_ctrl, y_upper_ctrl, s=38, color=CORP_ORANGE, zorder=5,
           edgecolors="white", linewidths=0.6, label="Control points")
ax.scatter(x_lower_ctrl, y_lower_ctrl, s=38, color=CORP_ORANGE, zorder=5,
           edgecolors="white", linewidths=0.6)

# fixed anchor at trailing edge (1, 0)
ax.scatter([1.0], [0.0], s=80, color=ANCHOR_RED, zorder=6,
           edgecolors="white", linewidths=1.0, marker="D", label="Anchor (1, 0)")
ax.annotate("Trailing edge\nanchor (1, 0)", xy=(1.0, 0.0), xytext=(0.78, 0.10),
            fontsize=8, color=ANCHOR_RED,
            arrowprops=dict(arrowstyle="->", color=ANCHOR_RED, lw=1.2),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=ANCHOR_RED, alpha=0.85))

# zig-zag callout annotation
ax.annotate("Erratic vertical\ncrossings", xy=(x_upper_ctrl[7], y_upper_ctrl[7]),
            xytext=(0.55, 0.22),
            fontsize=8, color=POLY_GRAY,
            arrowprops=dict(arrowstyle="->", color=POLY_GRAY, lw=1.0),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=POLY_GRAY, alpha=0.85))

ax.set_xlim(-0.12, 1.12)
ax.set_ylim(-0.25, 0.30)
ax.set_xlabel("x/c  (normalised chord)")
ax.set_ylabel("y/c")
ax.set_title("(A)  Unconstrained Baseline Parameterisation\n"
             "Unrestricted control-point search envelope", pad=10, color="#2C3E50")
ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
ax.set_aspect("equal")
ax.grid(True, linestyle=":", linewidth=0.5, color="#CCCCCC", alpha=0.8)


# ═══════════════════════════════════════════════════════════════════════════════
# SUBPLOT B  —  Proposed Monotonic 3D Approach
# ═══════════════════════════════════════════════════════════════════════════════
ax2 = axes[1]
ax2.set_facecolor("#FAFAFA")

ax2.plot(xu, yu, color=CORP_BLUE, lw=2.0, zorder=3, label="Airfoil contour")
ax2.fill(xu, yu, alpha=0.07, color=CORP_BLUE, zorder=2)

# Leading edge extension limit
LE_LIMIT = -0.05
ax2.axvline(LE_LIMIT, color="#BDC3C7", lw=1.5, linestyle="--", zorder=2)
ax2.text(LE_LIMIT - 0.002, 0.18, "LE limit\nx = −0.05", fontsize=7.5,
         color="#7F8C8D", ha="right", va="top")

N = 15  # number of control points per surface

# Upper surface: monotonically decreasing x from (1,0) to LE_LIMIT
x_up = np.linspace(1.0, LE_LIMIT, N)
y_up_base = 0.055 * np.sqrt(np.clip(x_up, 0, None)) * (1 - np.clip(x_up, 0, 1))
y_up_base = np.where(x_up < 0, 0.005, y_up_base)
# smooth, well-behaved – no erratic perturbation
y_up = y_up_base + np.linspace(0.005, 0.0, N)

# Lower surface: monotonically increasing x from LE_LIMIT to (1,0)
x_lo = np.linspace(LE_LIMIT, 1.0, N)
y_lo_base = -(0.055 * np.sqrt(np.clip(x_lo, 0, None)) * (1 - np.clip(x_lo, 0, 1)))
y_lo_base = np.where(x_lo < 0, -0.005, y_lo_base)
y_lo = y_lo_base - np.linspace(0.005, 0.0, N)

# Polygon lines
ax2.plot(x_up, y_up, "-", color=UPPER_GREEN, lw=1.4, zorder=4, alpha=0.85)
ax2.plot(x_lo, y_lo, "-", color=LOWER_BLUE, lw=1.4, zorder=4, alpha=0.85)
# Close the loop at TE and LE
ax2.plot([x_up[-1], x_lo[0]], [y_up[-1], y_lo[0]], "-", color="#8E44AD", lw=1.2,
         zorder=4, alpha=0.6)  # LE connection
ax2.plot([x_up[0], x_lo[-1]], [y_up[0], y_lo[-1]], "-", color=ANCHOR_RED, lw=1.5,
         zorder=4, alpha=0.8)  # TE closure

# Scatter control points
ax2.scatter(x_up, y_up, s=42, color=UPPER_GREEN, zorder=6, edgecolors="white",
            linewidths=0.6, label=f"Upper surface (n={N})")
ax2.scatter(x_lo, y_lo, s=42, color=LOWER_BLUE, zorder=6, edgecolors="white",
            linewidths=0.6, label=f"Lower surface (n={N})")

# Anchor at TE
ax2.scatter([1.0], [0.0], s=90, color=ANCHOR_RED, zorder=7, edgecolors="white",
            linewidths=1.0, marker="D", label="Shared anchor (1, 0)")

# Directional arrows showing the cumulative sum loop
arrow_kw = dict(arrowstyle="-|>", color=UPPER_GREEN, lw=1.5,
                mutation_scale=12, zorder=8)
for i in [1, 5, 9]:
    ax2.annotate("", xy=(x_up[i], y_up[i]), xytext=(x_up[i - 1], y_up[i - 1]),
                 arrowprops=dict(**arrow_kw))

arrow_kw_lo = dict(arrowstyle="-|>", color=LOWER_BLUE, lw=1.5,
                   mutation_scale=12, zorder=8)
for i in [1, 5, 9]:
    ax2.annotate("", xy=(x_lo[i], y_lo[i]), xytext=(x_lo[i - 1], y_lo[i - 1]),
                 arrowprops=dict(**arrow_kw_lo))

# Monotonicity labels
ax2.text(0.50, 0.14, "Monotonic ←", fontsize=8.5, color=UPPER_GREEN,
         fontweight="bold", ha="center",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=UPPER_GREEN, alpha=0.85))
ax2.text(0.50, -0.13, "Monotonic →", fontsize=8.5, color=LOWER_BLUE,
         fontweight="bold", ha="center",
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=LOWER_BLUE, alpha=0.85))

# cumsum loop annotation
ax2.annotate("Cumulative-sum\nloop: Δxₛ > 0 enforced",
             xy=(0.22, 0.07), xytext=(0.30, 0.22),
             fontsize=7.5, color="#2C3E50",
             arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.0),
             bbox=dict(boxstyle="round,pad=0.2", fc="#FDFEFE", ec="#BDC3C7", alpha=0.92))

ax2.set_xlim(-0.12, 1.12)
ax2.set_ylim(-0.25, 0.30)
ax2.set_xlabel("x/c  (normalised chord)")
ax2.set_ylabel("y/c")
ax2.set_title("(B)  Proposed Monotonic-Loop Parameterisation\n"
              "Strictly ordered control-point sequence", pad=10, color="#2C3E50")
ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
ax2.set_aspect("equal")
ax2.grid(True, linestyle=":", linewidth=0.5, color="#CCCCCC", alpha=0.8)

fig.suptitle("Control Point Parameterisation: Baseline vs. Proposed Monotonic-Loop",
             fontsize=14, fontweight="bold", color="#1A252F", y=1.01)

plt.savefig("fig2_control_point_parameterization.pdf", bbox_inches="tight", dpi=200)
plt.savefig("fig2_control_point_parameterization.png", bbox_inches="tight", dpi=200)
print("Saved fig2_control_point_parameterization.pdf / .png")
plt.show()
