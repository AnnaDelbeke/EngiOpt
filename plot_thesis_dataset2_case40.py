"""
Thesis figures: Dataset 2, Case 40 — initial vs final wing, 15 spanwise slices.
Produces two PDFs for LaTeX inclusion:
  1. Uniform slice sampling
  2. Tip-biased slice sampling (5 inboard + 10 outboard)
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

CASE_NUM = 40
ROWS, COLS = 5, 3

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         8,
    "axes.titlesize":    7.5,
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})


def make_figure(df, case_num, title, out_path):
    df_case = df[df["case_num"] == float(case_num)]
    slice_nums    = sorted(df_case["slice_num"].unique())
    slice_initial = int(min(slice_nums))
    slice_final   = int(max(slice_nums))

    df_init  = df_case[df_case["slice_num"] == slice_initial]
    df_final = df_case[df_case["slice_num"] == slice_final]

    sub_slices = sorted(df_init["sub_slice_num"].unique())
    etas       = [df_init[df_init["sub_slice_num"] == s]["eta"].iloc[0] for s in sub_slices]
    n_slices   = len(sub_slices)
    assert n_slices == 15, f"Expected 15 sub-slices, got {n_slices}"

    # Per-slice y-limits: same range for initial and final so panels are comparable
    slice_ylims = {}
    for ss in sub_slices:
        y = np.concatenate([
            df_init[df_init["sub_slice_num"] == ss]["CoordinateY"].values,
            df_final[df_final["sub_slice_num"] == ss]["CoordinateY"].values,
        ])
        spread = y.max() - y.min()
        slice_ylims[ss] = (y.min() - spread * 0.1, y.max() + spread * 0.1)

    fig = plt.figure(figsize=(14, 7.5))
    outer = gridspec.GridSpec(
        1, 2, figure=fig,
        left=0.06, right=0.98,
        top=0.86, bottom=0.08,
        wspace=0.14,
    )

    for p, (df_panel, color, panel_title) in enumerate(zip(
        [df_init, df_final],
        ["#2166ac", "#d6604d"],
        ["(a) Initial wing", "(b) Optimised wing"],
    )):
        inner = gridspec.GridSpecFromSubplotSpec(
            ROWS, COLS, subplot_spec=outer[p],
            hspace=0.45, wspace=0.18,
        )
        for i, (ss, eta) in enumerate(zip(sub_slices, etas)):
            row, col = divmod(i, COLS)
            ax = fig.add_subplot(inner[row, col])

            df_s = df_panel[df_panel["sub_slice_num"] == ss]
            ax.plot(df_s["CoordinateX"], df_s["CoordinateY"],
                    "-", color=color, linewidth=0.9)

            ax.set_title(f"$\\eta = {eta:.2f}$", pad=3)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(slice_ylims[ss])
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
            ax.tick_params(labelsize=6, length=2)

            if row < ROWS - 1:
                ax.set_xticklabels([])
            if col > 0:
                ax.set_yticklabels([])

            if i == 0:
                ax.text(0.05, 0.75, "root", transform=ax.transAxes,
                        color="dimgray", fontsize=6, style="italic")
            if i == n_slices - 1:
                ax.text(0.05, 0.75, "tip", transform=ax.transAxes,
                        color="dimgray", fontsize=6, style="italic")

        mid_x = outer[p].get_position(fig).x0 + outer[p].get_position(fig).width / 2
        fig.text(mid_x, 0.895, panel_title,
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    fig.text(0.52, 0.03, r"$x/c$ (normalised chord)", ha="center", fontsize=10)
    fig.text(0.015, 0.47, r"$y/c$ (normalised thickness)", va="center",
             rotation="vertical", fontsize=10)
    fig.suptitle(title, fontsize=11, y=0.955)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── 1. Uniform sampling ───────────────────────────────────────────────────────
print("Loading uniform dataset...")
with open("Wing_TL/data/processed/new_dataset_slices.pkl", "rb") as f:
    df_uniform = pickle.load(f)

make_figure(
    df_uniform, CASE_NUM,
    title="Dataset 2 — Case 40: uniform sampling, 15 equally spaced $\\eta$ positions",
    out_path="thesis/figures/dataset2_case40_initial_final.pdf",
)

# ── 2. Tip-biased sampling ────────────────────────────────────────────────────
print("Loading tip-biased dataset...")
with open("Wing_TL/data/processed/new_dataset_tip_biased_slices.pkl", "rb") as f:
    df_tip = pickle.load(f)

make_figure(
    df_tip, CASE_NUM,
    title="Dataset 2 — Case 40: tip-biased sampling (5 inboard $+$ 10 outboard $\\eta$ positions)",
    out_path="thesis/figures/dataset2_case40_tip_biased_initial_final.pdf",
)

# ── 3. Custom eta sampling ────────────────────────────────────────────────────
print("Loading custom-eta dataset...")
with open("Wing_TL/data/processed/new_dataset_custom_etas_slices.pkl", "rb") as f:
    df_custom = pickle.load(f)

make_figure(
    df_custom, CASE_NUM,
    title="Dataset 2 — Case 40: custom $\\eta$ sampling (dense near tip)",
    out_path="thesis/figures/dataset2_case40_custom_etas_initial_final.pdf",
)
