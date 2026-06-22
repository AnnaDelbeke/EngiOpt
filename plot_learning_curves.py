"""
Learning curve plots — one PDF per DDM model, log-y scale.
Style: raw faint band + bold smoothed line, no grid, log y-axis.
Colours match thesis palette (serif font, blue train / green test).

2D DDM has train loss only (no val split in original training script).
"""

import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi":      150,
})

LOG_DIR = Path("logs")
OUT_DIR = Path("results/learning_curves")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COL_TRAIN = "#2166ac"   # thesis blue
COL_TEST  = "#4dac26"   # thesis green

# ── parsers ───────────────────────────────────────────────────────────────────

def parse_ddm2d(path):
    epochs, train, val = [], [], []
    pat = re.compile(r"Epoch\s+(\d+)/\d+\s+\|\s+Train Loss\s+([\d.]+)(?:\s+\|\s+Val Loss\s+([\d.]+))?")
    for line in Path(path).read_text().splitlines():
        m = pat.search(line)
        if m:
            epochs.append(int(m.group(1)))
            train.append(float(m.group(2)))
            if m.group(3):
                val.append(float(m.group(3)))
    v = np.array(val) if val else None
    return np.array(epochs), np.array(train), v

def parse_3d(path):
    epochs, train, val = [], [], []
    pat = re.compile(r"Epoch\s+(\d+)/\d+\s+\|\s+Train\s+([\d.]+).*?\|\s+Val\s+([\d.]+)")
    for line in Path(path).read_text().splitlines():
        m = pat.search(line)
        if m:
            epochs.append(int(m.group(1)))
            train.append(float(m.group(2)))
            val.append(float(m.group(3)))
    return np.array(epochs), np.array(train), np.array(val)

# ── smoothing ─────────────────────────────────────────────────────────────────

SMOOTH_TRAIN = 200
SMOOTH_TEST  = 500   # heavier smoothing — test is 98-sample single-batch, very noisy

def smooth(y, w):
    return np.convolve(y, np.ones(w) / w, mode="valid")

def sx(x, w):
    return x[w - 1:]

# ── model definitions ─────────────────────────────────────────────────────────

models = [
    dict(label="2D DDM",            slug="ddm_2d",           log=LOG_DIR / "train_ddm_2d_v7hope_val_3802943.out", parse=parse_ddm2d),
    dict(label="DDM-3D",            slug="ddm_3d",           log=LOG_DIR / "train_ddm_3d_run039_3464844.out",    parse=parse_3d),
    dict(label="DDM-PCA",           slug="ddm_pca",          log=LOG_DIR / "train_ddm_pca_3d_2878228.out",       parse=parse_3d),
    dict(label="DDM-W (geom-only)", slug="ddm_w_geom_only",  log=LOG_DIR / "ddm_w_3d_v29_geom_only_3440197.out", parse=parse_3d),
    dict(label="DDM-W (joint)",     slug="ddm_w_joint",      log=LOG_DIR / "train_ddm_w_3d_v29_2588069.out",     parse=parse_3d),
]

# ── one figure per model ──────────────────────────────────────────────────────

for cfg in models:
    epochs, train, val = cfg["parse"](cfg["log"])
    n_epochs = int(epochs[-1])

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    # training — faint raw + bold smoothed
    ax.semilogy(epochs, train, color=COL_TRAIN, lw=0.5, alpha=0.22)
    ax.semilogy(sx(epochs, SMOOTH_TRAIN), smooth(train, SMOOTH_TRAIN),
                color=COL_TRAIN, lw=1.8, label="Train Loss")

    # test — smoothed only (raw omitted: 98-sample eval is too noisy to plot raw)
    if val is not None:
        ax.semilogy(sx(epochs, SMOOTH_TEST), smooth(val, SMOOTH_TEST),
                    color=COL_TEST, lw=1.8, ls="--", label="Test Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xlim(0, n_epochs)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x)))
    )
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(axis='y', which='minor', length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    out_png = OUT_DIR / f"{cfg['slug']}.png"
    out_pdf = OUT_DIR / f"{cfg['slug']}.pdf"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_png}  |  {out_pdf}")
