#!/bin/bash
#SBATCH --job-name=sens_regime_gifs
#SBATCH --output=logs/plot_sensitivity_regime_gifs_%j.out
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u plot_sensitivity_regime_gifs.py \
    --json    results/sensitivity_3d/sensitivity_ddm_w_20260610_094143.json \
    --out_dir results/sensitivity_3d/regime_gifs
