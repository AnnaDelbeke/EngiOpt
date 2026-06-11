#!/bin/bash
#SBATCH --job-name=compare_wings_3d
#SBATCH --output=logs/compare_wings_3d_%j.out
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=16G

set -e
cd /cluster/home/adelbeke/EngiOpt
source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate

mkdir -p logs results/plots

python -m engiopt.analysis.compare_ddm_3d_vs_ddm_w \
    --ddm_w_checkpoint      results/ddm_w_3d/ddm_w_3d_v29_best.pth \
    --ddm_3d_checkpoint     results/ddm_3d/ddm_3d_run039_v1_best.pth \
    --bae_checkpoint        results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --lvae_checkpoint       results/lvae_3d/lvae_3d_v29_best.pth \
    --n_per_regime          1 \
    --seed                  0 \
    --out                   results/plots/compare_ddm_3d_vs_ddm_w.png
