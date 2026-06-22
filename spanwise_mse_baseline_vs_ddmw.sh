#!/bin/bash
#SBATCH --job-name=spanwise_mse_cmp
#SBATCH --output=logs/spanwise_mse_baseline_vs_ddmw_%j.out
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u plot_spanwise_mse_baseline_vs_ddmw.py \
    --ddmw_checkpoint  results/ddm_w_3d/ddm_w_3d_v29_best.pth \
    --bae_checkpoint   results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --lvae_checkpoint  results/lvae_3d/lvae_3d_v29_best.pth \
    --n_passes         10 \
    --out              results/plots/spanwise_mse_baseline_vs_ddmw.pdf
