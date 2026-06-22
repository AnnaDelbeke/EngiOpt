#!/bin/bash
#SBATCH --job-name=sens_ddm_w_v29_n30
#SBATCH --output=logs/sensitivity_ddm_w_3d_v29_n30_%j.out
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.analysis.sensitivity_analysis_3d \
    --model           ddm_w \
    --checkpoint      results/ddm_w_3d/ddm_w_3d_v29_best.pth \
    --bae_checkpoint  results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --lvae_checkpoint results/lvae_3d/lvae_3d_v29_best.pth \
    --n_anchors 10 \
    --n_inits   30 \
    --n_passes  3 \
    --out_dir   results/sensitivity_3d
