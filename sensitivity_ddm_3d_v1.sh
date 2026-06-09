#!/bin/bash
#SBATCH --job-name=sens_ddm_3d_v1
#SBATCH --output=logs/sensitivity_ddm_3d_v1_%j.out
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.analysis.sensitivity_analysis_3d \
    --model          ddm_3d \
    --checkpoint     results/ddm_3d/ddm_3d_v1_best.pth \
    --bae_checkpoint results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt \
    --n_anchors 10 \
    --n_inits   10 \
    --n_passes  3 \
    --out_dir   results/sensitivity_3d
