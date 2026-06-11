#!/bin/bash
#SBATCH --job-name=compare_pca_ddm_w
#SBATCH --output=logs/compare_pca_vs_ddm_w_%j.out
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.analysis.compare_pca_vs_ddm_w \
    --ddm_pca_checkpoint  results/ddm_pca/ddm_pca_v1_best.pth \
    --ddm_w_checkpoint    results/ddm_w_3d/ddm_w_3d_v29_best.pth \
    --bae_3d_checkpoint   results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --lvae_checkpoint     results/lvae_3d/lvae_3d_v29_best.pth \
    --n_passes            10 \
    --out_dir             results/evaluation
