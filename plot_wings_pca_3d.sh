#!/bin/bash
#SBATCH --job-name=plot_wings_pca_3d
#SBATCH --output=logs/plot_wings_pca_3d_%j.out
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.ddm.ddm_pca.plot_wings_pca_3d \
    --checkpoint     results/ddm_pca_3d/ddm_pca_3d_v2_best.pth \
    --bae_checkpoint results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --n_wings        5 \
    --out            results/plots/wings_ddm_pca_3d_v2.png
