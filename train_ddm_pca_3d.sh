#!/bin/bash
#SBATCH --job-name=train_ddm_pca_3d
#SBATCH --output=logs/train_ddm_pca_3d_%j.out
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.ddm.ddm_pca.train_ddm_pca_3d \
    --bae_checkpoint results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --model_name     ddm_pca_3d_v2 \
    --n_components   64 \
    --w_aoa          1.0
