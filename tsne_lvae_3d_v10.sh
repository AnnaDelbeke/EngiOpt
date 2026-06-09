#!/bin/bash
#SBATCH --job-name=tsne_lvae_3d_v10
#SBATCH --output=logs/tsne_lvae_3d_v10_%j.out
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=32G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.lvae.plot_latent_tsne \
    --checkpoint     results/lvae_3d/lvae_3d_v10_best.pth \
    --bae_checkpoint results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt \
    --save_dir       results/lvae_3d_evaluation \
    --perplexity     30
