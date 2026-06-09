#!/bin/bash
#SBATCH --job-name=recon_errors
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.4h
#SBATCH --output=/cluster/home/adelbeke/EngiOpt/logs/recon_errors_%j.out

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1
python -u -m engiopt.ddm.plot_reconstruction_errors \
    --lvae_checkpoint results/lvae/lae_dropout_0.25_flow_only_best.pth \
    --pca_checkpoint  results/ddm_pca/ddm_pca_v1_pca.pkl \
    --seed 0
