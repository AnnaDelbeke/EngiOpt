#!/bin/bash
#SBATCH --job-name=eval_ddm_w_3d_v1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.24h
#SBATCH --output=/cluster/home/adelbeke/EngiOpt/logs/eval_ddm_w_3d_v1_%j.out

module purge
module load stack/2024-06 python/3.10.13
cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate
export HF_DATASETS_OFFLINE=1
python -u -m engiopt.ddm.ddm_w.evaluate_ddm_w_3d \
    --checkpoint      results/ddm_w_3d/ddm_w_3d_v1_best.pth \
    --bae_checkpoint  results/bezier_ae_3d/run_014/models/bezier_ae_3d_best.pt \
    --lvae_checkpoint results/lvae_3d/lvae_3d_v5_best.pth \
    --n_passes        10 \
    --out_dir         results/evaluation
