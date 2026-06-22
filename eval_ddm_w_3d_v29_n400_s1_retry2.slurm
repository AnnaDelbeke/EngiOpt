#!/bin/bash
#SBATCH --job-name=ddm_w_3d_v29_eval_retry2
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --partition=gpu.4h
#SBATCH --exclude=eu-lo-g3-031,eu-lo-g2-023,eu-lo-g3-013,eu-lo-g3-026
#SBATCH --output=logs/ddm_w_3d_v29_eval_n400_s1_retry2_%j.out

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

python -u -m engiopt.ddm.ddm_w.evaluate_ddm_w_3d \
    --checkpoint      results/ddm_w_3d_ablation_v29/n400_s1/ddm_w_3d_ablation_n400_s1_best.pth \
    --bae_checkpoint  results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --lvae_checkpoint results/lvae_3d/lvae_3d_v29_best.pth \
    --out_dir         results/ddm_w_3d_ablation_v29/n400_s1
