#!/bin/bash
#SBATCH --job-name=sweep_ddm_w_3d_wp
#SBATCH --output=logs/sweep_ddm_w_3d_wp_%a_%j.out
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-3

WP_VALUES=(0.1 1.0 2.0 5.0)
wp=${WP_VALUES[$SLURM_ARRAY_TASK_ID]}
name="ddm_w_3d_v29_wp${wp/./}"

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

echo "=== Training $name (w_pressure=$wp) ==="
python -u -m engiopt.ddm.ddm_w.train_ddm_w_3d \
    --bae_checkpoint  results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --lvae_checkpoint results/lvae_3d/lvae_3d_v29_best.pth \
    --model_name "$name" \
    --w_aoa 15.0 \
    --w_pressure "$wp" \
    --lr 1e-4 \
    --wandb \
    --wandb_project engiopt-ddm-w-3d
