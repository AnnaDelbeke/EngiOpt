#!/bin/bash
#SBATCH --job-name=eval_ddm_3d_dropout
#SBATCH --output=logs/eval_ddm_3d_dropout_%j.out
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

for model in ddm_3d_do10 ddm_3d_do15 ddm_3d_do20 ddm_3d_do25; do
    echo "=== Evaluating $model ==="
    python -u -m engiopt.ddm.evaluate_ddm_3d \
        --checkpoint results/ddm_3d/${model}_best.pth \
        --bae_checkpoint results/bezier_ae_3d/run_040/models/bezier_ae_3d_best.pt \
        --n_passes 10 \
        --out_dir results/ddm_3d_evaluation
done
