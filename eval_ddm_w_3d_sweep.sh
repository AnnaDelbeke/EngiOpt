#!/bin/bash
#SBATCH --job-name=eval_ddm_w_3d_sweep
#SBATCH --output=logs/eval_ddm_w_3d_sweep_%a_%j.out
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-8

CHECKPOINTS=(
    results/ddm_w_3d/ddm_w_3d_v29_best.pth
    results/ddm_w_3d/ddm_w_3d_v29_wp01_best.pth
    results/ddm_w_3d/ddm_w_3d_v29_wp10_best.pth
    results/ddm_w_3d/ddm_w_3d_v29_wp20_best.pth
    results/ddm_w_3d/ddm_w_3d_v29_wp50_best.pth
    results/ddm_w_3d/ddm_w_3d_v29_waoa5_best.pth
    results/ddm_w_3d/ddm_w_3d_v29_waoa9_best.pth
    results/ddm_w_3d/ddm_w_3d_v29_waoa20_best.pth
    results/ddm_w_3d/ddm_w_3d_v29_waoa30_best.pth
)

ckpt=${CHECKPOINTS[$SLURM_ARRAY_TASK_ID]}

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

echo "=== Evaluating $ckpt ==="
python -u -m engiopt.ddm.ddm_w.evaluate_ddm_w_3d \
    --checkpoint      "$ckpt" \
    --bae_checkpoint  results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --lvae_checkpoint results/lvae_3d/lvae_3d_v29_best.pth \
    --n_passes 10 \
    --out_dir results/ddm_w_3d_evaluation
