#!/bin/bash
#SBATCH --job-name=regen_midspan
#SBATCH --output=logs/regen_midspan_%A_%a.out
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-9

N_SAMPLES=(50 100 150 200 300 400 500 600 700 767)

N=${N_SAMPLES[$SLURM_ARRAY_TASK_ID]}

if [ "$N" = "767" ]; then
    CKPT="results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt"
else
    CKPT="results/bezier_ae_3d_ablation/n${N}_s0/models/bae_3d_ablation_n${N}_s0_best.pt"
fi
OUT_DIR="results/bezier_ae_3d_ablation/n${N}_s0"

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

python -m engiopt.bezier_ae.evaluate_bezier_ae_3d_ablation \
    --checkpoint ${CKPT} \
    --n_samples ${N} \
    --seed 0 \
    --out_dir ${OUT_DIR}
