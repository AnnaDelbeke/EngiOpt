#!/bin/bash
#SBATCH --job-name=ddm_w_3d_ablation
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --partition=gpu.24h
#SBATCH --output=logs/ddm_w_3d_ablation_%A_%a.out
#SBATCH --array=0-19

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

# 10 sizes x 2 seeds = 20 jobs
N_SAMPLES=(50 100 150 200 300 400 500 600 700 767)
SEEDS=(0 1)

N_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
S_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))

N=${N_SAMPLES[$N_IDX]}
S=${SEEDS[$S_IDX]}

echo "Array task $SLURM_ARRAY_TASK_ID -> n_samples=$N, seed=$S"

python -u -m engiopt.ddm.ddm_w.train_ddm_w_3d \
    --bae_checkpoint  results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --lvae_checkpoint results/lvae_3d/lvae_3d_v29_best.pth \
    --w_pressure      0.5 \
    --w_aoa           15.0 \
    --lr              1e-4 \
    --n_samples       $N \
    --seed            $S \
    --save_dir        results/ddm_w_3d_ablation_v29/n${N}_s${S} \
    --wandb \
    --wandb_project   engiopt-ddm-w-3d-ablation-v29
