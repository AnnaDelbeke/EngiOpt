#!/bin/bash
#SBATCH --job-name=pca_ablation_3d
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.4h
#SBATCH --output=logs/pca_ablation_3d_%A_%a.out
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

python -u -m engiopt.ddm.ddm_pca.evaluate_pca_ablation_3d \
    --bae_checkpoint results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --n_samples      $N \
    --n_components   64 \
    --seed           $S \
    --out_dir        results/pca_ablation/n${N}_s${S}
