#!/bin/bash
#SBATCH --job-name=lvae_3d_ablation
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.24h
#SBATCH --output=logs/lvae_3d_ablation_%A_%a.out
#SBATCH --array=0-19

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

# 10 sizes x 2 seeds = 20 jobs  (mirrors DDM_W ablation sizing)
N_SAMPLES=(50 100 150 200 300 400 500 600 700 767)
SEEDS=(0 1)

N_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
S_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))

N=${N_SAMPLES[$N_IDX]}
S=${SEEDS[$S_IDX]}

echo "Array task $SLURM_ARRAY_TASK_ID -> n_samples=$N, seed=$S"

python -u -m engiopt.lvae.train_lvae_3d \
    --bae_checkpoint    results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --model_name        lvae_3d_ablation_n${N}_s${S} \
    --joint_encoder \
    --pressure_embed_dim 16 \
    --lae_latent_dim    64 \
    --dropout           0.25 \
    --lambda_lv         5e-1 \
    --w_bae             1000.0 \
    --w_eta             1000.0 \
    --w_aoa             9.0 \
    --w_pressure        1.0 \
    --w_perf            1.0 \
    --prune_every       500 \
    --prune_threshold   0.02 \
    --n_epochs          10000 \
    --lr                1e-3 \
    --n_samples         $N \
    --seed              $S \
    --save_dir          results/lvae_3d_ablation/n${N}_s${S} \
    --wandb \
    --wandb_project     engiopt-lvae-3d-ablation
