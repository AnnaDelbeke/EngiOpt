#!/bin/bash
#SBATCH --job-name=train_ddm3d_ablation
#SBATCH --output=logs/train_ddm3d_ablation_%A_%a.out
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-19

N_SAMPLES=(50 50 100 100 150 150 200 200 300 300 400 400 500 500 600 600 700 700 767 767)
SEEDS=(      0  1   0   1   0   1   0   1   0   1   0   1   0   1   0   1   0   1   0   1)

N=${N_SAMPLES[$SLURM_ARRAY_TASK_ID]}
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "Array task ${SLURM_ARRAY_TASK_ID} -> n_samples=${N}, seed=${S}"

OUT_DIR="results/ddm_3d_ablation/n${N}_s${S}"

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

python -u -m engiopt.ddm.train_ddm_3d \
    --bae_checkpoint results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --bae_latent_dim 128 \
    --model_name     ddm_3d_ablation_n${N}_s${S} \
    --n_samples      ${N} \
    --seed           ${S} \
    --save_dir       ${OUT_DIR} \
    --w_aoa          9.0
