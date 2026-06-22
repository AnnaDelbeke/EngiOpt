#!/bin/bash
#SBATCH --job-name=eval_bae3d_ablation
#SBATCH --output=logs/eval_bae3d_ablation_%A_%a.out
#SBATCH --gpus=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=8G
#SBATCH --array=0-17

N_SAMPLES=( 50  50 100 100 150 150 200 200 300 300 400 400 500 500 600 600 700 700)
SEEDS=(      0   1   0   1   0   1   0   1   0   1   0   1   0   1   0   1   0   1)

N=${N_SAMPLES[$SLURM_ARRAY_TASK_ID]}
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "Array task ${SLURM_ARRAY_TASK_ID} -> n_samples=${N}, seed=${S}"

CKPT="results/bezier_ae_3d_ablation/n${N}_s${S}/models/bae_3d_ablation_n${N}_s${S}_best.pt"
OUT_DIR="results/bezier_ae_3d_ablation/n${N}_s${S}"

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

python -m engiopt.bezier_ae.evaluate_bezier_ae_3d_ablation \
    --checkpoint ${CKPT} \
    --n_samples ${N} \
    --seed ${S} \
    --out_dir ${OUT_DIR}
