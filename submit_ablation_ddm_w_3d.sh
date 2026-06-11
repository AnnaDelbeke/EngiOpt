#!/bin/bash
#SBATCH --job-name=ddm_w_3d_ablation
#SBATCH --output=logs/ddm_w_3d_ablation_%A_%a.out
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-19

N_SAMPLES=(50 50 100 100 150 150 200 200 300 300 400 400 500 500 600 600 700 700 767 767)
SEEDS=(     0  1   0   1   0   1   0   1   0   1   0   1   0   1   0   1   0   1   0   1)

N=${N_SAMPLES[$SLURM_ARRAY_TASK_ID]}
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "Array task ${SLURM_ARRAY_TASK_ID} -> n_samples=${N}, seed=${S}"

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

python -u -m engiopt.ddm.ddm_w.train_ddm_w_3d \
    --bae_checkpoint  results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --lvae_checkpoint results/lvae_3d/lvae_3d_v29_best.pth \
    --n_samples       ${N} \
    --seed            ${S}
