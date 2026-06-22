#!/bin/bash
#SBATCH --job-name=eval_ddm_pca_ablation
#SBATCH --output=logs/eval_ddm_pca_ablation_%A_%a.out
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-17

N_SAMPLES=(100 100 150 150 200 200 300 300 400 400 500 500 600 600 700 700 767 767)
SEEDS=(      0   1   0   1   0   1   0   1   0   1   0   1   0   1   0   1   0   1)

N=${N_SAMPLES[$SLURM_ARRAY_TASK_ID]}
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
echo "Array task ${SLURM_ARRAY_TASK_ID} -> n_samples=${N}, seed=${S}"

CKPT="results/ddm_pca_3d_ablation/n${N}_s${S}/ddm_pca_3d_ablation_n${N}_s${S}_best.pth"
OUT_DIR="results/ddm_pca_3d_ablation/n${N}_s${S}"

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

python -u -m engiopt.ddm.ddm_pca.evaluate_ddm_pca_3d \
    --checkpoint     ${CKPT} \
    --bae_checkpoint results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --n_passes       10 \
    --out_dir        ${OUT_DIR}
