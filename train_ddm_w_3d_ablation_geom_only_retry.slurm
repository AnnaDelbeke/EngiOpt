#!/bin/bash
#SBATCH --job-name=ddm_w_3d_abl_geom_retry
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=1
#SBATCH --partition=gpu.24h
#SBATCH --output=logs/ddm_w_3d_ablation_geom_only_retry_%A_%a.out
#SBATCH --array=0-4

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

# Retry of the 5 tasks that hung on torch.cuda.is_available() during the
# first attempt (array idx 5, 11, 13, 14, 16 of train_ddm_w_3d_ablation_geom_only.slurm)
N_SAMPLES=(150 400 500 600 700)
SEEDS=(1    1   1   0   0)

N=${N_SAMPLES[$SLURM_ARRAY_TASK_ID]}
S=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "Retry array task $SLURM_ARRAY_TASK_ID -> n_samples=$N, seed=$S"

python -u -m engiopt.ddm.ddm_w.train_ddm_w_3d \
    --bae_checkpoint  results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --lvae_checkpoint results/lvae_3d_geom_only/lvae_3d_v29_geom_only_best.pth \
    --w_pressure      0.0 \
    --w_aoa           15.0 \
    --lr              1e-4 \
    --n_samples       $N \
    --seed            $S \
    --save_dir        results/ddm_w_3d_ablation_geom_only/n${N}_s${S} \
    --wandb \
    --wandb_project   engiopt-ddm-w-3d-ablation-geom-only
