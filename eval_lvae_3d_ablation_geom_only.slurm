#!/bin/bash
#SBATCH --job-name=lvae_3d_abl_geom_eval
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.4h
#SBATCH --output=logs/lvae_3d_ablation_geom_only_eval_%A_%a.out
#SBATCH --array=0-19

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

# Must match train_lvae_3d_ablation_geom_only.slurm exactly
N_SAMPLES=(50 100 150 200 300 400 500 600 700 767)
SEEDS=(0 1)

N_IDX=$(( SLURM_ARRAY_TASK_ID / 2 ))
S_IDX=$(( SLURM_ARRAY_TASK_ID % 2 ))

N=${N_SAMPLES[$N_IDX]}
S=${SEEDS[$S_IDX]}

CHECKPOINT="results/lvae_3d_ablation_geom_only/n${N}_s${S}/lvae_3d_ablation_geom_only_n${N}_s${S}_best.pth"

echo "Array task $SLURM_ARRAY_TASK_ID -> checkpoint=$CHECKPOINT"

python -u -m engiopt.lvae.evaluate_lvae_3d_ablation \
    --checkpoint     $CHECKPOINT \
    --bae_checkpoint results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --out_dir        results/lvae_3d_ablation_geom_only/n${N}_s${S}
