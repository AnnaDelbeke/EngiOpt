#!/bin/bash
#SBATCH --job-name=spanwise_metrics_ddmw
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.4h
#SBATCH --output=logs/spanwise_metrics_ddmw_%j.out

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

echo "=== Spanwise metrics DDM_W ==="
python3 plot_spanwise_metrics_ddmw.py \
    --checkpoint      results/ddm_w_3d/ddm_w_3d_v29_best.pth \
    --bae_checkpoint  results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --lvae_checkpoint results/lvae_3d/lvae_3d_v29_best.pth \
    --n_passes 10 \
    --out results/plots/spanwise_metrics_ddmw.pdf

echo "=== Done ==="
