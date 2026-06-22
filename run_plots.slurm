#!/bin/bash
#SBATCH --job-name=thesis_plots
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --partition=gpu.4h
#SBATCH --output=logs/thesis_plots_%j.out

module purge
module load stack/2024-06 python/3.10.13

cd ~/EngiOpt
mkdir -p logs
source .venv/bin/activate

export HF_DATASETS_OFFLINE=1

echo "=== Manifold distance scatter ==="
python3 plot_manifold_distance.py

echo "=== Nearest-neighbour novelty grid (skip — already done) ==="
# python3 plot_nn_novelty.py

echo "=== Done ==="
