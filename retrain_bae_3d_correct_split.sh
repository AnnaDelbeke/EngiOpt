#!/bin/bash
#SBATCH --job-name=retrain_bae_3d
#SBATCH --output=logs/retrain_bae_3d_%j.out
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G

# Retrain 3D-BAE with corrected train/val split.
# Previously the BAE pooled new_dataset["train"] + new_dataset["val"] and
# did its own 90/10 random split, meaning it trained on official val wings.
# This run uses only new_dataset["train"] for training and new_dataset["val"]
# as the proper held-out validation set, consistent with the LVAE and DDM-W.

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.bezier_ae.train_bezier_ae_3d
