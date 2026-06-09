#!/bin/bash
#SBATCH --job-name=eval_lvae_joint
#SBATCH --output=logs/eval_lvae_joint_%j.out
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.lvae.evaluate_lvae \
    --checkpoint results/lvae/joint/lae_joint_v1_best.pth
