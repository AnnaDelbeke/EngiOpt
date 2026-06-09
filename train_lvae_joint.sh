#!/bin/bash
#SBATCH --job-name=train_lvae_joint
#SBATCH --output=logs/train_lvae_joint_%j.out
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.lvae.train_lvae \
    --joint_encoder \
    --model_name lae_joint_v1 \
    --save_dir results/lvae/joint \
    --lambda_lv 0.0 \
    --dropout 0.25 \
    --wandb \
    --wandb_project engiopt-lvae-sweep
