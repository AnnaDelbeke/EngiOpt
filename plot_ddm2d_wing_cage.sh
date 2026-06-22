#!/bin/bash
#SBATCH --job-name=plot_ddm2d_cage
#SBATCH --output=logs/plot_ddm2d_cage_%j.out
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=8G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt
python -u -m engiopt.ddm.plot_thesis_2d_ddm_wing 0
python -u -m engiopt.ddm.plot_thesis_2d_ddm_wing 1
