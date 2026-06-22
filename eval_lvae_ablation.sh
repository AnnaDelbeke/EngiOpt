#!/bin/bash
#SBATCH --job-name=eval_lvae_ablation
#SBATCH --output=logs/eval_lvae_ablation_%j.out
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

BAE_CKPT="results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt"

for run in results/lvae_3d_ablation/n*_s*; do
    name=$(basename $run)
    ckpt="$run/lvae_3d_ablation_${name}_best.pth"
    if [ -f "$ckpt" ]; then
        echo "=== Joint: $name ==="
        python -u -m engiopt.lvae.evaluate_lvae_3d_ablation \
            --checkpoint "$ckpt" \
            --bae_checkpoint "$BAE_CKPT" \
            --out_dir "$run"
    else
        echo "MISSING: $ckpt"
    fi
done

for run in results/lvae_3d_ablation_geom_only/n*_s*; do
    name=$(basename $run)
    ckpt="$run/lvae_3d_ablation_geom_only_${name}_best.pth"
    if [ ! -f "$ckpt" ]; then
        ckpt="$run/lvae_3d_ablation_${name}_best.pth"
    fi
    if [ -f "$ckpt" ]; then
        echo "=== Geom-only: $name ==="
        python -u -m engiopt.lvae.evaluate_lvae_3d_ablation \
            --checkpoint "$ckpt" \
            --bae_checkpoint "$BAE_CKPT" \
            --out_dir "$run"
    else
        echo "MISSING: $ckpt"
    fi
done

echo "All done. Re-plotting..."
python -u plot_ablation_lvae.py
