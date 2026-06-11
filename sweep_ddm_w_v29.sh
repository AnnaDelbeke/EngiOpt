#!/bin/bash
# Submit 10 DDM-W v29 sweep runs in parallel

RUNS=(
  "wp05_waoa1   0.5  1"
  "wp05_waoa3   0.5  3"
  "wp05_waoa7   0.5  7"
  "wp05_waoa12  0.5  12"
  "wp05_waoa25  0.5  25"
  "wp10_waoa9   1.0  9"
  "wp10_waoa20  1.0  20"
  "wp02_waoa15  0.2  15"
  "wp03_waoa15  0.3  15"
  "wp100_waoa15 10.0 15"
)

for RUN in "${RUNS[@]}"; do
  read NAME WP WAOA <<< "$RUN"
  JOBSCRIPT=$(cat <<SLURM
#!/bin/bash
#SBATCH --job-name=ddm_w_${NAME}
#SBATCH --output=logs/ddm_w_v29_${NAME}_%j.out
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

python -u -m engiopt.ddm.ddm_w.train_ddm_w_3d \
    --bae_checkpoint  results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --lvae_checkpoint results/lvae_3d/lvae_3d_v29_best.pth \
    --model_name      ddm_w_3d_v29_${NAME} \
    --w_pressure      ${WP} \
    --w_aoa           ${WAOA}
SLURM
)
  echo "$JOBSCRIPT" | sbatch
  echo "Submitted: ddm_w_3d_v29_${NAME}  (w_p=${WP}, w_α=${WAOA})"
done
