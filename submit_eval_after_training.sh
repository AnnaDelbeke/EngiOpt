#!/bin/bash
# Submit eval jobs with dependency on each training job

declare -A JOB_IDS=(
  ["wp05_waoa1"]=2870269
  ["wp05_waoa3"]=2870271
  ["wp05_waoa7"]=2870273
  ["wp05_waoa12"]=2870276
  ["wp05_waoa25"]=2870278
  ["wp10_waoa9"]=2870282
  ["wp10_waoa20"]=2870285
  ["wp02_waoa15"]=2870286
  ["wp03_waoa15"]=2870287
  ["wp100_waoa15"]=2870288
)

for NAME in "${!JOB_IDS[@]}"; do
  TRAIN_JOB=${JOB_IDS[$NAME]}
  CKPT="results/ddm_w_3d/ddm_w_3d_v29_${NAME}_best.pth"

  EVAL_SCRIPT=$(cat <<SLURM
#!/bin/bash
#SBATCH --job-name=eval_ddm_w_${NAME}
#SBATCH --output=logs/eval_ddm_w_v29_${NAME}_%j.out
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=16G

source /cluster/home/adelbeke/EngiOpt/.venv/bin/activate
cd /cluster/home/adelbeke/EngiOpt

python -u -m engiopt.ddm.ddm_w.evaluate_ddm_w_3d \
    --checkpoint      ${CKPT} \
    --bae_checkpoint  results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt \
    --lvae_checkpoint results/lvae_3d/lvae_3d_v29_best.pth \
    --n_passes        10 \
    --out_dir         results/ddm_w_3d_evaluation
SLURM
)

  EVAL_JOB=$(echo "$EVAL_SCRIPT" | sbatch --dependency=afterok:${TRAIN_JOB} --parsable)
  echo "Eval job ${EVAL_JOB} → will run after training job ${TRAIN_JOB} (${NAME})"
done
