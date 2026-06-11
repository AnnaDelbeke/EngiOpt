#!/bin/bash
#SBATCH --job-name=sweep_summary
#SBATCH --output=logs/sweep_summary_%j.out
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --mem-per-cpu=1G

cd /cluster/home/adelbeke/EngiOpt
source .venv/bin/activate

python3 -c "
import json, glob, os

# current best for reference
REFERENCE = {
    'ddm_w_3d_v29_best (w_p=0.5 w_a=15)': 'results/ddm_w_3d_evaluation/eval_ddm_w_3d_v29_best_20260610_110904.json',
}
SWEEP_NAMES = [
    'wp05_waoa1','wp05_waoa3','wp05_waoa7','wp05_waoa12','wp05_waoa25',
    'wp10_waoa9','wp10_waoa20','wp02_waoa15','wp03_waoa15','wp100_waoa15',
]

rows = []

# reference
for label, path in REFERENCE.items():
    if os.path.exists(path):
        d = json.load(open(path))
        rows.append((label, d))

# sweep results - pick most recent eval file per name
for name in SWEEP_NAMES:
    pattern = f'results/ddm_w_3d_evaluation/eval_ddm_w_3d_v29_{name}_best_*.json'
    files = sorted(glob.glob(pattern))
    if not files:
        rows.append((f'v29_{name}', None))
        continue
    d = json.load(open(files[-1]))
    rows.append((f'v29_{name}', d))

# print table
cols = ['shape_mse','aoa_mse','mmd','vendi','pressure_mse']
header = f\"{'Model':<28} {'ShapeMSE(e-5)':>13} {'AoAMSE':>8} {'MMD':>8} {'Vendi':>7} {'PresMSE':>9}\"
print(header)
print('-' * len(header))
for label, d in rows:
    if d is None:
        print(f'{label:<28}  (no results yet)')
        continue
    s  = d.get('shape_mse', float('nan'))
    a  = d.get('aoa_mse',   float('nan'))
    m  = d.get('mmd',       float('nan'))
    v  = d.get('vendi',     float('nan'))
    p  = d.get('pressure_mse', float('nan'))
    print(f'{label:<28} {s*1e5:>13.3f} {a:>8.4f} {m:>8.5f} {v:>7.4f} {p:>9.5f}')
"
