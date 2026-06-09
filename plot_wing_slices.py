import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

processed_pkl_path = "/cluster/home/adelbeke/EngiOpt/Wing_TL/data/processed/new_dataset_slices.pkl"
case_nums = [14, 40, 515, 898, 1190]
save_dir = "/cluster/home/adelbeke/EngiOpt/results/plots"
os.makedirs(save_dir, exist_ok=True)

print("Loading dataset...")
with open(processed_pkl_path, 'rb') as f:
    df = pickle.load(f)

for case_num in case_nums:
    all_slice_nums = df[df['case_num'] == float(case_num)]['slice_num'].unique()
    final_slice_num = max(all_slice_nums) if len(all_slice_nums) > 0 else 0
    df_case = df[(df['case_num'] == float(case_num)) & (df['slice_num'] == final_slice_num)]

    if df_case.empty:
        print(f"Case {case_num}: not found in dataset, skipping.")
        continue

    unique_slices = sorted(df_case['sub_slice_num'].unique())
    print(f"Case {case_num}: {len(unique_slices)} spanwise slices found.")

    fig, axes = plt.subplots(5, 3, figsize=(15, 18), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, slice_val in enumerate(unique_slices[:15]):
        ax = axes[i]
        df_slice = df_case[df_case['sub_slice_num'] == slice_val]
        ax.plot(df_slice['CoordinateX'], df_slice['CoordinateY'], 'b-o', markersize=2, linewidth=1)
        ax.set_title(f"Slice #{i+1} (Pos: {slice_val})", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal', adjustable='box')
        if i == 0:
            ax.text(0.05, 0.85, "WING ROOT", transform=ax.transAxes, color='darkgreen', fontweight='bold', fontsize=9)
        if i == 14 or slice_val == max(unique_slices):
            ax.text(0.05, 0.85, "WING TIP", transform=ax.transAxes, color='darkred', fontweight='bold', fontsize=9)

    fig.suptitle(f"3D Wing Spanwise Decomposition — Case {case_num}", fontsize=16, fontweight='bold', y=0.95)
    fig.text(0.5, 0.08, 'CoordinateX (Chord Location)', ha='center', fontsize=12)
    fig.text(0.08, 0.5, 'CoordinateY (Profile Thickness)', va='center', rotation='vertical', fontsize=12)

    save_path = os.path.join(save_dir, f"case_{case_num}_spanwise_slices_final.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
