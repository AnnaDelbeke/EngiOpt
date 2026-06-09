import os
import pickle
import sys
import matplotlib.pyplot as plt

# Always target the true, full production dataset file
processed_pkl_path = "/cluster/home/adelbeke/EngiOpt/Wing_TL/data/processed/new_dataset_slices.pkl"

# Read case number from terminal argument, default to 40.0 if left blank
if len(sys.argv) > 1:
    sample_case = float(sys.argv[1])
else:
    sample_case = 40.0

if not os.path.exists(processed_pkl_path):
    print(f"❌ File not found at: {processed_pkl_path}")
    sys.exit(1)

print(f"✅ Loading full dataset: {processed_pkl_path}")
with open(processed_pkl_path, 'rb') as f:
    df_processed = pickle.load(f)

# Find max slice
max_slice = df_processed['sub_slice_num'].max()
print(f"✈️ Extracting outer wingtip slice ({max_slice}) for Case {sample_case}...")

# Filter data
df_sample = df_processed[(df_processed['case_num'] == sample_case) & (df_processed['sub_slice_num'] == max_slice)]
df_sample_init = df_sample[df_sample['slice_num'] == 0]

if len(df_sample_init) == 0:
    print(f"❌ Wow, no data rows found for Case {sample_case} on Slice {max_slice}!")
    sys.exit(1)

print(f"📊 Extracted {len(df_sample_init)} coordinate rows. Rendering plot...")

# Plotting Configuration
plt.figure(figsize=(10, 4))
plt.plot(df_sample_init['CoordinateX'], df_sample_init['CoordinateY'], 'b-o', markersize=3, label='Preprocessed Input Path')

plt.title(f"Checking Preprocessing Output (Case {sample_case}, Slice {max_slice})")
plt.xlabel("CoordinateX (Chord)")
plt.ylabel("CoordinateY (Thickness)")
plt.axis('equal')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Save image
save_dir = "/cluster/home/adelbeke/EngiOpt/results/plots"
os.makedirs(save_dir, exist_ok=True)
save_img_path = os.path.join(save_dir, f"wingtip_case_{int(sample_case)}.png")

plt.savefig(save_img_path, dpi=200, bbox_inches='tight')
plt.close()

print(f"📸 Success! Visual plot saved as an image to: {save_img_path}")
