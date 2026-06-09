import os
import pickle
import numpy as np

base_dir = "/cluster/home/adelbeke/EngiOpt"
processed_pkl_path = os.path.join(base_dir, "data", "processed", "new_dataset_slices.pkl")

print(f"🔄 Scanning entire dataset at: {processed_pkl_path}...")

with open(processed_pkl_path, 'rb') as f:
    df = pickle.load(f)

# Find all unique case numbers
all_cases = df['case_num'].unique()
bad_cases = []

for case in all_cases:
    df_case = df[df['case_num'] == case]
    
    # 1. Check if the case is completely empty or missing rows
    if len(df_case) == 0:
        print(f"❌ Case {case}: Completely empty (0 rows)!")
        bad_cases.append(case)
        continue
        
    # 2. Check for NaN values anywhere in coordinates
    if df_case[['CoordinateX', 'CoordinateY']].isna().any().any():
        print(f"❌ Case {case}: Contains NaN values!")
        bad_cases.append(case)
        continue

    # 3. Check for collapsed or zero-variance geometry (like Case 28)
    x_range = df_case['CoordinateX'].max() - df_case['CoordinateX'].min()
    y_range = df_case['CoordinateY'].max() - df_case['CoordinateY'].min()
    
    if x_range < 1e-3 or y_range < 1e-3:
        print(f"❌ Case {case}: Collapsed geometry! (X range: {x_range:.5f}, Y range: {y_range:.5f})")
        bad_cases.append(case)

print("\n--- 📊 SCAN SUMMARY ---")
print(f"Total cases checked: {len(all_cases)}")
print(f"Total bad/corrupted cases found: {len(bad_cases)}")
if bad_cases:
    print(f"🚨 List of bad cases to remove: {bad_cases}")
else:
    print("🎉 All clear! Every single case contains valid geometric data.")
