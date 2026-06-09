import os
import pickle
import sys

# 🛠️ Fix: Properly handle the path passed via terminal
if len(sys.argv) > 1:
    processed_pkl_path = sys.argv[1]
else:
    processed_pkl_path = "/cluster/home/adelbeke/EngiOpt/data/processed/new_dataset_slices.pkl"

print(f"🔄 Opening dataset: {processed_pkl_path}...")

if not os.path.exists(processed_pkl_path):
    print(f"❌ File not found at: {processed_pkl_path}")
    sys.exit(1)

with open(processed_pkl_path, 'rb') as f:
    df = pickle.load(f)

all_cases = sorted(df['case_num'].unique())
print(f"📊 Total unique cases found in file: {len(all_cases)}")

healthy_row_count = len(df[df['case_num'] == all_cases[0]])
print(f"✅ Baseline Case {all_cases[0]} has {healthy_row_count} coordinate rows.")

blank_or_incomplete = []

print("\n🔍 Hunting for blank, partial, or missing wingtip slices...")
print("-" * 60)

for case in all_cases:
    df_case = df[df['case_num'] == case]
    row_count = len(df_case)
    has_wingtip = 349.0 in df_case['sub_slice_num'].values
    
    if row_count == 0 or row_count < healthy_row_count or not has_wingtip:
        print(f"❌ Case {case:5.1f} is CORRUPTED/BLANK! | Rows: {row_count} | Has Wingtip 349.0: {has_wingtip}")
        blank_or_incomplete.append(int(case))

print("-" * 60)
print("\n📊 HUNT SUMMARY:")
print(f"Total Cases Scanned: {len(all_cases)}")
print(f"Total Blank/Corrupted Cases Found: {len(blank_or_incomplete)}")
if blank_or_incomplete:
    print(f"🚨 Python Blacklist Array: {blank_or_incomplete}")
else:
    print("🎉 All cases in this specific file are clean!")
