import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

def find_and_plot_gt():
    # 1. Search for any saved tensor data in your run directories
    search_path = "/cluster/home/adelbeke/EngiOpt/results/bezier_ae_3d/**/*.pt"
    files = glob.glob(search_path, recursive=True)
    
    if not files:
        print("❌ Could not find any saved .pt files automatically.")
        print("Please modify the 'file_path' variable below to point directly to a data file.")
        return

    # Let's grab the first file found for a quick check
    file_path = files[0]
    print(f"📦 Loading data file from: {file_path}")
    
    try:
        data = torch.load(file_path, map_location='cpu')
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    # 2. Extract coordinates depending on how your state dict / data is structured
    print(f"Data type: {type(data)}")
    if isinstance(data, dict):
        print(f"Keys found in dict: {list(data.keys())}")
        # Try common validation/test save keys
        coords = data.get('coords_gt', data.get('gt', data.get('inputs', None)))
    else:
        coords = data

    if coords is None:
        print("❌ Could not extract coordinate tensors automatically from the saved file keys.")
        return

    # Ensure numpy array format
    if isinstance(coords, torch.Tensor):
        coords = coords.detach().numpy()

    print(f"📐 Extracted shape: {coords.shape}")
    
    # Target shape adjustment to find a 2D slice [2, 192]
    # If it's a batch/span combo like [B, S, 2, N], slice the first batch and span
    if len(coords.shape) == 4:   # [B, S, C, N]
        slice_data = coords[0, 0]
    elif len(coords.shape) == 3: # [S, C, N] or [B, C, N]
        slice_data = coords[0]
    elif len(coords.shape) == 2 and coords.shape[0] == 2:
        slice_data = coords
    else:
        print(f"❌ Array structure shape {coords.shape} is unexpected. Trying to force-slice.")
        slice_data = coords.reshape(2, -1)

    # 3. Render the diagnostic visualizer
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Connected path tracking
    ax1.plot(slice_data[0], slice_data[1], 'b.-', alpha=0.6)
    ax1.scatter(slice_data[0, :1], slice_data[1, :1], color='red', s=50, zorder=5, label='Index 0 Start')
    ax1.set_title("Ground Truth: Connected Lines")
    ax1.grid(True)
    ax1.axis('equal')
    ax1.legend()

    # Panel 2: Continuous point index flow mapping
    indices = np.arange(slice_data.shape[1])
    sc = ax2.scatter(slice_data[0], slice_data[1], c=indices, cmap='viridis', s=20)
    fig.colorbar(sc, ax2, label='Coordinate Point Index')
    ax2.set_title("Ground Truth: Point Indexing Flow")
    ax2.grid(True)
    ax2.axis('equal')

    plt.suptitle(f"DATA DIAGNOSTIC: {os.path.basename(file_path)}")
    
    # Save image out to disk so you can open it easily
    out_img = "/cluster/home/adelbeke/EngiOpt/gt_diagnostic_check.png"
    plt.savefig(out_img, bbox_inches='tight')
    print(f"🎨 Success! Diagnostic plot saved to: {out_img}")

if __name__ == "__main__":
    find_and_plot_gt()