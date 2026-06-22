"""Find the validation wing with the highest tip-slice reconstruction MSE."""
import torch
from torch.utils.data import random_split

from engiopt.bezier_ae.bezier_ae_3d import BezierAutoencoder3D
from engiopt.bezier_ae.train_bezier_ae_3d import WingsBezierDataset3D
from engiopt.data_processing.new_dataset_adapter import NewWingsDataset

CHECKPOINT = "results/bezier_ae_3d/run_039/models/bezier_ae_3d_best.pt"
SLICES_PKL  = "Wing_TL/data/processed/new_dataset_slices.pkl"
SCALARS_PKL = "Wing_TL/data/processed/new_dataset_scalars.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

new_dataset = NewWingsDataset(SLICES_PKL, SCALARS_PKL, seed=0)
all_items   = [item for item in list(new_dataset["train"]) + list(new_dataset["val"])
               if item["final"] == 1]
full_dataset = WingsBezierDataset3D(all_items, num_extra_tip_slices=0)
train_size   = int(0.9 * len(full_dataset))
val_size     = len(full_dataset) - train_size
_, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(0),
)
print(f"Validation wings: {len(val_dataset)}")

ckpt  = torch.load(CHECKPOINT, map_location=device, weights_only=False)
model = BezierAutoencoder3D(
    n_spans=ckpt["n_spans"],
    n_control_points=ckpt["n_control_points"],
    n_data_points=192,
    slice_hidden_dims=ckpt["slice_hidden_dims"],
    span_hidden_dims=ckpt["span_hidden_dims"],
    latent_dim=ckpt["latent_dim"],
    cpx_bound=ckpt["cpx_bound"],
    cpy_bound=ckpt["cpy_bound"],
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

tip_mses = []
with torch.no_grad():
    for i in range(len(val_dataset)):
        x = val_dataset[i].unsqueeze(0).to(device)
        z = model.encode(x)
        y, _ = model.decode(z, return_cp=True)
        tip_mse = ((x[0, -1] - y[0, -1]) ** 2).mean().item()
        tip_mses.append((tip_mse, i))

tip_mses.sort(reverse=True)
print("\nTop 10 wings by tip MSE:")
for mse, idx in tip_mses[:10]:
    print(f"  val index {idx:3d}  tip MSE = {mse:.2e}")
