"""Constrained Performance-LVAE for 2D designs with one-sided constraint handling.

Extends the one-sided constraint method to handle two constraints:
1. Reconstruction constraint: NMSE_rec <= threshold_rec
2. Performance constraint: NMSE_perf <= threshold_perf

Uses reconstruction-first priority: reconstruction must be satisfied before
performance, and both must be satisfied before volume optimization begins.

For more information on LVAE, see: https://arxiv.org/abs/2404.17773
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import os
import random
import time

from engibench.utils.all_problems import BUILTIN_PROBLEMS
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import RobustScaler
import torch as th
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import tqdm
import tyro

from engiopt.transforms import get_image_condition_keys
from engiopt.transforms import get_image_condition_shape
from engiopt.transforms import get_performance_target
from engiopt.transforms import get_scalar_condition_keys
from engiopt.transforms import rasterize_index_conditions
from engiopt.vanilla_lvae.aes import ConstrainedPerfLeastVolumeAE_DP
from engiopt.vanilla_lvae.components import ConditionEncoder2D
from engiopt.vanilla_lvae.components import Encoder2D
from engiopt.vanilla_lvae.components import SNMLPPredictor
from engiopt.vanilla_lvae.components import TrueSNDecoder2D
from engiopt.vanilla_lvae.utils import filter_dataset_by_condition
import wandb


@dataclass
class Args:
    """Command-line arguments for constrained Performance-LVAE training."""

    # Problem and tracking
    problem_id: str = "beams2d"
    """Problem ID to run. Must be one of the built-in problems in engibench."""
    algo: str = os.path.basename(__file__)[: -len(".py")]
    """Algorithm name for tracking purposes."""
    track: bool = True
    """Whether to track with Weights & Biases."""
    wandb_project: str = "engiopt"
    """WandB project name."""
    wandb_entity: str | None = None
    """WandB entity name. If None, uses the default entity."""
    seed: int = 1
    """Random seed for reproducibility."""
    save_model: bool = False
    """Whether to save the model after training."""
    sample_interval: int = 500
    """Interval for sampling designs during training."""

    # Training parameters
    n_epochs: int = 10000
    """Number of training epochs."""
    batch_size: int = 128
    """Batch size for training."""
    lr: float = 1e-4
    """Learning rate for the optimizer."""

    # LVAE-specific
    latent_dim: int = 100
    """Dimensionality of the latent space (overestimate)."""
    perf_dim: int = -1
    """Number of latent dimensions dedicated to performance prediction. If -1 (default), uses all latent_dim dimensions."""

    # Constraint parameters (uses Normalized MSE = MSE / Var(data) for problem-independence)
    nmse_threshold_rec: float = 0.05
    """NMSE threshold for reconstruction. Training aims to stay at or below this. Default: 0.01 (R² = 99%)."""
    nmse_threshold_perf: float = 0.05
    """NMSE threshold for performance prediction. Default: 0.05 (R² = 95%)."""

    # Pruning parameters
    pruning_epoch: int = 500
    """Epoch to start pruning dimensions."""
    pruning_threshold: float = 0.05
    """Threshold for pruning (ratio for plummet, percentile for lognorm)."""
    pruning_strategy: str = "plummet"
    """Pruning strategy to use: 'plummet' or 'lognorm'."""
    alpha: float = 0.0
    """(lognorm only) Blending factor between reference and current distribution."""

    # Architecture
    resize_dimensions: tuple[int, int] = (100, 100)
    """Dimensions to resize input images to before encoding/decoding."""
    predictor_hidden_dims: tuple[int, ...] = (256, 128)
    """Hidden dimensions for the MLP predictor."""
    conditional_predictor: bool = False
    """Whether to include conditions in performance prediction (True) or use only latent codes (False)."""
    conditional_decoder: bool = False
    """Whether to condition the decoder on scalar + image conditions for multi-modality measurement."""
    cond_embed_dim: int = 64
    """Dimensionality of image condition embedding (only used when image conditions exist)."""
    decoder_lipschitz_scale: float = 1.0
    """Lipschitz bound for spectrally normalized decoder. Controls output scaling."""
    predictor_lipschitz_scale: float = 1.0
    """Lipschitz bound for spectrally normalized MLP predictor. Controls output scaling."""

    # Dataset filtering
    condition_filter_key: str | None = None
    """Condition key to filter dataset on (e.g., 'weight'). None = use all data."""
    condition_filter_value: float | None = None
    """Exact value to match (within tolerance). Mutually exclusive with condition_filter_range."""
    condition_filter_range: tuple[float, float] | None = None
    """Inclusive [lo, hi] range to filter on. Overrides condition_filter_value."""
    condition_filter_tolerance: float = 0.01
    """Tolerance for exact-value matching."""


if __name__ == "__main__":
    args = tyro.cli(Args)

    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=args.seed)

    design_shape = problem.design_space.shape
    scalar_cond_keys = get_scalar_condition_keys(problem, problem.dataset["train"])
    n_conds = len(scalar_cond_keys)

    # Detect image conditions (e.g., boundary matrices in thermoelastic2d)
    img_cond_keys = get_image_condition_keys(problem, problem.dataset["train"])
    n_img_conds = len(img_cond_keys)
    img_cond_shape: tuple[int, ...] | None = None
    if n_img_conds > 0:
        img_cond_shape = get_image_condition_shape(problem.dataset["train"], img_cond_keys)
        print(f"Found {n_img_conds} image condition(s): {img_cond_keys}, shape={img_cond_shape}")

    # Logging
    run_name = f"{args.problem_id}__{args.algo}__{args.seed}__{int(time.time())}"
    if args.track:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={**vars(args), "design_shape": list(design_shape)},
            save_code=True,
            name=run_name,
        )

    # Seeding for reproducibility
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    g = th.Generator().manual_seed(args.seed)  # For DataLoader shuffling

    os.makedirs("images", exist_ok=True)

    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    # Build encoder (always unconditional — sees design only)
    enc = Encoder2D(args.latent_dim, design_shape, args.resize_dimensions)

    # Build condition encoder for image conditions (if any)
    condition_encoder: ConditionEncoder2D | None = None
    img_cond_embed_dim = 0
    if n_img_conds > 0:
        condition_encoder = ConditionEncoder2D(
            n_img_conds=n_img_conds,
            cond_embed_dim=args.cond_embed_dim,
            resize_dimensions=args.resize_dimensions,
        )
        img_cond_embed_dim = args.cond_embed_dim

    # Compute condition dimensions for decoder and predictor
    # Scalar conditions are included when conditional_predictor is on (they get scaled)
    # Image condition embeddings are always included when image conditions exist
    n_scalar_for_cond = n_conds if args.conditional_predictor else 0

    cond_dim_for_decoder = 0
    if args.conditional_decoder:
        cond_dim_for_decoder = n_scalar_for_cond + img_cond_embed_dim

    cond_dim_for_predictor = n_scalar_for_cond + img_cond_embed_dim

    # Build decoder (conditional when cond_dim > 0)
    dec = TrueSNDecoder2D(
        args.latent_dim, design_shape, lipschitz_scale=args.decoder_lipschitz_scale, cond_dim=cond_dim_for_decoder
    )

    # Determine perf_dim: if -1 (default), use all latent dimensions
    perf_dim = args.latent_dim if args.perf_dim == -1 else args.perf_dim
    n_perf = 1  # Single performance objective

    # Build MLP predictor (input: perf_dim latent dims + condition embedding)
    predictor_input_dim = perf_dim + cond_dim_for_predictor
    predictor = SNMLPPredictor(
        input_dim=predictor_input_dim,
        output_dim=n_perf,
        hidden_dims=args.predictor_hidden_dims,
        lipschitz_scale=args.predictor_lipschitz_scale,
    )

    print(f"\n{'=' * 60}")
    print("Constrained Performance-LVAE Training (One-Sided)")
    print(f"Problem: {args.problem_id}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Decoder: TrueSNDecoder2D (lipschitz_scale={args.decoder_lipschitz_scale}, cond_dim={cond_dim_for_decoder})")
    print(f"Decoder mode: {'Conditional' if args.conditional_decoder else 'Unconditional'}")
    print(f"Perf dim: {perf_dim} (first {perf_dim} dims predict performance)")
    print(f"Predictor mode: {'Conditional' if args.conditional_predictor else 'Unconditional'}")
    print(f"Predictor: SNMLPPredictor (lipschitz_scale={args.predictor_lipschitz_scale})")
    print(f"Predictor input: {predictor_input_dim} (perf_dim={perf_dim}, cond_dim={cond_dim_for_predictor})")
    if n_img_conds > 0:
        print(f"Image conditions: {n_img_conds} channel(s), embed_dim={img_cond_embed_dim}")
    print(f"NMSE threshold (rec): {args.nmse_threshold_rec} (R² = {1 - args.nmse_threshold_rec:.2%})")
    print(f"NMSE threshold (perf): {args.nmse_threshold_perf} (R² = {1 - args.nmse_threshold_perf:.2%})")
    print(f"Pruning epoch: {args.pruning_epoch}")
    print(f"Pruning strategy: {args.pruning_strategy}")
    print(f"Pruning threshold: {args.pruning_threshold}")
    if args.pruning_strategy == "lognorm":
        print(f"Alpha (lognorm): {args.alpha}")
    print(f"{'=' * 60}\n")

    # Collect all parameters for optimizer (including condition encoder if present)
    all_params = list(enc.parameters()) + list(dec.parameters()) + list(predictor.parameters())
    if condition_encoder is not None:
        all_params += list(condition_encoder.parameters())

    # Initialize Constrained Performance-LVAE with dynamic pruning
    plvae = ConstrainedPerfLeastVolumeAE_DP(
        encoder=enc,
        decoder=dec,
        predictor=predictor,
        optimizer=Adam(all_params, lr=args.lr),
        latent_dim=args.latent_dim,
        perf_dim=perf_dim,
        nmse_threshold_rec=args.nmse_threshold_rec,
        nmse_threshold_perf=args.nmse_threshold_perf,
        pruning_epoch=args.pruning_epoch,
        pruning_threshold=args.pruning_threshold,
        pruning_strategy=args.pruning_strategy,
        alpha=args.alpha,
        conditional_decoder=args.conditional_decoder,
        condition_encoder=condition_encoder,
    ).to(device)

    # ---- DataLoader ----
    raw_train = problem.dataset["train"]
    raw_val = problem.dataset["val"]

    if args.condition_filter_key is not None:
        raw_train = filter_dataset_by_condition(
            raw_train,
            args.condition_filter_key,
            value=args.condition_filter_value,
            value_range=args.condition_filter_range,
            tolerance=args.condition_filter_tolerance,
        )
        raw_val = filter_dataset_by_condition(
            raw_val,
            args.condition_filter_key,
            value=args.condition_filter_value,
            value_range=args.condition_filter_range,
            tolerance=args.condition_filter_tolerance,
        )

    train_ds = raw_train.with_format("torch")
    val_ds = raw_val.with_format("torch")

    # Extract designs, conditions, and performance
    x_train = train_ds["optimal_design"][:].unsqueeze(1)
    c_train = th.stack([train_ds[key][:] for key in scalar_cond_keys], dim=-1)
    p_train = get_performance_target(problem, train_ds)

    x_val = val_ds["optimal_design"][:].unsqueeze(1)
    c_val = th.stack([val_ds[key][:] for key in scalar_cond_keys], dim=-1)
    p_val = get_performance_target(problem, val_ds)

    # Extract image conditions (if any)
    # Image conditions may be dense arrays (H, W) or sparse index arrays (variable length).
    # Sparse index arrays are rasterized into dense binary masks on design_shape.
    ic_train: th.Tensor | None = None
    ic_val: th.Tensor | None = None
    if n_img_conds > 0 and img_cond_shape is not None:
        if len(img_cond_shape) == 1:
            # Sparse node index arrays — rasterize to dense masks on node grid (H+1, W+1)
            node_grid = (design_shape[0] + 1, design_shape[1] + 1)
            _ic_tr = rasterize_index_conditions(raw_train, img_cond_keys, node_grid)
            _ic_va = rasterize_index_conditions(raw_val, img_cond_keys, node_grid)
            print(f"Sparse image conditions rasterized to {node_grid} node grid: {_ic_tr.shape}")
        else:
            # Dense image conditions — stack directly
            _ic_tr = th.stack([train_ds[k][:].float() for k in img_cond_keys], dim=1)
            _ic_va = th.stack([val_ds[k][:].float() for k in img_cond_keys], dim=1)
            print(f"Dense image conditions loaded: {_ic_tr.shape}")
        ic_train = _ic_tr
        ic_val = _ic_va

    # Scale performance values using RobustScaler
    p_scaler = RobustScaler()
    p_train_scaled = th.from_numpy(p_scaler.fit_transform(p_train.numpy())).to(p_train.dtype)
    p_val_scaled = th.from_numpy(p_scaler.transform(p_val.numpy())).to(p_val.dtype)

    # Scale conditions using RobustScaler (if using conditional predictor or conditional decoder)
    if args.conditional_predictor or args.conditional_decoder:
        c_scaler = RobustScaler()
        c_train_scaled = th.from_numpy(c_scaler.fit_transform(c_train.numpy())).to(c_train.dtype)
        c_val_scaled = th.from_numpy(c_scaler.transform(c_val.numpy())).to(c_val.dtype)
    else:
        # Dummy tensors when not using conditions (won't be used in predictor or decoder)
        c_train_scaled = th.zeros(len(x_train), 0)
        c_val_scaled = th.zeros(len(x_val), 0)

    # Set data variances for NMSE computation (problem-independent thresholds)
    plvae.set_data_variance(x_train)
    plvae.set_perf_variance(p_train_scaled)

    print(f"Data variance (designs): {plvae.data_var:.6f}")
    print(f"Perf variance (scaled): {plvae.perf_var:.6f}")

    # Build DataLoaders (include image conditions as 4th tensor when present)
    train_tensors = [x_train, c_train_scaled, p_train_scaled]
    val_tensors = [x_val, c_val_scaled, p_val_scaled]
    if ic_train is not None:
        train_tensors.append(ic_train)
        val_tensors.append(ic_val)

    loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
    )
    val_loader = DataLoader(
        TensorDataset(*val_tensors),
        batch_size=args.batch_size,
        shuffle=False,
    )

    # ---- Training loop ----
    for epoch in range(args.n_epochs):
        plvae.epoch_hook(epoch=epoch)

        bar = tqdm.tqdm(loader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(bar):
            x_batch = batch[0].to(device)
            c_batch = batch[1].to(device)
            p_batch = batch[2].to(device)
            ic_batch = batch[3].to(device) if len(batch) > 3 else None

            plvae.optim.zero_grad()

            # Compute loss (scalar, constraint-dependent)
            batch_tuple = (x_batch, c_batch, p_batch, ic_batch) if ic_batch is not None else (x_batch, c_batch, p_batch)
            loss = plvae.loss(batch_tuple)
            loss.backward()
            plvae.optim.step()

            bar.set_postfix(
                {
                    "rec": f"{plvae.rec_loss:.4f}",
                    "perf": f"{plvae.perf_loss:.4f}",
                    "vol": f"{plvae.vol_loss:.4f}",
                    "nmse_r": f"{plvae.nmse_rec:.4f}",
                    "nmse_p": f"{plvae.nmse_perf:.4f}",
                    "vol_on": int(plvae.vol_active),
                    "dim": plvae.dim,
                }
            )

            # Log to wandb
            if args.track:
                batches_done = epoch * len(bar) + i

                log_dict = {
                    "rec_loss": plvae.rec_loss,
                    "perf_loss": plvae.perf_loss,
                    "vol_loss": plvae.vol_loss,
                    "total_loss": loss.item(),
                    "nmse_rec": plvae.nmse_rec,
                    "nmse_perf": plvae.nmse_perf,
                    "nmse_threshold_rec": args.nmse_threshold_rec,
                    "nmse_threshold_perf": args.nmse_threshold_perf,
                    "vol_active": int(plvae.vol_active),
                    "rec_violated": int(plvae.nmse_rec > args.nmse_threshold_rec),
                    "perf_violated": int(plvae.nmse_perf > args.nmse_threshold_perf),
                    "active_dims": plvae.dim,
                    "epoch": epoch,
                }
                wandb.log(log_dict)

                print(
                    f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(bar)}] "
                    f"[rec: {plvae.rec_loss:.4f}] [perf: {plvae.perf_loss:.4f}] "
                    f"[vol: {plvae.vol_loss:.4f}] [nmse_rec: {plvae.nmse_rec:.4f}] "
                    f"[nmse_perf: {plvae.nmse_perf:.4f}] [vol_active: {int(plvae.vol_active)}] "
                    f"[dims: {plvae.dim}]"
                )

                # Sample and visualize at regular intervals
                if batches_done % args.sample_interval == 0:
                    with th.no_grad():
                        # Encode training designs
                        xs = x_train.to(device)
                        z = plvae.encode(xs)
                        z_std, idx = th.sort(z.std(0), descending=True)
                        z_mean = z.mean(0)
                        n_active = (z_std > 0).sum().item()

                        # Build condition embedding for visualization (if conditional decoder)
                        viz_cond_emb = None
                        if args.conditional_decoder or cond_dim_for_predictor > 0:
                            c_all = c_train_scaled.to(device)
                            ic_all = ic_train.to(device) if ic_train is not None else None
                            viz_cond_emb = plvae._build_cond_embedding(c_all, ic_all)

                        # Helper: decode with optional conditions
                        def _viz_decode(z_viz, cond=None):
                            if args.conditional_decoder and cond is not None:
                                return plvae.decoder(z_viz, cond=cond).cpu().numpy()
                            return plvae.decode(z_viz).cpu().numpy()

                        # Generate interpolated designs (use conditions from corresponding samples)
                        x_ints = []
                        for alpha in [0, 0.25, 0.5, 0.75, 1]:
                            z_ = (1 - alpha) * z[:25] + alpha * th.roll(z, -1, 0)[:25]
                            cond_25 = viz_cond_emb[:25] if viz_cond_emb is not None else None
                            x_ints.append(_viz_decode(z_, cond_25))

                        # Generate random designs (use conditions from first 25 samples as reference)
                        z_rand = z_mean.unsqueeze(0).repeat([25, 1])
                        z_rand[:, idx[:n_active]] += z_std[:n_active] * th.randn_like(z_rand[:, idx[:n_active]])
                        cond_rand = viz_cond_emb[:25] if viz_cond_emb is not None else None
                        x_rand = _viz_decode(z_rand, cond_rand)

                        # Get performance predictions on training data
                        pz_train = z[:, :perf_dim]
                        if viz_cond_emb is not None:
                            p_pred_scaled = plvae.predictor(th.cat([pz_train, viz_cond_emb], dim=-1))
                        else:
                            p_pred_scaled = plvae.predictor(pz_train)

                        # Inverse transform to get true-scale values for plotting
                        p_actual = p_scaler.inverse_transform(p_train_scaled.cpu().numpy()).flatten()
                        p_predicted = p_scaler.inverse_transform(p_pred_scaled.cpu().numpy()).flatten()

                        # Move tensors to CPU for plotting
                        z_std_cpu = z_std.cpu().numpy()
                        xs_cpu = xs.cpu().numpy()

                    # Plot 1: Latent dimension statistics
                    plt.figure(figsize=(12, 6))
                    plt.subplot(211)
                    plt.bar(np.arange(len(z_std_cpu)), z_std_cpu)
                    plt.yscale("log")
                    plt.xlabel("Latent dimension index")
                    plt.ylabel("Standard deviation")
                    plt.title(f"Number of principal components = {n_active}")
                    plt.subplot(212)
                    plt.bar(np.arange(n_active), z_std_cpu[:n_active])
                    plt.yscale("log")
                    plt.xlabel("Latent dimension index")
                    plt.ylabel("Standard deviation")
                    plt.savefig(f"images/dim_{batches_done}.png")
                    plt.close()

                    # Plot 2: Interpolated designs
                    fig, axs = plt.subplots(25, 6, figsize=(12, 25))
                    for i_row, j in product(range(25), range(5)):
                        axs[i_row, j + 1].imshow(x_ints[j][i_row].reshape(design_shape))
                        axs[i_row, j + 1].axis("off")
                        axs[i_row, j + 1].set_aspect("equal")
                    for ax, alpha in zip(axs[0, 1:], [0, 0.25, 0.5, 0.75, 1]):
                        ax.set_title(rf"$\alpha$ = {alpha}")
                    for i_row in range(25):
                        axs[i_row, 0].imshow(xs_cpu[i_row].reshape(design_shape))
                        axs[i_row, 0].axis("off")
                        axs[i_row, 0].set_aspect("equal")
                    axs[0, 0].set_title("groundtruth")
                    fig.tight_layout()
                    plt.savefig(f"images/interp_{batches_done}.png")
                    plt.close()

                    # Plot 3: Random designs from latent space
                    fig, axs = plt.subplots(5, 5, figsize=(15, 7.5))
                    for k, (i_row, j) in enumerate(product(range(5), range(5))):
                        axs[i_row, j].imshow(x_rand[k].reshape(design_shape))
                        axs[i_row, j].axis("off")
                        axs[i_row, j].set_aspect("equal")
                    fig.tight_layout()
                    plt.suptitle("Gaussian random designs from latent space")
                    plt.savefig(f"images/norm_{batches_done}.png")
                    plt.close()

                    # Plot 4: Predicted vs actual performance
                    plt.figure(figsize=(8, 8))
                    plt.scatter(p_actual, p_predicted, alpha=0.5, s=20)
                    min_val = min(p_actual.min(), p_predicted.min())
                    max_val = max(p_actual.max(), p_predicted.max())
                    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="1:1 line")
                    plt.xlabel("Actual Performance")
                    plt.ylabel("Predicted Performance")
                    mse_value = np.mean((p_actual - p_predicted) ** 2)
                    plt.title(f"MSE: {mse_value:.4e}")
                    plt.grid(visible=True, alpha=0.3)
                    plt.legend()
                    plt.axis("equal")
                    plt.tight_layout()
                    plt.savefig(f"images/perf_pred_vs_actual_{batches_done}.png")
                    plt.close()

                    # Plot 5: Constraint satisfaction over time (new plot)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.axhline(
                        y=args.nmse_threshold_rec,
                        color="blue",
                        linestyle="--",
                        label=f"rec threshold ({args.nmse_threshold_rec})",
                    )
                    ax.axhline(
                        y=args.nmse_threshold_perf,
                        color="orange",
                        linestyle="--",
                        label=f"perf threshold ({args.nmse_threshold_perf})",
                    )
                    ax.scatter([0], [plvae.nmse_rec], color="blue", s=100, marker="o", label=f"current rec NMSE")
                    ax.scatter([1], [plvae.nmse_perf], color="orange", s=100, marker="o", label=f"current perf NMSE")
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(["Reconstruction", "Performance"])
                    ax.set_ylabel("NMSE")
                    ax.set_title(f"Constraint Status - Volume Active: {plvae.vol_active}")
                    ax.legend(loc="upper right")
                    ax.set_ylim(0, max(0.1, plvae.nmse_rec * 1.5, plvae.nmse_perf * 1.5))
                    plt.tight_layout()
                    plt.savefig(f"images/constraint_status_{batches_done}.png")
                    plt.close()

                    # Log plots to wandb
                    wandb.log(
                        {
                            "dim_plot": wandb.Image(f"images/dim_{batches_done}.png"),
                            "interp_plot": wandb.Image(f"images/interp_{batches_done}.png"),
                            "norm_plot": wandb.Image(f"images/norm_{batches_done}.png"),
                            "perf_pred_vs_actual": wandb.Image(f"images/perf_pred_vs_actual_{batches_done}.png"),
                            "constraint_status": wandb.Image(f"images/constraint_status_{batches_done}.png"),
                        }
                    )

        # ---- Validation ----
        with th.no_grad():
            plvae.eval()
            val_rec = val_perf = val_vol = 0.0
            val_nmse_rec = val_nmse_perf = 0.0
            n = 0
            for batch_v in val_loader:
                x_v = batch_v[0].to(device)
                c_v = batch_v[1].to(device)
                p_v = batch_v[2].to(device)
                ic_v = batch_v[3].to(device) if len(batch_v) > 3 else None
                val_batch = (x_v, c_v, p_v, ic_v) if ic_v is not None else (x_v, c_v, p_v)
                _ = plvae.loss(val_batch)  # Computes and stores metrics
                bsz = x_v.size(0)
                val_rec += plvae.rec_loss * bsz
                val_perf += plvae.perf_loss * bsz
                val_vol += plvae.vol_loss * bsz
                val_nmse_rec += plvae.nmse_rec * bsz
                val_nmse_perf += plvae.nmse_perf * bsz
                n += bsz
            val_rec /= n
            val_perf /= n
            val_vol /= n
            val_nmse_rec /= n
            val_nmse_perf /= n

        # Trigger pruning check at end of epoch
        plvae.epoch_report(epoch=epoch, callbacks=[], batch=None, loss=loss, pbar=None)

        if args.track:
            val_log_dict = {
                "epoch": epoch,
                "val_rec": val_rec,
                "val_perf": val_perf,
                "val_vol_loss": val_vol,
                "val_nmse_rec": val_nmse_rec,
                "val_nmse_perf": val_nmse_perf,
            }
            wandb.log(val_log_dict, commit=True)

        th.cuda.empty_cache()
        plvae.train()

        # Save models at end of training
        if args.save_model and epoch == args.n_epochs - 1:
            ckpt_plvae = {
                "epoch": epoch,
                "encoder": plvae.encoder.state_dict(),
                "decoder": plvae.decoder.state_dict(),
                "predictor": plvae.predictor.state_dict(),
                "optimizer": plvae.optim.state_dict(),
                "pruning_mask": plvae._p.cpu(),
                "pruning_frozen_z": plvae._z.cpu(),
                "args": vars(args),
            }
            if plvae.condition_encoder is not None:
                ckpt_plvae["condition_encoder"] = plvae.condition_encoder.state_dict()
            th.save(ckpt_plvae, "constrained_vanilla_plvae.pth")
            if args.track:
                artifact = wandb.Artifact(f"{args.problem_id}_{args.algo}", type="model")
                artifact.add_file("constrained_vanilla_plvae.pth")
                alias = f"seed_{args.seed}_rec{args.nmse_threshold_rec}_perf{args.nmse_threshold_perf}"
                if args.condition_filter_key is not None:
                    if args.condition_filter_range is not None:
                        lo, hi = args.condition_filter_range
                        alias += f"_{args.condition_filter_key}_{lo}-{hi}"
                    elif args.condition_filter_value is not None:
                        alias += f"_{args.condition_filter_key}_{args.condition_filter_value}"
                wandb.log_artifact(artifact, aliases=[alias])

    if args.track:
        wandb.finish()
