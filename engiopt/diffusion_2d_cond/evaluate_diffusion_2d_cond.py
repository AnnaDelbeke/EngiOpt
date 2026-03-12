"""Evaluation for the Diffusion 2d_cond w/ seed looping and CSV saving."""

from __future__ import annotations

import dataclasses
import itertools
import os

from diffusers import UNet2DConditionModel
from engibench.utils.all_problems import BUILTIN_PROBLEMS
import numpy as np
import pandas as pd
import torch as th
import tyro

from engiopt import metrics
from engiopt.dataset_sample_conditions import sample_conditions
from engiopt.diffusion_2d_cond.diffusion_2d_cond import beta_schedule
from engiopt.diffusion_2d_cond.diffusion_2d_cond import DiffusionSampler
import wandb


@dataclasses.dataclass
class Args:
    """Command-line arguments for a single-seed Diffusion 2D Conditional evaluation."""

    problem_id: str = "beams2d"
    """Problem identifier."""
    seed: int = 1
    """Random seed to run."""
    wandb_project: str = "engiopt"
    """Wandb project name."""
    wandb_entity: str | None = None
    """Wandb entity name."""
    n_samples: int = 50
    """Number of generated samples per seed."""
    sigma: float | None = None
    """Kernel bandwidth for MMD and DPP metrics. If None, uses median heuristic from reference data."""
    output_csv: str = "diffusion_2d_cond_{problem_id}_metrics.csv"
    """Output CSV path template; may include {problem_id}."""

    # LV metrics (optional) - provide seed and at least one threshold list to enable
    lvae_seed: int | None = None
    """LVAE seed. If provided along with thresholds, computes LV-MMD and LV-DPP."""
    lvae_rec_thresholds: str = ""
    """Comma-separated LVAE reconstruction NMSE thresholds (e.g. '0.0005,0.001,0.005')."""
    lvae_perf_thresholds: str = ""
    """Comma-separated LVAE performance NMSE thresholds (e.g. '0.001,1000')."""

    # LV metric suite (optional) — requires lvae_seed + thresholds above
    compute_lv_suite: bool = False
    """Compute full LV metric suite (projection residual, dual gap). Requires decoder."""

    # WandB logging
    log_to_wandb: bool = False
    """Log evaluation metrics to the original WandB training run."""


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Parse comma-separated threshold strings into lists of floats
    rec_thresholds = (
        [float(x) for x in args.lvae_rec_thresholds.split(",") if x.strip()] if args.lvae_rec_thresholds else []
    )
    perf_thresholds = (
        [float(x) for x in args.lvae_perf_thresholds.split(",") if x.strip()] if args.lvae_perf_thresholds else []
    )

    seed = args.seed
    problem = BUILTIN_PROBLEMS[args.problem_id]()
    problem.reset(seed=seed)

    # Seeding for reproducibility
    th.manual_seed(seed)
    rng = np.random.default_rng(seed)
    th.backends.cudnn.deterministic = True

    # Select device
    if th.backends.mps.is_available():
        device = th.device("mps")
    elif th.cuda.is_available():
        device = th.device("cuda")
    else:
        device = th.device("cpu")

    ### Set up testing conditions ###
    conditions_tensor, sampled_conditions, sampled_designs_np, selected_indices = sample_conditions(
        problem=problem,
        n_samples=args.n_samples,
        device=device,
        seed=seed,
    )

    # Add channel dim
    conditions_tensor = conditions_tensor.unsqueeze(1)

    ### Set Up Diffusion Model ###
    if args.wandb_entity is not None:
        artifact_path = f"{args.wandb_entity}/{args.wandb_project}/{args.problem_id}_diffusion_2d_cond_model:seed_{seed}"
    else:
        artifact_path = f"{args.wandb_project}/{args.problem_id}_diffusion_2d_cond_model:seed_{seed}"

    api = wandb.Api()
    artifact = api.artifact(artifact_path, type="model")

    class RunRetrievalError(ValueError):
        def __init__(self):
            super().__init__("Failed to retrieve the run")

    run = artifact.logged_by()
    if run is None or not hasattr(run, "config"):
        raise RunRetrievalError

    artifact_dir = artifact.download()
    ckpt_path = os.path.join(artifact_dir, "model.pth")
    ckpt = th.load(ckpt_path, map_location=device)

    # Build UNet
    model = UNet2DConditionModel(
        sample_size=problem.design_space.shape,
        in_channels=1,
        out_channels=1,
        cross_attention_dim=64,
        block_out_channels=(32, 64, 128, 256),
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        layers_per_block=run.config["layers_per_block"],
        transformer_layers_per_block=1,
        encoder_hid_dim=len(problem.conditions_keys),
        only_cross_attention=True,
    ).to(device)

    # Noise schedule
    options = {
        "cosine": run.config["noise_schedule"] == "cosine",
        "exp_biasing": run.config["noise_schedule"] == "exp",
        "exp_bias_factor": 1,
    }
    betas = beta_schedule(
        t=run.config["num_timesteps"],
        start=1e-4,
        end=0.02,
        scale=1.0,
        options=options,
    )
    ddm_sampler = DiffusionSampler(run.config["num_timesteps"], betas)

    model.load_state_dict(ckpt["model"])
    model.eval()

    # Generate and reshape
    design_shape: tuple = problem.design_space.shape
    gen_designs = th.randn((args.n_samples, 1, *design_shape), device=device)
    assert run.config["num_timesteps"] is not None
    for i in reversed(range(run.config["num_timesteps"])):
        t = th.full((args.n_samples,), i, device=device, dtype=th.long)
        gen_designs = ddm_sampler.sample_timestep(model, gen_designs, t, conditions_tensor)

    gen_designs = gen_designs.squeeze(1)
    gen_designs_np = gen_designs.detach().cpu().numpy().reshape(args.n_samples, *problem.design_space.shape)
    gen_designs_np = np.clip(gen_designs_np, 1e-3, 1.0)

    # Compute metrics
    metrics_dict = metrics.metrics(
        problem,
        gen_designs_np,
        sampled_designs_np,
        sampled_conditions,
        sigma=args.sigma,
    )
    # Add metadata to metrics
    metrics_dict.update(
        {
            "seed": seed,
            "problem_id": args.problem_id,
            "model_id": "diffusion_2d_cond",
            "n_samples": args.n_samples,
        }
    )

    # Compute LV metrics for all (rec, perf) threshold combinations
    if args.lvae_seed is not None and rec_thresholds and perf_thresholds:
        from sklearn.decomposition import PCA

        from engiopt.vanilla_lvae.utils import encode_designs
        from engiopt.vanilla_lvae.utils import load_lvae_encoder

        if args.compute_lv_suite:
            from engiopt.lv_metrics import lv_dual_projection_gap
            from engiopt.lv_metrics import lv_project_designs
            from engiopt.vanilla_lvae.utils import load_lvae_encoder_decoder

        metrics_dict["lvae_seed"] = args.lvae_seed

        # Log base metrics to WandB immediately before LVAE loop
        if args.log_to_wandb and run is not None:
            run.summary["eval/mmd"] = metrics_dict.get("mmd")
            run.summary["eval/dpp"] = metrics_dict.get("dpp")
            run.summary["eval/iog"] = metrics_dict.get("iog")
            run.summary["eval/cog"] = metrics_dict.get("cog")
            run.summary["eval/fog"] = metrics_dict.get("fog")
            run.summary["eval/iog_median"] = metrics_dict.get("iog_median")
            run.summary["eval/cog_median"] = metrics_dict.get("cog_median")
            run.summary["eval/fog_median"] = metrics_dict.get("fog_median")
            run.summary["eval/iog_iqr"] = metrics_dict.get("iog_iqr")
            run.summary["eval/cog_iqr"] = metrics_dict.get("cog_iqr")
            run.summary["eval/fog_iqr"] = metrics_dict.get("fog_iqr")
            run.summary["eval/mmd_sigma"] = metrics_dict.get("mmd_sigma")
            run.summary.update()
            print("  Logged base metrics to WandB.")

        for rec_thresh, perf_thresh in itertools.product(rec_thresholds, perf_thresholds):
            suffix = f"_rec{rec_thresh}_perf{perf_thresh}"
            print(f"Loading LVAE (seed={args.lvae_seed}, rec={rec_thresh}, perf={perf_thresh})...")
            try:
                decoder = None
                if args.compute_lv_suite:
                    encoder, decoder, lvae_config = load_lvae_encoder_decoder(
                        problem_id=args.problem_id,
                        seed=args.lvae_seed,
                        rec_threshold=rec_thresh,
                        perf_threshold=perf_thresh,
                        wandb_project=args.wandb_project,
                        wandb_entity=args.wandb_entity,
                        device=device,
                    )
                else:
                    encoder, lvae_config = load_lvae_encoder(
                        problem_id=args.problem_id,
                        seed=args.lvae_seed,
                        rec_threshold=rec_thresh,
                        perf_threshold=perf_thresh,
                        wandb_project=args.wandb_project,
                        wandb_entity=args.wandb_entity,
                        device=device,
                    )

                # Encode designs to latent space
                z_gen = encode_designs(encoder, gen_designs_np, device)
                z_data = encode_designs(encoder, sampled_designs_np, device)
                n_active = int((np.var(z_data, axis=0) > 1e-8).sum())

                lv_sigma = metrics.compute_median_sigma(z_data)
                lv_mmd_val = metrics.mmd(z_gen, z_data, sigma=lv_sigma)
                lv_dpp_val = metrics.dpp_diversity(z_gen, sigma=lv_sigma)

                metrics_dict[f"lv_mmd{suffix}"] = lv_mmd_val
                metrics_dict[f"lv_dpp{suffix}"] = lv_dpp_val
                metrics_dict[f"lv_sigma{suffix}"] = lv_sigma
                metrics_dict[f"lvae_n_active_dims{suffix}"] = n_active

                # LV metric suite: projection residual
                if args.compute_lv_suite and decoder is not None:
                    proj_residuals, _ = lv_project_designs(encoder, decoder, gen_designs_np, device)
                    metrics_dict[f"lv_proj_residual_mean{suffix}"] = float(proj_residuals.mean())
                    print(f"  LV-ProjRes: {proj_residuals.mean():.6f}")

                    # Dual-LVAE projection gap: compare perf vs recon-only projections
                    recon_only_thresh = 1000.0
                    if perf_thresh != recon_only_thresh:
                        try:
                            enc_ro, dec_ro, _ = load_lvae_encoder_decoder(
                                problem_id=args.problem_id,
                                seed=args.lvae_seed,
                                rec_threshold=rec_thresh,
                                perf_threshold=recon_only_thresh,
                                wandb_project=args.wandb_project,
                                wandb_entity=args.wandb_entity,
                                device=device,
                            )
                            dual = lv_dual_projection_gap(encoder, decoder, enc_ro, dec_ro, gen_designs_np, device)
                            metrics_dict[f"lv_dual_gap_mean{suffix}"] = dual["dual_gap_mean"]
                            metrics_dict[f"lv_residual_perf_mean{suffix}"] = dual["residual_perf_mean"]
                            metrics_dict[f"lv_residual_recon_mean{suffix}"] = dual["residual_recon_mean"]
                            print(f"  LV-DualGap: {dual['dual_gap_mean']:.6f}")
                        except Exception as e:
                            print(f"  Dual gap failed (recon-only LVAE not found): {e}")

                # PCA baseline: project to same dimensionality as LVAE active dims
                n_pca = min(n_active, args.n_samples - 1)
                if n_pca > 0:
                    flat_gen = gen_designs_np.reshape(gen_designs_np.shape[0], -1)
                    flat_data = sampled_designs_np.reshape(sampled_designs_np.shape[0], -1)
                    pca = PCA(n_components=n_pca)
                    pca.fit(flat_data)
                    pca_gen = pca.transform(flat_gen)
                    pca_data = pca.transform(flat_data)

                    pca_sigma = metrics.compute_median_sigma(pca_data)
                    pca_mmd_val = metrics.mmd(pca_gen, pca_data, sigma=pca_sigma)
                    pca_dpp_val = metrics.dpp_diversity(pca_gen, sigma=pca_sigma)

                    metrics_dict[f"pca_mmd{suffix}"] = pca_mmd_val
                    metrics_dict[f"pca_dpp{suffix}"] = pca_dpp_val
                    metrics_dict[f"pca_sigma{suffix}"] = pca_sigma
                    print(f"  PCA({n_pca}): MMD={pca_mmd_val:.6f}, DPP={pca_dpp_val:.6e}")

                print(f"  LV-MMD: {lv_mmd_val:.6f}, LV-DPP: {lv_dpp_val:.6e}, Active: {n_active}")

                # Log this combo immediately so a timeout can't erase it
                if args.log_to_wandb and run is not None:
                    prefix = f"eval/lv_rec{rec_thresh}_perf{perf_thresh}_lvae{args.lvae_seed}"
                    run.summary[f"{prefix}/lv_mmd"] = metrics_dict.get(f"lv_mmd{suffix}")
                    run.summary[f"{prefix}/lv_dpp"] = metrics_dict.get(f"lv_dpp{suffix}")
                    run.summary[f"{prefix}/lv_sigma"] = metrics_dict.get(f"lv_sigma{suffix}")
                    run.summary[f"{prefix}/n_active_dims"] = metrics_dict.get(f"lvae_n_active_dims{suffix}")
                    run.summary[f"{prefix}/pca_mmd"] = metrics_dict.get(f"pca_mmd{suffix}")
                    run.summary[f"{prefix}/pca_dpp"] = metrics_dict.get(f"pca_dpp{suffix}")
                    run.summary[f"{prefix}/pca_sigma"] = metrics_dict.get(f"pca_sigma{suffix}")
                    if args.compute_lv_suite:
                        run.summary[f"{prefix}/lv_proj_residual_mean"] = metrics_dict.get(f"lv_proj_residual_mean{suffix}")
                        run.summary[f"{prefix}/lv_dual_gap_mean"] = metrics_dict.get(f"lv_dual_gap_mean{suffix}")
                        run.summary[f"{prefix}/lv_residual_perf_mean"] = metrics_dict.get(f"lv_residual_perf_mean{suffix}")
                        run.summary[f"{prefix}/lv_residual_recon_mean"] = metrics_dict.get(
                            f"lv_residual_recon_mean{suffix}"
                        )
                    run.summary.update()

            except Exception as e:
                print(f"  Failed for rec={rec_thresh}, perf={perf_thresh}: {e}")

    # Save per-sample data to .npz for detailed analysis (e.g., distribution plots)
    per_sample_keys = ["iog_list", "cog_list", "fog_list", "viol_list"]
    per_sample_data = {k: np.array(metrics_dict[k]) for k in per_sample_keys if k in metrics_dict}
    cond_array = np.column_stack([np.array(sampled_conditions[c]) for c in sampled_conditions.column_names])
    per_sample_data["conditions"] = cond_array
    per_sample_data["condition_names"] = np.array(sampled_conditions.column_names)
    per_sample_data["gen_designs"] = gen_designs_np
    per_sample_data["ref_designs"] = sampled_designs_np
    if per_sample_data:
        npz_path = args.output_csv.format(problem_id=args.problem_id).replace(".csv", f"_seed{seed}_per_sample.npz")
        np.savez(npz_path, **per_sample_data)
        print(f"  Per-sample data saved to {npz_path}")

    # Remove list-valued keys before CSV serialization (keep only scalars)
    csv_dict = {k: v for k, v in metrics_dict.items() if not isinstance(v, list)}

    # Append result row to CSV
    metrics_df = pd.DataFrame([csv_dict])
    out_path = args.output_csv.format(problem_id=args.problem_id)
    write_header = not os.path.exists(out_path)
    metrics_df.to_csv(out_path, mode="a", header=write_header, index=False)

    # Log base metrics to WandB (only reached if no LVAE loop or LVAE loop skipped)
    if args.log_to_wandb and run is not None and not (args.lvae_seed is not None and rec_thresholds and perf_thresholds):
        run.summary["eval/mmd"] = metrics_dict.get("mmd")
        run.summary["eval/dpp"] = metrics_dict.get("dpp")
        run.summary["eval/iog"] = metrics_dict.get("iog")
        run.summary["eval/cog"] = metrics_dict.get("cog")
        run.summary["eval/fog"] = metrics_dict.get("fog")
        run.summary["eval/iog_median"] = metrics_dict.get("iog_median")
        run.summary["eval/cog_median"] = metrics_dict.get("cog_median")
        run.summary["eval/fog_median"] = metrics_dict.get("fog_median")
        run.summary["eval/iog_iqr"] = metrics_dict.get("iog_iqr")
        run.summary["eval/cog_iqr"] = metrics_dict.get("cog_iqr")
        run.summary["eval/fog_iqr"] = metrics_dict.get("fog_iqr")
        run.summary["eval/mmd_sigma"] = metrics_dict.get("mmd_sigma")
        run.summary.update()
        print(f"  Logged metrics to WandB run: {run.name}")

    print(f"Seed {seed} done; appended to {out_path}")
