# Evaluation Script Architecture & Refactoring Plan

## Current State

The 4 main eval scripts (`evaluate_cgan_cnn_2d.py`, `evaluate_gan_cnn_2d.py`, `evaluate_diffusion_2d_cond.py`, `evaluate_vqgan.py`) are 370-470 lines each and share ~70% identical code. Each script follows the same structure but inlines everything, making changes error-prone (every metric addition requires 4 identical edits + keeping WandB logging in sync).

### Script structure (same in all 4)

```
1. Args dataclass              (~40 lines, 95% identical)
2. Setup: problem, device, RNG (~15 lines, identical)
3. Conditions sampling          (~6 lines, identical)
4. Model loading from WandB     (~20-100 lines, UNIQUE per model)
5. Design generation            (~10 lines, UNIQUE per model)
6. Clip to boundaries           (1 line, identical)
7. Base metrics (IOG/COG/FOG)   (~16 lines, identical)
8. per_sample_data init         (1 line, identical)
9. LVAE loop:                   (~190 lines, identical)
   a. Load encoder
   b. Encode gen + ref designs, slice to active dims
   c. LV-MMD, LV-DPP
   d. Build ref_stats from full training set
   e. Mahalanobis, novelty-quality, volume coverage
   f. Per-sample array accumulation
   g. [optional] Load decoder → recon residual stats, dual gap
   h. PCA baseline
   i. WandB logging per combo
10. NPZ save                    (~15 lines, identical)
11. CSV save                    (~9 lines, identical)
```

Only steps 4 and 5 differ between scripts. Everything else is copy-pasted.

---

## Proposed Refactoring

### `engiopt/eval_utils.py` — shared evaluation utilities

Extract the identical code into reusable functions:

```python
# --- Setup ---
def setup_evaluation(args) -> tuple[Problem, th.device, ...]:
    """Parse thresholds, init problem, set device, sample conditions."""

# --- LVAE metric loop ---
def compute_lvae_metrics(
    encoder, gen_designs_np, sampled_designs_np, problem,
    rec_thresh, perf_thresh, suffix, args,
    metrics_dict, per_sample_data, device,
) -> None:
    """One iteration of the LVAE loop: LV-MMD/DPP, ref_stats, Mahalanobis,
    novelty-quality, volume coverage, optional suite, PCA baseline.
    Mutates metrics_dict and per_sample_data in place."""

# --- WandB logging ---
def log_base_metrics_to_wandb(run, metrics_dict) -> None:
    """Log simulation-level metrics (IOG/COG/FOG/MMD/DPP) to WandB summary."""

def log_lvae_combo_to_wandb(run, metrics_dict, suffix, prefix, compute_lv_suite) -> None:
    """Log one LVAE (rec, perf) combo's metrics to WandB summary."""

# --- Saving ---
def save_per_sample_npz(
    metrics_dict, per_sample_data, sampled_conditions,
    gen_designs_np, sampled_designs_np, output_csv, problem_id, seed,
) -> None:
    """Merge simulation per-sample lists + LVAE per-sample arrays → NPZ."""

def save_metrics_csv(metrics_dict, output_csv, problem_id, seed) -> None:
    """Filter to scalars, append one row to CSV."""
```

### After refactoring, each eval script becomes ~60-80 lines

```python
"""Evaluation for CGAN CNN 2D."""
from engiopt.eval_utils import (
    setup_evaluation, compute_lvae_metrics, log_base_metrics_to_wandb,
    log_lvae_combo_to_wandb, save_per_sample_npz, save_metrics_csv,
)

@dataclass
class Args(BaseEvaluationArgs):
    """Adds run_id for sweep support."""
    run_id: str | None = None

if __name__ == "__main__":
    args = tyro.cli(Args)
    problem, device, conditions_tensor, sampled_conditions, sampled_designs_np, ... = setup_evaluation(args)

    # --- Model-specific: load generator and generate designs ---
    conditions_tensor = conditions_tensor.unsqueeze(-1).unsqueeze(-1)
    ...load Generator from WandB...
    gen_designs = model(z, conditions_tensor)
    gen_designs_np = ...clip...

    # --- Shared: metrics, LVAE loop, save ---
    metrics_dict = metrics.metrics(problem, gen_designs_np, sampled_designs_np, sampled_conditions, sigma=args.sigma)
    metrics_dict.update({"seed": seed, "problem_id": args.problem_id, "model_id": "cgan_cnn_2d", ...})
    per_sample_data = {}

    if args.lvae_seed and rec_thresholds and perf_thresholds:
        log_base_metrics_to_wandb(run, metrics_dict)
        for rec_thresh, perf_thresh in itertools.product(rec_thresholds, perf_thresholds):
            suffix = f"_rec{rec_thresh}_perf{perf_thresh}"
            compute_lvae_metrics(encoder, gen_designs_np, ..., metrics_dict, per_sample_data, device)
            log_lvae_combo_to_wandb(run, metrics_dict, suffix, ...)

    save_per_sample_npz(metrics_dict, per_sample_data, ...)
    save_metrics_csv(metrics_dict, ...)
```

---

## What's duplicated vs. unique (line counts)

| Section | Lines per script | Duplicated across 4? |
|---|---|---|
| Args (common fields) | 35 | Yes (except model_id default, run_id) |
| Setup (device, problem, RNG) | 15 | Yes |
| Threshold parsing | 6 | Yes |
| Conditions sampling | 6 | Yes |
| **Model loading + generation** | **20-100** | **No — unique per model** |
| Base metrics computation | 16 | Yes |
| LVAE loop body | 190 | Yes |
| NPZ save | 15 | Yes |
| CSV save | 9 | Yes |
| **Total shared** | **~290** | **x4 = ~1160 duplicated lines** |
| **Total unique** | **~50-100** | — |

---

## Minor inconsistencies to fix during refactoring

1. **Base metrics WandB timing**: cgan_cnn_2d and diffusion_2d_cond log base metrics BEFORE the LVAE loop. gan_cnn_2d logs AFTER. diffusion_2d_cond also has a fallback at the end. Should standardize to "before LVAE loop" since that's the timeout-safe strategy.

2. **VQGAN diagnostic prints**: `evaluate_vqgan.py` has extra NaN/Inf diagnostic prints for z_gen/z_data and uses `traceback.print_exc()`. These are debugging artifacts that can be removed or gated behind a `--verbose` flag.

3. **`lv_project_designs` import**: The `compute_lv_suite` block still imports `lv_project_designs` even though it's now called indirectly via `lv_reconstruction_residual_stats`. The unused import can be removed.

4. **evaluate_lv_only.py**: Currently unused per user, but its `load_generator_and_generate()` factory pattern is a good model for how each eval script could separate model-specific logic.

---

## Suggested order of operations

1. **Create `engiopt/eval_utils.py`** with the shared functions listed above
2. **Refactor `evaluate_cgan_cnn_2d.py`** first as the template
3. **Refactor remaining 3 scripts** following the same pattern
4. **Verify** all 4 produce identical CSV/NPZ/WandB output by running on one problem/seed
5. **Clean up** `evaluate_lv_only.py` if it becomes useful again, or mark as deprecated
