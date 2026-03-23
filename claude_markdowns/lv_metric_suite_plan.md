# Latent-Space Metric Suite for Benchmarking Generative Models in EngiBench

## Motivation

Current simulation-free metrics (pixel-space MMD, DPP) are unsatisfying for engineering design:
- **DPP** rewards noisy garbage because random noise is maximally diverse in pixel space
- **Pixel MMD** treats all pixel differences equally, ignoring which variations matter for performance
- Neither metric correlates reliably with optimization-based metrics (IOG/COG/FOG) that require expensive simulation

A trained LVAE provides a **learned design manifold** with physics-informed structure. The least-volume objective produces a tight enclosure of the data, giving us a principled notion of "inside vs outside" and calibrated latent distances. This document plans a tiered suite of latent-space metrics and a validation protocol to verify they correlate with simulation-based ground truth.

**Standing rule**: All latent-space metrics and visualizations operate exclusively on **active (unpruned) dimensions**. Pruned dimensions are clamped to constant frozen values and carry no information. The active-dim mask is accessible via `get_active_mask(encoder)` from `vanilla_lvae/utils.py`, which reads `PrunedEncoder._p` — the authoritative pruning mask saved during training.

---

## Tier 1: Low-Hanging Fruit — DONE

These build directly on existing infrastructure in `metrics.py`, `lv_metrics.py`, and `vanilla_lvae/utils.py`.

### 1.1 LV-MMD (Latent-Volume MMD) — DONE
**What**: MMD computed on LVAE-encoded latent representations instead of pixel space.
**Why it's useful**: Latent distances reflect design-meaningful variation. Two designs that look different in pixel space but are functionally similar will be close in latent space.
**Status**: Implemented in all eval scripts. Commit `4afe8c7`.

### 1.2 LV-DPP (Latent-Volume DPP) — DONE
**What**: DPP diversity computed in LVAE latent space rather than pixel space.
**Why it's useful**: Diversity is measured along axes of meaningful design variation. Noise that pixel-DPP rewards would map to similar (low-information) latent codes, naturally down-weighting garbage diversity.
**Status**: Implemented alongside LV-MMD. Commit `4afe8c7`.

### 1.3 Reconstruction Residual Distribution — DONE
**What**: Per-sample MSE between generated designs and their LVAE projections (encode-decode). Summary stats: mean, median, 90th percentile, std.
**Why it's useful**: High residual = design is far from the learned manifold. Provides a simulation-free quality proxy.
**Key insight**: This is an anomaly score, not a diversity metric. Pair with LV-DPP for a quality-diversity characterization.
**Status**: `lv_reconstruction_residual_stats()` in `lv_metrics.py`. Integrated in all eval scripts under `--compute-lv-suite` flag (requires decoder). Per-sample residuals saved to NPZ.

---

## Tier 2: Moderate Effort, Strong Utility

### 2.1 Mahalanobis Typicality Score — DONE
**What**: Mahalanobis distance of each encoded design from the training latent centroid, using the training set's covariance structure.
**Why it's useful**: Designs near the center are "typical"; designs at the boundary are "novel but plausible"; designs far outside are "garbage." Degrades gracefully — unlike a hard reconstruction-error filter, novel-but-valid designs get moderate scores.
**Reports**: `mahal_mean`, `mahal_median`, `plausibility_rate` (fraction below chi-squared 95th percentile).
**Status**: `lv_mahalanobis_typicality()` in `lv_metrics.py`. Reference stats built from **full training set** (not sampled test designs). Integrated in all eval scripts. Per-sample distances saved to NPZ.

### 2.2 Latent Mode Coverage — SKIPPED
**Decision**: Clustering-based mode coverage was deemed unlikely to be informative for the target problems. The novelty-quality Pareto front (3.1) and volume coverage ratio (3.2) provide better continuous alternatives for detecting mode collapse.

### 2.3 Conditional LV-MMD — DEFINED, NOT WIRED
**What**: LV-MMD computed per condition bin, then averaged.
**Why deferred**: Performance-constrained latent spaces already capture condition structure. Some problems have image conditions or many conditions where fair partitioning is hard. The wrapper `lv_conditional_mmd()` exists in `lv_metrics.py` but is not called from eval scripts. `evaluate_lv_only.py` already calls `metrics.conditional_mmd()` directly on latent codes.

---

## Tier 3: Higher Effort, Research-Grade Utility

### 3.1 Novelty-Quality Pareto Front — DONE
**What**: For each generated design, compute:
- **Quality**: negative Mahalanobis distance (higher = more plausible)
- **Novelty**: minimum L2 distance to any training point in latent space (higher = more novel)

Compute the 2D hypervolume indicator as a scalar summary.
**Why it's useful**: Directly addresses the tension between quality and diversity. A good generator pushes the Pareto front.
**Status**: `lv_novelty_quality_pareto()` in `lv_metrics.py`. Uses `pymoo.indicators.hv.HV` for hypervolume. Auto-computed reference point. Integrated in all eval scripts. Per-sample novelty/quality saved to NPZ.

### 3.2 Latent Volume Coverage Ratio — DONE
**What**: Log-det ratio of generated vs training covariance in active latent space.
**Why it's useful**: Continuous measure of how much of the training latent volume the generator covers. Near 0 = similar spread; negative = more concentrated; positive = wider.
**Status**: `lv_volume_coverage_ratio()` in `lv_metrics.py`. Integrated in all eval scripts.

### 3.3 Dual-LVAE Projection Gap — DONE (pre-existing)
**What**: Compare projections from a performance-constrained LVAE vs. a reconstruction-only LVAE.
**Why it's useful**: Isolates the performance-relevant component of deviation from the manifold.
**Status**: `lv_dual_projection_gap()` in `lv_metrics.py`. Integrated under `--compute-lv-suite`. Per-sample gap arrays saved to NPZ.

### 3.4 Performance-Weighted Latent DPP — SKIPPED
**Decision**: Practicality concerns — requires performance-predicting LVAE variant and temperature tuning. May revisit if the basic metrics prove insufficient for ranking generators.

---

## Implementation Architecture

### Core module: `engiopt/lv_metrics.py`

All latent-space metric functions live here. Key components:

| Component | Purpose |
|---|---|
| `LatentReferenceStats` (NamedTuple) | Precomputed training-set statistics: mean, cov, cov_inv, log_det, z_train |
| `compute_latent_reference_stats(z_train)` | Factory that builds `LatentReferenceStats` from encoded training designs |
| `lv_reconstruction_residual_stats(enc, dec, designs, device)` | Per-sample + summary reconstruction MSE |
| `lv_mahalanobis_typicality(z_gen, ref_stats)` | Mahalanobis distances + plausibility rate |
| `lv_conditional_mmd(z_gen, z_ref, conditions)` | Wrapper around `metrics.conditional_mmd` with `lv_` prefix |
| `lv_novelty_quality_pareto(z_gen, ref_stats)` | Novelty/quality arrays + hypervolume indicator |
| `lv_volume_coverage_ratio(z_gen, ref_stats)` | Log-det covariance ratio |
| `lv_project_designs(enc, dec, designs, device)` | Per-sample projection residual (pre-existing) |
| `lv_dual_projection_gap(enc_p, dec_p, enc_r, dec_r, designs, device)` | Dual-LVAE gap (pre-existing) |

### Reference statistics

`ref_stats` is built from the **full training set** (`problem.dataset["train"]["optimal_design"]`), not the sampled test designs. This ensures stable covariance estimates independent of `n_samples`. The encoding is batched (batch_size=256) so large training sets are handled efficiently.

### Eval script integration

All 4 eval scripts (`evaluate_cgan_cnn_2d.py`, `evaluate_gan_cnn_2d.py`, `evaluate_diffusion_2d_cond.py`, `evaluate_vqgan.py`) follow the same pattern inside their LVAE loop:

1. **Always computed** (encoder only): LV-MMD, LV-DPP, Mahalanobis, novelty-quality Pareto, volume coverage, PCA baseline
2. **Gated by `--compute-lv-suite`** (needs decoder): reconstruction residual stats, dual-LVAE projection gap

### Output destinations

| Destination | What gets saved |
|---|---|
| **CSV** | All scalar metrics (suffixed by LVAE config) |
| **NPZ** | Per-sample arrays: `iog_list`, `cog_list`, `fog_list`, `viol_list`, `conditions`, `gen_designs`, `ref_designs`, and per-LVAE-config: `lv_mahal_distances`, `lv_novelty`, `lv_quality`, `lv_residuals`, `lv_dual_gap`, `lv_residual_perf`, `lv_residual_recon` |
| **WandB** | All scalars under `eval/lv_rec{R}_perf{P}_lvae{S}/` prefix |

---

## Validation Protocol: Do These Metrics Correlate With Simulation?

The central question: **do simulation-free latent metrics predict which generators actually produce better designs (as measured by IOG/COG/FOG)?**

### Step 1: Collect Ground Truth Across Models
For each EngiBench problem (beams2d, heatconduction2d, etc.), evaluate multiple generators (CGAN, Diffusion, LVAE, etc.) at multiple seeds. For each (problem, generator, seed), compute:
- **Simulation metrics**: IOG, COG, FOG, violation rate (from existing `metrics()`)
- **Pixel metrics**: MMD, DPP (existing)
- **Latent metrics**: All implemented metrics above

This gives a table of ~50-100 rows (models x problems x seeds) with both simulation and proxy metrics.

### Step 2: Rank Correlation Analysis
For each proxy metric, compute **Spearman rank correlation** with each simulation metric:

| Proxy Metric | vs. IOG | vs. COG | vs. FOG | vs. Violation Rate |
|---|---|---|---|---|
| Pixel MMD | ? | ? | ? | ? |
| Pixel DPP | ? | ? | ? | ? |
| LV-MMD | ? | ? | ? | ? |
| LV-DPP | ? | ? | ? | ? |
| Recon Residual (mean) | ? | ? | ? | ? |
| Mahalanobis (mean) | ? | ? | ? | ? |
| Plausibility Rate | ? | ? | ? | ? |
| NQ Hypervolume | ? | ? | ? | ? |
| Volume Coverage Ratio | ? | ? | ? | ? |

**Hypothesis**: LV-MMD should correlate more strongly with COG/FOG than pixel MMD. LV-DPP should not anti-correlate with quality (as pixel DPP might). Plausibility rate should correlate with violation rate.

### Step 3: Within-Problem vs. Across-Problem Correlation
Compute correlations both:
- **Within each problem** (do metrics rank generators correctly for beams2d specifically?)
- **Across all problems** (do metrics generalize?)

Latent metrics should be strongest within-problem (since the LVAE is problem-specific). If they also work across problems, that's a bonus.

### Step 4: Sensitivity to LVAE Quality
Train LVAEs of varying quality (different epochs, latent dims, pruning strategies). Check whether metric-simulation correlations are robust to LVAE hyperparameters or degrade when the LVAE itself is poor.

**Critical question**: How good does the LVAE need to be for latent metrics to be useful? If correlations hold even for mediocre LVAEs, the metric suite is practical. If they require a perfectly tuned LVAE, the approach is fragile.

### Step 5: Bootstrap Confidence Intervals
With small sample sizes (n=10 per evaluation), estimates are noisy. Use bootstrapping:
- Resample the (model, seed) pairs 1000 times
- Compute rank correlations on each resample
- Report 95% confidence intervals on the correlation coefficients

---

## Implementation Roadmap

### Phase 1 (Quick Wins) — DONE
1. ~~Fix eval scripts to slice latent codes to active dims~~ (commit `4afe8c7`)
2. ~~Add `get_active_mask()` helper~~ (commit `4afe8c7`)
3. ~~Fix visualize_latent_space.py and .ipynb~~ (commit `4afe8c7`)
4. ~~Log design_shape to wandb config~~ (commit `2494a89`)
5. ~~Add reconstruction residual summary stats~~ (`lv_reconstruction_residual_stats()`)

### Phase 2 (Core Suite) — DONE
1. ~~Implement `LatentReferenceStats` + factory from full training set~~
2. ~~Implement Mahalanobis typicality score~~
3. ~~Implement novelty-quality Pareto front with hypervolume~~
4. ~~Implement volume coverage ratio~~
5. ~~Integrate all metrics into all 4 eval scripts~~
6. ~~Save per-sample arrays to NPZ~~
7. ~~Define conditional LV-MMD wrapper~~ (available but not wired into eval scripts)

### Phase 3 (Validation — next)
1. Run all metrics on existing evaluation results (WandB artifacts)
2. Collect the correlation table
3. Produce the rank correlation matrix
4. Write up results

### Phase 4 (Research Extensions — ongoing)
1. LVAE sensitivity analysis (Validation Step 4)
2. Conditional LV-MMD integration (if needed per problem)
3. Performance-weighted DPP (if basic metrics prove insufficient)

---

## Summary Table

| Metric | Tier | Status | Measures | Needs Decoder? | Expected Correlation Target |
|---|---|---|---|---|---|
| LV-MMD | 1 | **Done** | Distributional fidelity | No | COG, FOG |
| LV-DPP | 1 | **Done** | Meaningful diversity | No | Should NOT anti-correlate with quality |
| Recon Residual Stats | 1 | **Done** | On-manifold quality | Yes | IOG, violation rate |
| Mahalanobis Score | 2 | **Done** | Typicality/plausibility | No | IOG, violation rate |
| Mode Coverage | 2 | **Skipped** | — | — | — |
| Conditional LV-MMD | 2 | **Defined** | Per-condition fidelity | No | Conditional IOG/FOG |
| Novelty-Quality Pareto | 3 | **Done** | Quality-diversity tradeoff | No | Composite quality+diversity |
| Volume Coverage Ratio | 3 | **Done** | Latent space utilization | No | Mode coverage (continuous) |
| Dual Projection Gap | 3 | **Done** | Perf-relevant deviation | Yes | FOG specifically |
| Perf-Weighted DPP | 3 | **Skipped** | — | — | — |

---

## Open Questions

1. **LVAE as infrastructure**: Should each problem have one "reference LVAE" trained once on the full dataset, shared across all generator evaluations? This is the cleanest protocol but requires commitment to LVAE hyperparameters.

2. **Training data scope**: Should the reference LVAE train on all data or only high-performing designs? Training on all data gives broader coverage but may dilute performance-relevant structure. Training on high-performing designs makes the manifold more selective but risks the out-of-distribution issue discussed above.

3. **Sigma selection**: For latent-space MMD/DPP, should sigma be computed from the reference (training) latent codes or from the generated ones? Using reference sigma ensures consistency across generators; using generated sigma adapts to each generator's spread.

4. **Dimensionality dependence**: Active latent dims vary across problems and LVAE runs. Do metrics need to be normalized by dimensionality to be comparable across problems?
