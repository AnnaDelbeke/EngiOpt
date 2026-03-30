# Lipschitz Bounds in Performance LVAE: Research Notes

## Core Finding: Sigmoid Creates a Hidden 1/4 Lipschitz Factor

The `TrueSNDecoder2D` previously applied `sigmoid(x * lipschitz_scale)` as its final operation. Because spectral normalization is applied layer-by-layer, the overall Lipschitz bound is the product of per-layer bounds. Sigmoid's maximum derivative is 1/4 (at x=0), so:

- **Decoder Lipschitz bound** = `lipschitz_scale / 4` (not `lipschitz_scale`)
- **MLP Lipschitz bound** = `lipschitz_scale` (linear final layer, no saturation)

With default scales of 1.0, the decoder was 0.25-Lipschitz while the MLP was 1.0-Lipschitz — a hidden 4x gap. Setting `predictor_lipschitz_scale = 5 * decoder_lipschitz_scale` actually produced a 20x ratio (5.0 vs 0.25).

### Why This Matters for Binary Topology Designs

Sigmoid saturates exponentially away from x=0. For topology optimization designs (pixels near 0 or 1), the effective Lipschitz constant is far below the theoretical 1/4 maximum. The decoder operates on the flat part of sigmoid, making it essentially passive during volume minimization — the volume loss compresses the latent space freely because the decoder can't fight back through vanishing gradients.

### Observable Consequence: MLP-Driven Latent Tails

In thermoelastic2d experiments, even with predictor_lipschitz_scale = 5x decoder, the latent space showed extreme tails:
- Bulk of data crammed near the origin
- Outlier designs with high compliance pushed to z ~ 25 along top active dimensions
- The MLP's Lipschitz constraint was the sole driver of latent separation

The MLP requires minimum latent distance >= |perf_a - perf_b| / L_mlp. With heavy-tailed compliance (100-3000 range after RobustScaling), outliers need large distances that the passive decoder doesn't resist.

## Resolution: Remove Sigmoid, Use Raw Decoder Output

Changed decoder to output `x * lipschitz_scale` (no activation). Now:
- Decoder Lipschitz = `lipschitz_scale` exactly (honest, no hidden factors)
- MLP Lipschitz = `lipschitz_scale` exactly (unchanged)
- Both components on equal footing; volume loss gets honest resistance from both

MSE loss (already in use) naturally penalizes outputs far from [0,1] targets. Clamp to [0,1] at inference only — training gets full gradient signal even when decoder overshoots boundaries.

### Expected Behavior Change

With both components having honest Lipschitz bounds:
- **Correlated geometry + performance** (usual case): decoder and MLP agree on latent separation, jointly organize the latent space
- **Similar geometry, different performance**: MLP forces separation decoder doesn't need — encodes performance-relevant variation
- **Different geometry, similar performance**: decoder forces separation MLP doesn't need — preserves design diversity for LV-DPP

Neither component dominates; both contribute signal. This is the desirable regime for the LV metric suite.

## Open Research Question: Performance Scaling and Lipschitz Bounds

### The Dynamic Range Problem

For Lipschitz-constrained performance prediction, the choice of performance scaler directly affects the effective Lipschitz constraint. With an L-Lipschitz MLP, the minimum latent distance between two samples is:

```
||z_1 - z_2|| >= |p_1 - p_2| / L
```

If performance values have high dynamic range (e.g., compliance 100-3000), the scaler determines how much latent separation outliers require.

### RobustScaler (Current Approach)

RobustScaler shifts by median and divides by IQR — a linear transform. It preserves the shape of the distribution, including heavy tails. A compliance outlier at 3000 with median 300 and IQR 200 becomes (3000-300)/200 = 13.5 in normalized units. The Lipschitz constraint must accommodate this full range.

**Problem**: If L is calibrated to accommodate outliers (large enough that 13.5 / L is manageable), the constraint becomes vacuous for the bulk of the data. Two designs at normalized performance 0.5 and 1.0 only need to be 0.5/L apart — essentially zero if L is large. The latent space learns no performance structure for typical designs.

### Quantile Normalization (Proposed)

Map each objective to N(0,1) via empirical CDF (rank-based). This makes the Lipschitz bound ordinal rather than cardinal:
- Data is uniformly spaced in normalized performance by construction
- Lipschitz bound allocates latent distance proportional to data density, not raw value gaps
- No wasted latent volume on empty performance ranges (e.g., a gap from 500-2000 in raw compliance)
- Works for any distribution shape: heavy tails, multimodal, bounded, signed

**Key insight**: Cardinal structure requires good training data coverage across the performance range. Gaps in coverage create dead zones in latent space where the SN decoder must interpolate without supervision, wasting volume. Ordinal structure avoids this entirely.

**Trade-off**: Quantile normalization discards information about the magnitude of performance gaps. Two designs that are 1 quantile apart could differ by 10 or 1000 in raw performance. For LVAE optimization in latent space, ordinal structure (which direction improves performance) is arguably more useful than cardinal structure (by how much).

### Per-Objective Independence

For multi-objective problems (e.g., thermoelastic: structural compliance, thermal compliance, volume fraction), each objective lives on a different scale. The scaler should normalize each objective column independently. Otherwise the objective with the largest raw range dominates the Lipschitz constraint, and others get ignored.

Both RobustScaler and quantile normalization handle this naturally when applied per-column.

## Implications for the LV Metric Suite

### LV-MMD
The latent distribution of generated samples should match the training distribution. Heavy tails (driven by MLP outliers) destroy this: most mass is near the origin while outliers are at z ~ 25. Quantile normalization + honest decoder bounds should produce more compact, uniform latent distributions.

### LV-DPP
Diversity in latent space should reflect diversity in design space. With sigmoid-saturated decoder, latent spread was driven by performance extremes, not geometric variety. Geometrically similar designs with very different compliance were pushed far apart, inflating DPP without meaningful design diversity. Equal Lipschitz bounds ensure both geometric and performance diversity contribute to latent structure.

### Optimization in Latent Space
With ordinal performance scaling and honest Lipschitz bounds on both components, latent-space optimization should follow smooth performance gradients. No dead zones from cardinal gaps, no vanishing gradients from sigmoid saturation.

## Experimental Questions to Validate

1. **Does removing sigmoid improve reconstruction quality or latent compactness?** Compare LV-MMD and reconstruction NMSE with/without sigmoid on beams2d, heatconduction2d, thermoelastic2d.

2. **Does quantile vs. robust scaling affect latent tail behavior?** Visualize top-2 active dimensions colored by each objective. Expect quantile to produce more uniform latent distributions with fewer extreme outliers.

3. **How do equal decoder/MLP Lipschitz bounds change the active dimension count?** With sigmoid, the decoder was passive — pruning decisions were dominated by performance. With honest bounds, expect different pruning dynamics.

4. **Is there an optimal ratio between decoder and MLP Lipschitz scales?** Equal bounds are the principled default, but the relative importance of geometric vs. performance structure might warrant tuning.

5. **Does the initialization matter for raw decoder?** Without sigmoid, decoder outputs start near 0 rather than 0.5. Monitor early-epoch convergence speed and consider bias initialization if needed.
