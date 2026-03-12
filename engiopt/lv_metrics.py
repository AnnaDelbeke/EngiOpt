"""Latent-space metrics for evaluating generative models using a trained LVAE.

Provides per-sample metrics computed in the performance-constrained LVAE
latent space: projection residual and dual-LVAE projection gap.
Distribution-level metrics (LV-MMD, LV-DPP) are computed directly via
``engiopt.metrics`` in the evaluation scripts.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from engiopt.vanilla_lvae.utils import decode_designs
from engiopt.vanilla_lvae.utils import encode_designs

if TYPE_CHECKING:
    from torch import nn


# ---------------------------------------------------------------------------
# Encode-decode helper
# ---------------------------------------------------------------------------


def _encode_decode(
    encoder: nn.Module,
    decoder: nn.Module,
    designs: npt.NDArray,
    device: str | object,
    batch_size: int = 256,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Encode then decode, returning latent codes and reconstructions.

    Args:
        encoder: Trained LVAE encoder.
        decoder: Trained LVAE decoder.
        designs: Designs of shape (N, H, W) or (N, 1, H, W).
        device: Torch device.
        batch_size: Batch size for processing.

    Returns:
        Tuple of (latent_codes (N, D), reconstructed_designs (N, H, W)).
    """
    z = encode_designs(encoder, designs, device, batch_size=batch_size)
    recon = decode_designs(decoder, z, device, batch_size=batch_size)
    return z, recon


# ---------------------------------------------------------------------------
# Per-sample metrics
# ---------------------------------------------------------------------------


def lv_project_designs(
    encoder: nn.Module,
    decoder: nn.Module,
    designs: npt.NDArray,
    device: str | object,
    batch_size: int = 256,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Project designs onto the learned manifold and measure the distance.

    Per-sample MSE between input designs and their LVAE projection
    (encode→decode).  Higher residual means the generator produced designs
    that required more correction by the LVAE.

    Args:
        encoder: Trained LVAE encoder.
        decoder: Trained LVAE decoder.
        designs: Designs of shape (N, H, W) or (N, 1, H, W).
        device: Torch device.
        batch_size: Batch size for processing.

    Returns:
        Tuple of (per-sample MSE (N,), projected designs (N, H, W)).
    """
    _, recon = _encode_decode(encoder, decoder, designs, device, batch_size)
    orig = designs.squeeze(1) if designs.ndim == 4 else designs  # noqa: PLR2004
    mse = np.mean((orig - recon) ** 2, axis=(1, 2))
    return mse, recon


def lv_dual_projection_gap(
    encoder_perf: nn.Module,
    decoder_perf: nn.Module,
    encoder_recon: nn.Module,
    decoder_recon: nn.Module,
    designs: npt.NDArray,
    device: str | object,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Compare projections from perf-constrained vs recon-only LVAEs.

    The gap between the two projections isolates the *performance-attributable*
    component of the correction.  Large gap → the deviation from the manifold
    was in a performance-relevant direction.  Small gap → cosmetic noise only.

    Args:
        encoder_perf: Perf-constrained LVAE encoder.
        decoder_perf: Perf-constrained LVAE decoder.
        encoder_recon: Recon-only LVAE encoder.
        decoder_recon: Recon-only LVAE decoder.
        designs: Designs of shape (N, H, W) or (N, 1, H, W).
        device: Torch device.
        batch_size: Batch size for processing.

    Returns:
        Dictionary with per-sample arrays and scalar summaries:
        - ``dual_gap``: per-sample MSE between the two projections (N,).
        - ``residual_perf``: per-sample MSE, original vs perf projection (N,).
        - ``residual_recon``: per-sample MSE, original vs recon projection (N,).
        - ``dual_gap_mean``, ``residual_perf_mean``, ``residual_recon_mean``.
    """
    _, proj_perf = _encode_decode(encoder_perf, decoder_perf, designs, device, batch_size)
    _, proj_recon = _encode_decode(encoder_recon, decoder_recon, designs, device, batch_size)

    orig = designs.squeeze(1) if designs.ndim == 4 else designs  # noqa: PLR2004

    dual_gap = np.mean((proj_perf - proj_recon) ** 2, axis=(1, 2))
    residual_perf = np.mean((orig - proj_perf) ** 2, axis=(1, 2))
    residual_recon = np.mean((orig - proj_recon) ** 2, axis=(1, 2))

    return {
        "dual_gap": dual_gap,
        "residual_perf": residual_perf,
        "residual_recon": residual_recon,
        "dual_gap_mean": float(dual_gap.mean()),
        "residual_perf_mean": float(residual_perf.mean()),
        "residual_recon_mean": float(residual_recon.mean()),
    }
