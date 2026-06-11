"""
DDM_W3D: thin subclass of DDM_W for the 3D BAE + LVAE3D pipeline.

The only difference from DDM_W is in loss() and generate():
  - LVAE decoder now returns (z_bae_pred, aoa_pred, eta_y_pred, pressure_pred, None)
    where z_bae_pred is [B, bae_latent_dim] (joint wing latent, not [B,S,3,30])
  - BAE decode is bae_model.decode(z_bae) → [B, S, 2, 192] (all slices at once)

Everything else (denoiser, diffusion schedule, save/load) is inherited unchanged.
"""

import torch
import torch.nn as nn

from engiopt.ddm.ddm_w.ddm_w import DDM_W, MLPDenoiser


class DDM_W3D(DDM_W):
    """DDM_W adapted for the 3D BAE + LVAE3D pipeline."""

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, batch, return_components=False):
        """Same as DDM_W.loss but uses the 3D LVAE decoder interface."""
        w_opt, aoa, params, w_init, pressure_gt = batch
        device = w_opt.device
        B = w_opt.shape[0]

        t = torch.randint(0, self.sampler.T, (B,), device=device)

        w_noisy,   w_noise   = self._forward_diffusion(w_opt, t)
        aoa_noisy, aoa_noise = self._forward_diffusion(aoa,   t)

        w_noise_pred, aoa_noise_pred = self.denoiser(
            w_noisy, aoa_noisy, params, w_init, t
        )

        loss_w   = nn.functional.mse_loss(w_noise_pred,   w_noise)
        loss_aoa = nn.functional.mse_loss(aoa_noise_pred, aoa_noise)

        loss_p = torch.tensor(0.0, device=device)
        if self.w_pressure > 0 and pressure_gt is not None:
            schedule = self.sampler.schedule_x
            sqrt_ab  = self._get_index(schedule.sqrt_alphas_cumprod,           t, w_noisy.shape)
            sqrt_1ab = self._get_index(schedule.sqrt_one_minus_alphas_cumprod, t, w_noisy.shape)
            w0_est = (w_noisy - sqrt_1ab * w_noise_pred) / sqrt_ab.clamp(min=1e-8)

            if self.w_mean is not None:
                w0_raw = w0_est * self.w_std.to(device) + self.w_mean.to(device)
            else:
                w0_raw = w0_est

            w0_masked = self.lvae_model._apply_mask(w0_raw)
            lvae_params = self._lvae_params(params)
            self.lvae_model.encoder.to(device)
            self.lvae_model.decoder.to(device)

            with torch.no_grad():
                # LVAE3D decoder: returns (z_bae_pred, aoa, eta_y, pressure, None)
                _, _, _, pressure_pred, _ = self.lvae_model.decoder(w0_masked, lvae_params)

            lvae_ps = getattr(self.lvae_model, 'scaler_pressures', None)
            if lvae_ps is not None:
                p_mean = torch.tensor(float(lvae_ps.mean), device=device)
                p_std  = torch.tensor(float(lvae_ps.std),  device=device)
                p_raw  = pressure_pred * p_std + p_mean
            else:
                p_raw = pressure_pred

            if hasattr(self, 'p_ddm_mean'):
                p_norm = (p_raw - self.p_ddm_mean) / max(self.p_ddm_std, 1e-8)
            else:
                p_norm = p_raw

            loss_p = nn.functional.mse_loss(p_norm, pressure_gt)

        total = loss_w + self.w_aoa * loss_aoa + self.w_pressure * loss_p

        if return_components:
            return total, loss_w, loss_aoa, loss_p
        return total

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, w_init: torch.Tensor, params: torch.Tensor,
                 device: str, T: int = None):
        """
        Denoise in w-space, decode through LVAE3D + 3D BAE.

        Returns
        -------
        coords    : [B, S, 2, 192]
        aoas      : [B]
        pressures : [B, S, 192]
        te_shifts : [B, S]
        w_raw     : [B, w_dim]
        """
        T = T or self.sampler.T
        B = w_init.shape[0]

        w_noisy   = torch.randn(B, self.w_dim, device=device)
        aoa_noisy = torch.randn(B, 1,          device=device)

        schedule = self.sampler.schedule_x

        for i in reversed(range(T)):
            t = torch.full((B,), i, device=device, dtype=torch.long)
            w_noise_pred, aoa_noise_pred = self.denoiser(
                w_noisy, aoa_noisy, params, w_init, t
            )

            alpha_t      = schedule.alphas[i]
            alpha_bar_t  = schedule.alphas_cumprod[i]
            alpha_bar_t1 = schedule.alphas_cumprod[i - 1] if i > 0 else torch.tensor(1.0)
            beta_t       = 1.0 - alpha_t

            w0_est = (w_noisy - (1 - alpha_bar_t).sqrt() * w_noise_pred) / alpha_bar_t.sqrt().clamp(min=1e-8)
            w0_est = w0_est.clamp(-5, 5)

            if i > 0:
                posterior_var  = beta_t * (1 - alpha_bar_t1) / (1 - alpha_bar_t).clamp(min=1e-8)
                posterior_mean = (
                    alpha_bar_t1.sqrt() * beta_t / (1 - alpha_bar_t).clamp(min=1e-8) * w0_est
                    + alpha_t.sqrt() * (1 - alpha_bar_t1) / (1 - alpha_bar_t).clamp(min=1e-8) * w_noisy
                )
                w_noisy = posterior_mean + posterior_var.sqrt() * torch.randn_like(w_noisy)

                aoa0_est  = (aoa_noisy - (1 - alpha_bar_t).sqrt() * aoa_noise_pred) / alpha_bar_t.sqrt().clamp(min=1e-8)
                aoa0_est  = aoa0_est.clamp(-5, 5)
                aoa_mean  = (
                    alpha_bar_t1.sqrt() * beta_t / (1 - alpha_bar_t).clamp(min=1e-8) * aoa0_est
                    + alpha_t.sqrt() * (1 - alpha_bar_t1) / (1 - alpha_bar_t).clamp(min=1e-8) * aoa_noisy
                )
                aoa_noisy = aoa_mean + posterior_var.sqrt() * torch.randn_like(aoa_noisy)
            else:
                w_noisy   = w0_est
                aoa_noisy = (aoa_noisy - (1 - alpha_bar_t).sqrt() * aoa_noise_pred) / alpha_bar_t.sqrt().clamp(min=1e-8)
                aoa_noisy = aoa_noisy.clamp(-5, 5)

        w_gen   = w_noisy
        aoa_gen = aoa_noisy

        if self.w_mean is not None:
            w_raw = w_gen * self.w_std.to(device) + self.w_mean.to(device)
        else:
            w_raw = w_gen

        w_masked = self.lvae_model._apply_mask(w_raw)
        lvae_params = self._lvae_params(params)
        self.lvae_model.decoder.to(device)

        # LVAE3D decoder: (z_bae_pred, aoa_pred, eta_y_pred, pressure_pred, None)
        z_bae_pred, aoa_pred_from_w, eta_y_pred, pressure_pred, _ = self.lvae_model.decoder(w_masked, lvae_params)
        # z_bae_pred : [B, bae_latent_dim]
        # eta_y_pred : [B, S, 1]
        # pressure_pred : [B, S, 192]

        # Decode 3D BAE: [B, bae_latent_dim] → [B, S, 2, 192]
        self.bae_model.to(device)
        coords = self.bae_model.decode(z_bae_pred)   # [B, S, 2, 192]

        # Denormalise AoA — use decoder's reconstruction (consistent with joint latent w)
        if self.scaler_aoas is not None:
            aoa_np = self.scaler_aoas.inverse_transform(aoa_pred_from_w.cpu().numpy())
            aoas   = torch.tensor(aoa_np, dtype=torch.float32).squeeze(1)
        else:
            aoas = aoa_pred_from_w.squeeze(1)

        # Denormalise pressure
        lvae_ps = getattr(self.lvae_model, 'scaler_pressures', None)
        if lvae_ps is not None:
            p_mean = torch.tensor(float(lvae_ps.mean), device=device)
            p_std  = torch.tensor(float(lvae_ps.std),  device=device)
            pressures = pressure_pred * p_std + p_mean
        else:
            pressures = pressure_pred

        te_shifts = eta_y_pred.squeeze(-1)   # [B, S]

        return coords.cpu(), aoas.cpu(), pressures.cpu(), te_shifts.cpu(), w_raw.cpu(), z_bae_pred.cpu()
