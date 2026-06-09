import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from engiopt.bezier_ae.bezier_ae import BezierLayer

def _mlp(dims: list[int], activation: nn.Module) -> nn.Sequential:
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation)
    return nn.Sequential(*layers)


class BezierAutoencoder3D(nn.Module):
    def __init__(
        self,
        n_spans: int = 9,
        n_control_points: int = 32,
        n_data_points: int = 192,
        slice_hidden_dims: list[int] = [512, 256],
        span_hidden_dims: list[int] = [512, 256],
        latent_dim: int = 512,
        w_multiplier: float = 2.0,
        w_min: float = 0.1,
        cpx_bound: list[float] = [0.0, 1.0],
        cpy_bound: list[float] = [-0.75, 0.75],
        x_scaler: list[float] = [0.0, 1.0],
        y_scaler: list[float] = [0.0, 1.0],
    ):
        super().__init__()
        self.n_spans          = n_spans
        self.n_control_points = n_control_points
        self.n_data_points    = n_data_points
        self.latent_dim       = latent_dim
        self.w_multiplier     = w_multiplier
        self.w_min            = w_min
        self.cpx_bound        = cpx_bound
        self.cpy_bound        = cpy_bound

        slice_in = 2 * n_data_points

        # Total free points = n_control_points - 2 (2 TE anchors)
        self.n_free_total = n_control_points - 2
        self.n_free_half  = self.n_free_total // 2

        # ---- encoder -------------------------------------------------------
        self.slice_encoder = _mlp([slice_in] + slice_hidden_dims, nn.GELU())
        span_in = n_spans * slice_hidden_dims[-1]
        self.span_encoder = _mlp([span_in] + span_hidden_dims + [latent_dim], nn.GELU())

        # ---- decoder -------------------------------------------------------
        self.span_decoder = _mlp(
            [latent_dim] + list(reversed(span_hidden_dims)) + [span_in], nn.GELU()
        )

        cp_out = 2 * (2 * self.n_free_half)
        w_out  = 2 * self.n_free_half
        slice_dec_dims = [slice_hidden_dims[-1]] + list(reversed(slice_hidden_dims[:-1]))

        self.slice_cp_decoder = _mlp(slice_dec_dims + [cp_out], nn.GELU())
        self.slice_w_decoder = nn.Sequential(
            *_mlp(slice_dec_dims + [w_out], nn.GELU()),
            nn.Sigmoid(),
        )

        self.bezier_layer = BezierLayer(n_control_points, n_data_points)
        self._slice_hidden_out = slice_hidden_dims[-1]

        scale_x, shift_x = x_scaler[1], x_scaler[0]
        scale_y, shift_y = y_scaler[1], y_scaler[0]
        te = torch.tensor(
            [scale_x * (1.0 - shift_x), scale_y * (0.0 - shift_y)],
            dtype=torch.float32,
        ).view(1, 2, 1)
        self.register_buffer("te_point", te)

        base_intvls = torch.ones(1, n_data_points) / (n_data_points - 1)
        base_intvls[0, 0] = 0.0
        self.register_buffer("base_intvls", base_intvls)

    def _cp_transform(self, cp_ae: Tensor) -> Tensor:
        """
        Processes upper and lower segments to allow control points to extend 
        slightly past the leading edge [0, 1] boundaries, giving the Bezier 
        curve the leverage needed to fully reconstruct round airfoil noses.
        """
        B_S, _, _ = cp_ae.shape
        
        # Unpack separate channels for upper and lower surfaces
        # Shape per variable: [B_S, n_free_half]
        raw_x_up = cp_ae[:, 0, :self.n_free_half]
        raw_y_up = cp_ae[:, 1, :self.n_free_half]
        
        raw_x_lo = cp_ae[:, 0, self.n_free_half:]
        raw_y_lo = cp_ae[:, 1, self.n_free_half:]

        # --- SOLUTION A: BOUNDED OVERLAP OVER THE LEADING EDGE ---
        # Instead of strict 0 to 1 normalization, we allow a slight buffer zone 
        # (e.g., down to -0.05) so control points can hook around the nose.
        le_ext = -0.05 

        # 1. Upper Surface: Moves from Trailing Edge (1.0) down past the Leading Edge (le_ext)
        steps_up = F.softplus(raw_x_up) + 1e-4
        cum_up = torch.cumsum(steps_up, dim=-1)
        # Scale cumulative steps to go from 1.0 down to le_ext smoothly
        x_up = 1.0 - (cum_up / (cum_up[:, -1:] + 1e-6)) * (1.0 - le_ext)

        # 2. Lower Surface: Moves from past the Leading Edge (le_ext) up to Trailing Edge (1.0)
        steps_lo = F.softplus(raw_x_lo) + 1e-4
        cum_lo = torch.cumsum(steps_lo, dim=-1)
        # Scale cumulative steps to go from le_ext up to 1.0 smoothly
        x_lo = le_ext + (cum_lo / (cum_lo[:, -1:] + 1e-6)) * (1.0 - le_ext)

        # Combine X profiles back into a continuous loop segment
        x_scaled = torch.cat([x_up, x_lo], dim=-1)

        # 3. Y Thickness (Bounded via Sigmoid)
        y_all = torch.cat([raw_y_up, raw_y_lo], dim=-1)
        y_bounded = torch.sigmoid(y_all)
        y_scaled = self.cpy_bound[0] + y_bounded * (self.cpy_bound[1] - self.cpy_bound[0])

        return torch.stack([x_scaled, y_scaled], dim=1)
    
    def _add_te_points(self, cp: Tensor) -> Tensor:
        te = self.te_point.expand(cp.shape[0], -1, -1)
        return torch.cat([te, cp, te], dim=2)

    def _render_slice(self, cp_full: Tensor, w_full: Tensor) -> Tensor:
        intvls = self.base_intvls.expand(cp_full.shape[0], -1).to(cp_full.device)
        dp, _, _ = self.bezier_layer(intvls, cp_full, w_full)
        return dp

    def encode(self, x: Tensor) -> Tensor:
        B, S, C, N = x.shape
        x_flat = x.reshape(B * S, C * N)
        h = self.slice_encoder(x_flat)
        h = h.reshape(B, S * h.shape[-1])
        z = self.span_encoder(h)
        return z

    def decode(self, z: Tensor, return_cp: bool = False):
        B = z.shape[0]
        S = self.n_spans

        h = self.span_decoder(z)
        h = h.reshape(B * S, self._slice_hidden_out)

        cp_ae = self.slice_cp_decoder(h)
        w_ae  = self.slice_w_decoder(h)

        cp_ae = cp_ae.view(B * S, 2, self.n_free_total)
        w_ae  = w_ae.view(B * S, 1, self.n_free_total)

        cp = self._cp_transform(cp_ae)
        cp = self._add_te_points(cp)

        w = w_ae * self.w_multiplier
        w = w.clamp(self.w_min, self.w_multiplier)
        w = F.pad(w, (1, 1), value=1.0)

        coords = self._render_slice(cp, w)
        coords = coords.reshape(B, S, 2, self.n_data_points)

        if return_cp:
            cp_out = cp.reshape(B, S, 2, self.n_control_points)
            return coords, cp_out
        return coords

    def forward(self, x: Tensor):
        z      = self.encode(x)
        coords = self.decode(z, return_cp=False)
        return coords, z

    def encode_to_z_ae(self, x: Tensor) -> Tensor: return self.encode(x)
    def decode_from_z_ae(self, z: Tensor) -> Tensor: return self.decode(z)


def loss_reg_fn_3d(coords_pred: Tensor, coords_gt: Tensor, reg_weight: float = 0.001, curvature_weight: float = 0.0001) -> Tensor:
    loss_recon = F.mse_loss(coords_pred, coords_gt)
    diff = coords_pred[:, 1:] - coords_pred[:, :-1]
    loss_smooth = diff.pow(2).mean()
    d2 = coords_pred[:, :, :, 2:] - 2 * coords_pred[:, :, :, 1:-1] + coords_pred[:, :, :, :-2]
    return loss_recon + reg_weight * loss_smooth + curvature_weight * d2.pow(2).mean()
