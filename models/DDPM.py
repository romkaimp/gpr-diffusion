import torch
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: (B,) integer or float tensor of diffusion steps
        device = x.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = x.unsqueeze(1) * emb.unsqueeze(0)   # (B, half)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:  # pad
            emb = F.pad(emb, (0, 1))
        return emb


def make_mlp(in_dim, hidden_dim, out_dim, dropout=0.0):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class ResidualDilatedBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 residual_channels,
                 pos_emb_dim,
                 t_emb_dim,
                 z_emb_dim,
                 dilation=1,
                 kernel_size=(3, 3),
                 padding=None):
        super().__init__()

        kh, kw = kernel_size
        if padding is None:
            pad_h = (kh - 1) // 2
            pad_w = (kw - 1) // 2 * dilation
            padding = (pad_h, pad_w)

        # 1x1 conv to mix channels of Xn -> gives \tilde{X}_n
        self.conv_in = nn.Conv2d(in_channels, residual_channels, kernel_size=1)

        # linear-ish projections for embeddings: project to residual channel count and broadcast add
        self.proj_n = nn.Linear(pos_emb_dim, residual_channels)
        self.proj_t = nn.Linear(t_emb_dim, residual_channels)
        self.proj_z = nn.Linear(z_emb_dim, residual_channels)

        # small conv pipeline applied after additions
        self.pre_conv = nn.Sequential(
            nn.Conv2d(residual_channels, residual_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(inplace=True)
        )

        # Dilated conv (temporal dilation via width dimension)
        self.dilated = nn.Conv2d(residual_channels, residual_channels * 2, kernel_size=kernel_size,
                                 padding=padding, dilation=(1, dilation))

        # Output projection back to input channels of residual branch
        self.output_proj = nn.Conv2d(residual_channels, in_channels, kernel_size=1)

        # gating will be applied to dilated conv output (split channels)
        self.residual_channels = residual_channels

    def forward(self, x, emb_n, emb_t, emb_z):
        """
        x: (B, in_channels, H, W)
        emb_n: (B, pos_emb_dim)
        emb_t: (B, t_emb_dim)
        emb_z: (B, z_emb_dim)
        """
        # Map input to residual channel space
        x_tilde = self.conv_in(x)  # (B, residual_channels, H, W)

        # project embeddings and unsqueeze to add across spatial dims (H,W)
        # shapes -> (B, residual_channels) -> (B, residual_channels, 1, 1) broadcast add
        pn = self.proj_n(emb_n).unsqueeze(-1).unsqueeze(-1)
        pt = self.proj_t(emb_t).unsqueeze(-1).unsqueeze(-1)
        pz = self.proj_z(emb_z).unsqueeze(-1).unsqueeze(-1)

        h = x_tilde + pn + pt  # add positional and time embeddings
        h = self.pre_conv(h)
        h = h + pz  # add z influence before dilated conv (diagram shows z added later, but commutative)

        # dilated conv output doubled channels for gating
        d = self.dilated(h)  # (B, residual_channels*2, H, W)

        # gated activation: split and apply tanh * sigmoid
        a, b = d.chunk(2, dim=1)   # each (B, residual_channels, H, W)
        gated = torch.tanh(a) * torch.sigmoid(b)

        # project back and residual sum
        res = self.output_proj(gated)  # (B, in_channels, H, W)
        out = x + res

        return out


class DownSampleTime(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Conv2d(
            channel,
            channel,
            kernel_size=(1, 3),   # фильтр только вдоль времени
            stride=(1, 2),        # уменьшаем только W (time)
            padding=(0, 1),       # паддинг только по времени
        )

    def forward(self, x):
        return self.conv(x)


class UpSampleTime(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.conv = nn.Conv2d(
            channel,
            channel,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),   # сохраняем размер
        )

    def forward(self, x):
        # увеличиваем только по времени
        out = F.interpolate(x, scale_factor=(1, 2), mode="nearest")
        out = self.conv(out)
        return out