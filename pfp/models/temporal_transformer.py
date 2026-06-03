from __future__ import annotations

import torch
import torch.nn as nn


class TemporalTransformerBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        output_dim: int | None = None,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        max_horizon: int = 64,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.global_cond_dim = int(global_cond_dim)
        self.output_dim = int(self.input_dim if output_dim is None else output_dim)
        self.d_model = int(d_model)
        self.max_horizon = int(max_horizon)
        self.use_layer_norm = bool(use_layer_norm)

        self.sample_proj = nn.Linear(self.input_dim, self.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_horizon, self.d_model))

        if self.global_cond_dim > 0:
            self.cond_proj = nn.Linear(self.global_cond_dim, self.d_model)
        else:
            self.cond_proj = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(num_heads),
            dim_feedforward=int(self.d_model * float(mlp_ratio)),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))
        self.final_norm = nn.LayerNorm(self.d_model) if self.use_layer_norm else nn.Identity()
        self.out_proj = nn.Linear(self.d_model, self.output_dim)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del timestep  # kept for interface compatibility with ConditionalUnet1D usage.
        assert sample.ndim == 3, f"sample must be [B,T,C], got shape={tuple(sample.shape)}"
        b, t, _ = sample.shape
        assert t <= self.max_horizon, f"T={t} exceeds max_horizon={self.max_horizon}"

        x = self.sample_proj(sample)
        x = x + self.pos_emb[:, :t, :]

        if self.global_cond_dim > 0:
            assert global_cond is not None, "global_cond is required when global_cond_dim > 0"
            assert global_cond.shape == (b, self.global_cond_dim), (
                f"global_cond must be [B,{self.global_cond_dim}], got {tuple(global_cond.shape)}"
            )
            x = x + self.cond_proj(global_cond)[:, None, :]
        elif global_cond is not None and self.cond_proj is not None:
            x = x + self.cond_proj(global_cond)[:, None, :]

        x = self.transformer(x)
        x = self.final_norm(x)
        out = self.out_proj(x)
        assert out.shape == (b, t, self.output_dim), (
            f"output must be [B,T,{self.output_dim}], got {tuple(out.shape)}"
        )
        return out
