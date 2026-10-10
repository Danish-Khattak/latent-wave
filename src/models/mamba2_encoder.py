from __future__ import annotations

from functools import partial
import math

import torch
from torch import nn
from torch.nn.init import trunc_normal_

from .components import DropPath, MLP, PatchEmbed
from .utils import apply_masks, get_2d_sincos_pos_embed


def _official_mamba2():
    try:
        from mamba_ssm import Mamba2
    except Exception as exc:  # preserve the original loader/kernel failure
        raise RuntimeError(
            "The Mamba-2 encoder requires the locked official mamba-ssm source, "
            "causal-conv1d, and their CUDA/Triton kernels. Install the project "
            "environment with `uv sync --no-build-isolation-package mamba-ssm`."
        ) from exc
    return Mamba2


class Mamba2Block(nn.Module):
    """Pre-norm bidirectional official Mamba-2 mixer plus the ViT MLP branch."""

    def __init__(
        self,
        dim: int,
        *,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 1,
        headdim: int = 64,
        directionality: str = "bidirectional",
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        if directionality not in {"bidirectional", "forward"}:
            raise ValueError("directionality must be 'bidirectional' or 'forward'.")
        mamba2 = _official_mamba2()
        self.directionality = directionality
        self.norm1 = norm_layer(dim)
        kwargs = {
            "d_model": dim,
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
            "headdim": headdim,
        }
        self.forward_mamba = mamba2(**kwargs)
        self.reverse_mamba = mamba2(**kwargs) if directionality == "bidirectional" else None
        self.direction_proj = nn.Linear(dim * (2 if self.reverse_mamba is not None else 1), dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, hidden_features=int(dim * mlp_ratio))

        trunc_normal_(self.direction_proj.weight, std=0.02)
        nn.init.zeros_(self.direction_proj.bias)
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _run_valid_sequences(module: nn.Module, x: torch.Tensor, lengths: torch.Tensor, reverse: bool) -> torch.Tensor:
        """Run each distinct valid length without exposing padding to the SSM."""
        batch, padded_length, dim = x.shape
        lengths = lengths.to(device=x.device, dtype=torch.long)
        if lengths.shape != (batch,) or torch.any(lengths < 1) or torch.any(lengths > padded_length):
            raise ValueError("lengths must contain one valid length in [1, padded_length] per row.")

        outputs: list[torch.Tensor | None] = [None] * batch
        for length in torch.unique(lengths, sorted=True).tolist():
            indices = torch.nonzero(lengths == length, as_tuple=False).flatten()
            valid = x.index_select(0, indices)[:, :length].contiguous()
            if reverse:
                valid = valid.flip(1).contiguous()
            encoded = module(valid)
            if reverse:
                encoded = encoded.flip(1)
            for local, row in enumerate(indices.tolist()):
                outputs[row] = encoded[local]

        rows = []
        for row, length in enumerate(lengths.tolist()):
            encoded = outputs[row]
            assert encoded is not None
            if length < padded_length:
                encoded = torch.nn.functional.pad(encoded, (0, 0, 0, padded_length - length))
            rows.append(encoded)
        return torch.stack(rows, dim=0).reshape(batch, padded_length, dim)

    def _forward_equal_length(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        forward = self.forward_mamba(normalized)
        if self.reverse_mamba is None:
            mixed = forward
        else:
            reverse = self._reverse_aligned(normalized)
            mixed = torch.cat((forward, reverse), dim=-1)
        x = x + self.drop_path(self.direction_proj(mixed))
        return x + self.drop_path(self.mlp(self.norm2(x)))

    def _reverse_aligned(self, normalized: torch.Tensor) -> torch.Tensor:
        if self.reverse_mamba is None:
            raise RuntimeError("This block has no reverse Mamba-2 path.")
        return self.reverse_mamba(normalized.flip(1).contiguous()).flip(1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected (batch, sequence, dim), got {tuple(x.shape)}")
        if lengths is None:
            return self._forward_equal_length(x)

        valid = lengths.to(device=x.device, dtype=torch.long)
        normalized = self.norm1(x)
        forward = self._run_valid_sequences(self.forward_mamba, normalized, valid, reverse=False)
        if self.reverse_mamba is None:
            mixed = forward
        else:
            reverse = self._run_valid_sequences(self.reverse_mamba, normalized, valid, reverse=True)
            mixed = torch.cat((forward, reverse), dim=-1)
        valid_mask = torch.arange(x.shape[1], device=x.device)[None, :] < valid[:, None]
        mask = valid_mask.unsqueeze(-1).to(dtype=x.dtype)
        x = (x + self.drop_path(self.direction_proj(mixed))) * mask
        x = (x + self.drop_path(self.mlp(self.norm2(x)))) * mask
        return x


class Mamba2Encoder(nn.Module):
    """Patch encoder with the same `(B,C,H,W)` input and `(B,C*P,D)` output as ViT."""

    def __init__(
        self,
        img_size: tuple[int, int] | list[int],
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int = 8,
        *,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 1,
        headdim: int = 64,
        directionality: str = "bidirectional",
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        del num_heads  # retained in the factory signature so ViT/Mamba configs align
        self.patch_embed = PatchEmbed(img_size=img_size[0], patch_size=patch_size, embed_dim=embed_dim)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        num_patches = (self.patch_embed.img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, num_patches, embed_dim), requires_grad=False)
        pos = get_2d_sincos_pos_embed(embed_dim, int(num_patches**0.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            Mamba2Block(
                embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
                directionality=directionality,
                mlp_ratio=mlp_ratio,
                drop_path=rates[index],
                norm_layer=norm_layer,
            )
            for index in range(depth)
        )
        self.norm = norm_layer(embed_dim)

    def forward(self, x: torch.Tensor, masks=None) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input (B,C,H,W), got {tuple(x.shape)}")
        if x.shape[-2] % self.patch_size or x.shape[-1] % self.patch_size:
            raise ValueError("Input height and width must be divisible by patch_size.")
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        tokens = self.patch_embed(x)
        grid_size = math.isqrt(tokens.shape[2])
        if grid_size * grid_size != tokens.shape[2]:
            raise ValueError(f"Expected a square patch grid, got {tokens.shape[2]} tokens per channel.")
        if tokens.shape[2] != self.pos_embed.shape[2]:
            pos = torch.tensor(
                get_2d_sincos_pos_embed(self.embed_dim, grid_size, cls_token=False),
                device=tokens.device,
                dtype=tokens.dtype,
            ).unsqueeze(0).unsqueeze(0)
        else:
            pos = self.pos_embed.to(device=tokens.device, dtype=tokens.dtype)
        tokens = tokens + pos
        if masks is not None:
            tokens = apply_masks(tokens, masks)
        tokens = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)
