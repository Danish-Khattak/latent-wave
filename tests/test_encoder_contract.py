from __future__ import annotations

from types import SimpleNamespace

import torch

from models.encoder import Encoder, build_encoder


def test_vit_baseline_constructs_and_preserves_patch_contract():
    encoder = Encoder(img_size=(224, 224), patch_size=16, embed_dim=32, depth=2, num_heads=4)
    x = torch.randn(2, 3, 224, 224)
    tokens = encoder(x)
    assert tokens.shape == (2, 3 * 14 * 14, 32)
    assert torch.isfinite(tokens).all()


def test_encoder_factory_keeps_vit_selectable():
    config = SimpleNamespace(
        architecture="vit", img_size=(64, 64), patch_size=16,
        embed_dim=32, depth=1, num_heads=4, mlp_ratio=4.0, drop_path_rate=0.0,
    )
    encoder = build_encoder(config)
    assert isinstance(encoder, Encoder)
    assert encoder(torch.randn(1, 2, 64, 64)).shape == (1, 2 * 16, 32)
