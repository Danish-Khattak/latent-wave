from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from collators import get_collator
from datasets.pretrain import PretrainCache
from models.jepa import JEPA, update_target_encoder
from models.mamba2_encoder import Mamba2Block, Mamba2Encoder
from pretrain_jepa import _save_checkpoint


def _require_cuda_mamba2():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the official Mamba-2 kernels.")
    try:
        from mamba_ssm import Mamba2  # noqa: F401
    except Exception as exc:
        pytest.skip(f"official mamba_ssm import unavailable: {exc}")


def _small_config():
    return SimpleNamespace(
        encoder=SimpleNamespace(
            architecture="mamba2", img_size=(64, 64), patch_size=16, embed_dim=32,
            depth=2, num_heads=4, mlp_ratio=2.0, drop_path_rate=0.0,
            mamba2=SimpleNamespace(d_state=16, d_conv=4, expand=1, headdim=16, directionality="bidirectional"),
        ),
        predictor=SimpleNamespace(embed_dim=16, depth=1, num_heads=4),
        optimization=SimpleNamespace(
            ema=[0.8, 0.99], epochs=2, start_lr=1e-4, ref_lr=1e-3, final_lr=1e-5,
            warmup=0, ref_weight_decay=0.04, final_weight_decay=0.04, grad_clip=None,
        ),
    )


def _small_jepa(device="cuda"):
    _require_cuda_mamba2()
    model = JEPA(_small_config(), iterations_per_epoch=2, eval_fns=[], metrics=[]).to(device)
    return model


def _masks(batch=2, visible=12, target=4, patches=16, device="cuda"):
    all_indices = torch.arange(patches, device=device)
    enc = all_indices[:visible].repeat(batch, 1)
    pred = all_indices[visible:visible + target].repeat(batch, 1)
    return [enc], [pred]


@pytest.mark.cuda_mamba2
def test_mamba2_encoder_shapes_finite_and_variable_channels():
    _require_cuda_mamba2()
    encoder = Mamba2Encoder(
        (64, 64), 16, 32, 2, 4, d_state=16, d_conv=4, expand=1,
        headdim=16, directionality="bidirectional", mlp_ratio=2.0,
    ).cuda().eval()
    for channels in (1, 3, 5):
        output = encoder(torch.randn(1, channels, 64, 64, device="cuda"))
        assert output.shape == (1, channels * 16, 32)
        assert torch.isfinite(output).all()


@pytest.mark.cuda_mamba2
def test_reverse_path_is_flipped_back_to_original_token_alignment():
    _require_cuda_mamba2()
    block = Mamba2Block(32, d_state=16, d_conv=4, expand=1, headdim=16).cuda().eval()
    x = torch.randn(2, 19, 32, device="cuda")
    normalized = block.norm1(x)
    actual = block._reverse_aligned(normalized)
    expected = block.reverse_mamba(normalized.flip(1).contiguous()).flip(1)
    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, block.reverse_mamba(normalized))
    assert torch.isfinite(actual).all()


@pytest.mark.cuda_mamba2
def test_variable_visible_lengths_ignore_padded_tokens():
    _require_cuda_mamba2()
    block = Mamba2Block(32, d_state=16, d_conv=4, expand=1, headdim=16).cuda().eval()
    lengths = torch.tensor([5, 9, 13], device="cuda")
    x = torch.randn(3, 13, 32, device="cuda")
    changed_padding = x.clone()
    changed_padding[0, 5:] = 1e4
    changed_padding[1, 9:] = -1e4
    first = block(x, lengths)
    second = block(changed_padding, lengths)
    for row, length in enumerate(lengths.tolist()):
        torch.testing.assert_close(first[row, :length], second[row, :length], rtol=0, atol=0)
        assert torch.count_nonzero(first[row, length:]) == 0


@pytest.mark.cuda_mamba2
def test_batch_rows_do_not_share_recurrent_state():
    _require_cuda_mamba2()
    block = Mamba2Block(32, d_state=16, d_conv=4, expand=1, headdim=16).cuda().eval()
    first = torch.randn(1, 17, 32, device="cuda")
    second = torch.randn(1, 17, 32, device="cuda")
    together = block(torch.cat((first, second), dim=0))
    second_alone = block(second)
    torch.testing.assert_close(together[1:], second_alone, rtol=1e-5, atol=1e-6)


@pytest.mark.cuda_mamba2
def test_context_target_state_names_freeze_and_eval_mode():
    model = _small_jepa()
    student = model.context_encoder.state_dict()
    target = model.target_encoder.state_dict()
    assert student.keys() == target.keys()
    assert all(student[name].shape == target[name].shape for name in student)
    assert all(torch.equal(student[name], target[name]) for name in student)
    assert all(not p.requires_grad for p in model.target_encoder.parameters())
    model.train()
    assert model.training
    assert not model.target_encoder.training


def test_ema_blends_float_state_and_copies_integer_buffers():
    class Buffered(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
            self.register_buffer("running", torch.zeros(1))
            self.register_buffer("counter", torch.zeros((), dtype=torch.long))

    context, target = Buffered(), Buffered()
    with torch.no_grad():
        context.weight.fill_(2.0)
        target.weight.fill_(0.0)
        context.running.fill_(4.0)
        target.running.fill_(2.0)
        context.counter.fill_(9)
        target.counter.fill_(1)
    update_target_encoder(context, target, 0.75)
    assert target.weight.item() == pytest.approx(0.5)
    assert target.running.item() == pytest.approx(2.5)
    assert target.counter.item() == 9


@pytest.mark.cuda_mamba2
def test_jepa_loss_gradients_optimizer_then_ema_and_checkpoint_resume(tmp_path):
    model = _small_jepa()
    x = torch.randn(2, 2, 64, 64, device="cuda")
    masks_enc, masks_pred = _masks(device="cuda")
    batch = (x, masks_enc, masks_pred)
    loss, _, _, _ = model.compute_loss_per_patch(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.context_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.predictor.parameters())

    optimizer = torch.optim.AdamW([*model.context_encoder.parameters(), *model.predictor.parameters()], lr=1e-4)
    parameter_name = next(name for name, parameter in model.context_encoder.named_parameters() if parameter.requires_grad)
    context_parameters = dict(model.context_encoder.named_parameters())
    target_parameters = dict(model.target_encoder.named_parameters())
    old_target = target_parameters[parameter_name].detach().clone()
    optimizer.step()
    model.update_target(0.5)
    expected = old_target * 0.5 + context_parameters[parameter_name].detach() * 0.5
    torch.testing.assert_close(target_parameters[parameter_name], expected)
    assert not torch.equal(target_parameters[parameter_name], old_target)
    assert all(p.grad is None for p in model.target_encoder.parameters())

    path = tmp_path / "latest.pt"
    dummy_config = SimpleNamespace(
        model_dump=lambda: {"test": True},
        encoder=SimpleNamespace(architecture="mamba2"),
        seed=0,
    )
    _save_checkpoint(
        path, model, optimizer, [], epoch=2, global_step=7, best_val=0.3,
        sampler_generators={}, config=dummy_config, git_revision={"revision": "test"},
        elapsed_seconds=12.5,
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    restored = _small_jepa().cpu()
    restored.load_state_dict(checkpoint["model"], strict=True)
    assert checkpoint["epoch"] == 2 and checkpoint["global_step"] == 7
    assert checkpoint["best_val_loss"] == pytest.approx(0.3)
    assert checkpoint["architecture"] == "mamba2" and checkpoint["seed"] == 0
    assert checkpoint["git_revision"] == {"revision": "test"}
    assert checkpoint["elapsed_seconds"] == pytest.approx(12.5)


@pytest.mark.cuda_mamba2
def test_masked_target_indices_match_full_target_token_order():
    model = _small_jepa()
    x = torch.randn(2, 2, 64, 64, device="cuda")
    masks_enc, masks_pred = _masks(device="cuda")
    _, actual_target, full_target = model(x, masks_enc, masks_pred)
    normalized = F.layer_norm(full_target, (full_target.shape[-1],))
    batch, _, dim = normalized.shape
    normalized = normalized.reshape(batch, 2, 16, dim)
    expected = normalized[:, :, 12:16].reshape(batch, -1, dim)
    torch.testing.assert_close(actual_target, expected)


@pytest.mark.cuda_mamba2
def test_representative_hdf5_sample_runs_loader_masks_mamba_predictor_loss():
    _require_cuda_mamba2()
    repo = Path(__file__).resolve().parents[1]
    dataset = PretrainCache(repo.parent / "data/pretraining/nrpos-indoor.h5", "normalized_csi")
    batch = [dataset[0], dataset[1]]
    collator = get_collator("multi_block")(
        input_size=(224, 224), patch_size=16, allow_overlap=False,
        aspect_ratio_type="span_time", aspect_ratio=[0.75, 1.5],
        enc_mask_scale=[1.0, 1.0], pred_mask_scale=[0.15, 0.2],
        num_enc_masks=1, num_pred_masks=2, min_keep=10,
    )
    x, masks_enc, masks_pred = collator(batch)
    config_path = repo / "configs/mamba2_smoke.yml"
    from configs.jepa import init_config
    model = JEPA(init_config(str(config_path)), iterations_per_epoch=1, eval_fns=[], metrics=[]).cuda()
    x = x.cuda()
    masks_enc = [mask.cuda() for mask in masks_enc]
    masks_pred = [mask.cuda() for mask in masks_pred]
    predictions, targets, full_target = model(x, masks_enc, masks_pred)
    loss = nn.functional.mse_loss(predictions, targets)
    assert full_target.shape == (2, 5 * 196, 256)
    assert predictions.shape == targets.shape
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.context_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.predictor.parameters())
