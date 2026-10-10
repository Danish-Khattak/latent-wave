from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from collators.random_mask import RandomMask
from pretrain_jepa import _epoch_payload, _evaluate, _full_accumulation_steps, _progress_payload


class _ValueDataset(Dataset):
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return torch.full((1, 224, 224), float(self.values[index]))


class _RandomValidationModel:
    def __init__(self):
        self.seen = []

    def eval(self):
        return self

    def compute_loss_per_patch(self, batch):
        images, _, _ = batch
        self.seen.extend(images[:, 0, 0, 0].cpu().tolist())
        return images.mean() + torch.rand((), device=images.device), None, None, None


def test_epoch_payload_reports_matched_run_progress_and_estimates():
    config = SimpleNamespace(
        encoder=SimpleNamespace(architecture="mamba2"),
        optimization=SimpleNamespace(epochs=10),
        seed=2,
    )

    payload = _epoch_payload(
        config=config,
        epoch_index=1,
        global_step=20,
        total_steps=100,
        train_per_source={"a": 1.0, "b": 3.0},
        val_per_source={"a": 0.8, "b": 1.2},
        learning_rate=0.0002,
        ema_momentum=0.997,
        elapsed_seconds=120,
        epoch_elapsed_seconds=60,
        gpu_memory_allocated_bytes=1000,
        gpu_memory_reserved_bytes=2000,
        contributions={"a": 10, "b": 30},
    )

    assert payload["architecture"] == "mamba2"
    assert payload["seed"] == 2
    assert (payload["epoch"], payload["total_epochs"]) == (2, 10)
    assert (payload["step"], payload["total_steps"]) == (20, 100)
    assert payload["train_loss_mean"] == pytest.approx(2.0)
    assert payload["val_loss_mean"] == pytest.approx(1.0)
    assert payload["eta_seconds"] == pytest.approx(480.0)
    assert payload["source_examples_per_second"] == pytest.approx(40 / 60)
    assert payload["gpu_memory_allocated_bytes"] == 1000
    assert payload["gpu_memory_reserved_bytes"] == 2000


def test_training_microbatches_align_to_full_accumulation_updates():
    assert _full_accumulation_steps(1716, 8) == 1712
    assert _full_accumulation_steps(24, 8) == 24
    with pytest.raises(ValueError, match="too few full batches"):
        _full_accumulation_steps(7, 8)


def test_in_epoch_progress_reports_rolling_loss_validation_and_eta():
    config = SimpleNamespace(
        encoder=SimpleNamespace(architecture="vit"),
        optimization=SimpleNamespace(epochs=10),
        seed=1,
    )
    payload = _progress_payload(
        config, 2, 0.5, 25, 100, {"a": 0.5, "b": 0.7}, 0.9,
        0.0001, 0.998, 250, 3000, 4000, {"a": 80, "b": 80},
    )
    assert payload["architecture"] == "vit"
    assert payload["epoch"] == 3 and payload["epoch_fraction"] == 0.5
    assert payload["train_loss_mean"] == pytest.approx(0.6)
    assert payload["previous_validation_loss"] == pytest.approx(0.9)
    assert payload["eta_seconds"] == pytest.approx(750)
    assert payload["gpu_memory_allocated_bytes"] == 3000


def test_validation_is_repeatable_and_does_not_cycle_shorter_sources():
    large_collator = RandomMask(ratios=[0.5, 0.5], input_size=(224, 224), patch_size=16)
    small_collator = RandomMask(ratios=[0.5, 0.5], input_size=(224, 224), patch_size=16)
    loaders = {
        "large": DataLoader(_ValueDataset([1, 2, 3, 4, 5]), batch_size=2, collate_fn=large_collator),
        "small": DataLoader(_ValueDataset([10, 20]), batch_size=2, collate_fn=small_collator),
    }
    config = SimpleNamespace(seed=7, optimization=SimpleNamespace(precision="32-true"))
    model = _RandomValidationModel()
    torch.manual_seed(902)
    rng_before = torch.get_rng_state().clone()

    first = _evaluate(model, loaders, torch.device("cpu"), config)
    first_seen = list(model.seen)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert first_seen == [1, 2, 3, 4, 5, 10, 20]
    assert large_collator._itr_counter.value == 2
    assert small_collator._itr_counter.value == 0

    model.seen.clear()
    second = _evaluate(model, loaders, torch.device("cpu"), config)
    assert second == first
    assert model.seen == first_seen
