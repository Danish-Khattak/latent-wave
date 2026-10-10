from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import random
import signal
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler, Subset

from collators import get_collator
from configs.jepa import init_config
from datasets.pretrain import PretrainCache
from models.jepa import JEPA
from utils import count_parameters, set_seed


_STOP_AFTER_EPOCH = False
_VALIDATION_PROTOCOL = "fixed_seed_masks_channels_each_example_once_v1"


def _handle_stop_signal(signum, _frame):
    global _STOP_AFTER_EPOCH
    _STOP_AFTER_EPOCH = True
    print(f"[signal] {signal.Signals(signum).name}; will checkpoint after this epoch", flush=True)


def _project_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (Path(__file__).resolve().parents[1] / value).resolve()


def _model_metadata(model: JEPA, device: torch.device) -> dict:
    total, trainable = count_parameters(model)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "student_encoder_parameters": sum(p.numel() for p in model.context_encoder.parameters()),
        "target_encoder_parameters": sum(p.numel() for p in model.target_encoder.parameters()),
        "predictor_parameters": sum(p.numel() for p in model.predictor.parameters()),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_capability": list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None,
    }


def _versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    result = {"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda}
    for name in ("mamba-ssm", "causal-conv1d", "lightning", "h5py", "numpy", "pydantic-settings", "scikit-learn"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def _git_revision() -> dict:
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {"root": str(root), "revision": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"), "dirty": bool(git("status", "--porcelain"))}


def _build_loaders(config, mask_config, smoke: bool = False, smoke_batch_size: int = 1):
    if smoke_batch_size < 1:
        raise ValueError("smoke_batch_size must be at least 1.")
    source_loaders = {}
    val_loaders = {}
    source_paths = {}
    for index, ds_config in enumerate(config.data.pretrain_datasets):
        path = _project_path(ds_config.path)
        if not path.is_file():
            raise FileNotFoundError(f"Pretraining HDF5 file not found: {path}")
        with h5py.File(path, "r") as h5:
            if ds_config.key not in h5:
                raise KeyError(f"Missing HDF5 key {ds_config.key!r} in {path}")
            shape = h5[ds_config.key].shape
            if len(shape) != 4:
                raise ValueError(f"Expected (N,C,H,W) in {path}:{ds_config.key}, got {shape}")
            if not np.isfinite(h5[ds_config.key][0]).all():
                raise ValueError(f"Non-finite input found in representative sample {path}:{ds_config.key}[0]")

        dataset = PretrainCache(path, ds_config.key)
        if len(dataset) < 2:
            raise ValueError(f"Need at least two samples to make an unlabeled train/validation split: {path}")
        if smoke and index > 0:
            continue
        val_count = max(1, int(len(dataset) * config.data.validation_fraction))
        train_count = len(dataset) - val_count
        split_generator = torch.Generator().manual_seed(config.seed + index + 1)
        permutation = torch.randperm(len(dataset), generator=split_generator).tolist()
        train_dataset = Subset(dataset, permutation[:train_count])
        val_dataset = Subset(dataset, permutation[train_count:])

        generator = torch.Generator().manual_seed(config.seed + 1000 + index)
        sampler = RandomSampler(train_dataset, generator=generator)
        collator = get_collator(config.masks.type)(
            input_size=config.encoder.img_size,
            patch_size=config.encoder.patch_size,
            **getattr(config.masks, config.masks.type).model_dump(),
        )
        val_collator = get_collator(config.masks.type)(
            input_size=config.encoder.img_size,
            patch_size=config.encoder.patch_size,
            **getattr(config.masks, config.masks.type).model_dump(),
        )
        source_loaders[ds_config.name] = DataLoader(
            train_dataset,
            batch_size=smoke_batch_size if smoke else config.data.batch_size,
            sampler=sampler,
            num_workers=0 if smoke else config.data.num_workers,
            pin_memory=bool(config.data.pin_mem),
            collate_fn=collator,
            persistent_workers=bool(config.data.num_workers and not smoke),
            drop_last=not smoke,
        )
        val_loaders[ds_config.name] = DataLoader(
            val_dataset,
            batch_size=smoke_batch_size if smoke else config.data.batch_size,
            shuffle=False,
            # Validation masks and channel subsets are seeded by _evaluate below.
            # Keeping collators in the main process makes that seed reproducible.
            num_workers=0,
            pin_memory=bool(config.data.pin_mem),
            collate_fn=val_collator,
            persistent_workers=False,
        )
        source_paths[ds_config.name] = {"path": str(path), "key": ds_config.key, "samples": len(dataset), "shape": list(shape)}
    if not source_loaders:
        raise ValueError("No pretraining datasets were configured.")
    return source_loaders, val_loaders, source_paths


def _next_max_cycle_batches(loaders: dict, iterators: dict, step: int) -> dict:
    batches = {}
    for name, loader in loaders.items():
        try:
            batches[name] = next(iterators[name])
        except StopIteration:
            iterators[name] = iter(loader)
            batches[name] = next(iterators[name])
    return batches


def _move_batch(batch, device):
    x, masks_enc, masks_pred = batch
    x = x.to(device, non_blocking=True)
    masks_enc = [mask.to(device, non_blocking=True) for mask in masks_enc]
    masks_pred = [mask.to(device, non_blocking=True) for mask in masks_pred]
    return x, masks_enc, masks_pred


def _autocast(config, device):
    precision = config.optimization.precision
    if precision == "32-true":
        return torch.autocast(device_type=device.type, enabled=False)
    if precision == "bf16-mixed":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16-mixed requires a CUDA device that reports bfloat16 support.")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(f"Unsupported precision setting: {precision}")


@torch.no_grad()
def _evaluate(model, loaders, device, config, limit_steps: int | None = None):
    model.eval()
    if any(loader.num_workers != 0 for loader in loaders.values()):
        raise RuntimeError("Deterministic validation requires validation DataLoaders with num_workers=0.")

    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    results = {}
    # Keep validation randomness isolated from the training RNG stream. Reset the
    # collator counter and both CPU/CUDA RNGs for every source on each evaluation,
    # so the same unlabeled examples see the same masks and sampled channels.
    with torch.random.fork_rng(devices=cuda_devices):
        for source_index, (name, loader) in enumerate(loaders.items()):
            collator_counter = getattr(loader.collate_fn, "_itr_counter", None)
            if collator_counter is not None:
                with collator_counter.get_lock():
                    collator_counter.value = -1

            validation_seed = int(config.seed) + 10_000 + source_index
            torch.manual_seed(validation_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(validation_seed)

            total_loss = 0.0
            total_examples = 0
            for step, batch in enumerate(loader):
                if limit_steps is not None and step >= limit_steps:
                    break
                with _autocast(config, device):
                    loss, _, _, _ = model.compute_loss_per_patch(_move_batch(batch, device))
                batch_size = int(batch[0].shape[0])
                total_loss += float(loss) * batch_size
                total_examples += batch_size

            if total_examples == 0:
                raise ValueError(f"Validation source {name!r} yielded no examples.")
            results[name] = total_loss / total_examples
    return results


def _save_checkpoint(
    path: Path,
    model,
    optimizer,
    schedulers,
    epoch,
    global_step,
    best_val,
    sampler_generators,
    config,
    git_revision,
    elapsed_seconds=0.0,
    best_epoch=None,
    last_val_loss=None,
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "schedulers": [scheduler.state_dict() for scheduler in schedulers],
        "architecture": config.encoder.architecture,
        "seed": config.seed,
        "git_revision": git_revision,
        "epoch": epoch,
        "global_step": global_step,
        "elapsed_seconds": elapsed_seconds,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "last_val_loss": last_val_loss,
        "validation_protocol": _VALIDATION_PROTOCOL,
        "sampler_states": {name: generator.get_state() for name, generator in sampler_generators.items()},
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "resolved_config": config.model_dump(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _write_run_files(output_dir: Path, config, source_paths, model, device):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yml").write_text(yaml.safe_dump(config.model_dump(), sort_keys=False), encoding="utf-8")
    metadata = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": _git_revision(),
        "versions": _versions(),
        "data": source_paths,
        "architecture": config.encoder.architecture,
        "seed": config.seed,
        "output_dir": str(output_dir),
        "status": "preflight_complete",
        "completed_epochs": 0,
        "global_step": 0,
        "best_checkpoint_epoch": None,
        "validation_protocol": _VALIDATION_PROTOCOL,
        "model": _model_metadata(model, device),
        "hardware": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "nvidia_smi": subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True, capture_output=True).stdout.strip() or None,
        },
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _write_run_metadata(output_dir: Path, metadata: dict):
    temporary = output_dir / "run_metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary.replace(output_dir / "run_metadata.json")


def _epoch_payload(
    config,
    epoch_index,
    global_step,
    total_steps,
    train_per_source,
    val_per_source,
    learning_rate,
    ema_momentum,
    elapsed_seconds,
    epoch_elapsed_seconds,
    gpu_memory_allocated_bytes,
    gpu_memory_reserved_bytes,
    contributions,
):
    completed_epochs = epoch_index + 1
    remaining_epochs = max(0, config.optimization.epochs - completed_epochs)
    mean_epoch_seconds = elapsed_seconds / completed_epochs
    return {
        "architecture": config.encoder.architecture,
        "seed": config.seed,
        "epoch": completed_epochs,
        "total_epochs": config.optimization.epochs,
        "step": global_step,
        "total_steps": total_steps,
        "train_loss": train_per_source,
        "train_loss_mean": float(np.mean(list(train_per_source.values()))),
        "val_loss": val_per_source,
        "val_loss_mean": float(np.mean(list(val_per_source.values()))),
        "validation_protocol": _VALIDATION_PROTOCOL,
        "learning_rate": learning_rate,
        "ema_momentum": ema_momentum,
        "elapsed_seconds": elapsed_seconds,
        "epoch_elapsed_seconds": epoch_elapsed_seconds,
        "eta_seconds": mean_epoch_seconds * remaining_epochs,
        "source_examples_per_second": sum(contributions.values()) / max(epoch_elapsed_seconds, 1e-9),
        "gpu_memory_allocated_bytes": gpu_memory_allocated_bytes,
        "gpu_memory_reserved_bytes": gpu_memory_reserved_bytes,
        "dataset_contribution_examples": contributions,
    }


def _full_accumulation_steps(max_steps: int, grad_accumulation: int) -> int:
    if grad_accumulation < 1:
        raise ValueError("grad_accumulation must be at least one.")
    aligned_steps = max_steps - max_steps % grad_accumulation
    if aligned_steps < grad_accumulation:
        raise ValueError("The largest source has too few full batches for one accumulated update.")
    return aligned_steps


def _progress_payload(
    config,
    epoch_index,
    progress_fraction,
    global_step,
    total_steps,
    train_per_source,
    previous_val_loss,
    learning_rate,
    ema_momentum,
    elapsed_seconds,
    gpu_memory_allocated_bytes,
    gpu_memory_reserved_bytes,
    contributions,
):
    progress_epochs = epoch_index + progress_fraction
    mean_epoch_seconds = elapsed_seconds / max(progress_epochs, 1e-9)
    return {
        "architecture": config.encoder.architecture,
        "seed": config.seed,
        "epoch": epoch_index + 1,
        "total_epochs": config.optimization.epochs,
        "epoch_fraction": progress_fraction,
        "step": global_step,
        "total_steps": total_steps,
        "train_loss_mean": float(np.mean(list(train_per_source.values()))),
        "train_loss": train_per_source,
        "previous_validation_loss": previous_val_loss,
        "learning_rate": learning_rate,
        "ema_momentum": ema_momentum,
        "elapsed_seconds": elapsed_seconds,
        "eta_seconds": mean_epoch_seconds * max(0.0, config.optimization.epochs - progress_epochs),
        "gpu_memory_allocated_bytes": gpu_memory_allocated_bytes,
        "gpu_memory_reserved_bytes": gpu_memory_reserved_bytes,
        "dataset_contribution_examples": contributions,
    }


def _record_gpu_peak(metadata: dict, output_dir: Path, field: str, device: torch.device) -> int:
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    metadata["hardware"][field] = peak
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return peak


def main():
    parser = argparse.ArgumentParser(description="Label-free LatentWave JEPA pretraining with selectable ViT or official Mamba-2 encoders.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override the config seed; the resolved value is saved with the run.")
    parser.add_argument("--resume", default=None, help="Resume from latest.pt or another full training checkpoint.")
    parser.add_argument(
        "--reset-validation-selection",
        action="store_true",
        help="Clear prior best/last validation values when resuming after a validation-protocol change.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate config, HDF5 paths, model, device and one forward pass.")
    parser.add_argument("--smoke", action="store_true", help="Run one real HDF5-backed optimizer step and exit.")
    parser.add_argument("--smoke-batch-size", type=int, default=1, help="HDF5 sample count for the bounded smoke step (default: 1).")
    args = parser.parse_args()
    if args.reset_validation_selection and not args.resume:
        parser.error("--reset-validation-selection requires --resume")

    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (Path(__file__).resolve().parents[1] / config_path).resolve()
    config = init_config(str(config_path))
    if args.seed is not None:
        config.seed = args.seed
    set_seed(config.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; refusing to fall back to CPU.")
    if config.encoder.architecture == "mamba2" and device.type != "cuda":
        raise RuntimeError("The Mamba-2 experiment requires CUDA; CPU fallback is intentionally disabled.")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)

    train_loaders, val_loaders, source_paths = _build_loaders(
        config, config.masks, smoke=args.smoke, smoke_batch_size=args.smoke_batch_size
    )
    max_steps = max(len(loader) for loader in train_loaders.values())
    grad_accumulation = 1 if args.smoke else config.optimization.grad_accumulation_steps
    if not args.smoke:
        max_steps = _full_accumulation_steps(max_steps, grad_accumulation)
    updates_per_epoch = math.ceil(max_steps / grad_accumulation)
    model = JEPA(config, updates_per_epoch, eval_fns=[], metrics=[]).to(device)
    model.train()
    output_dir = _project_path(args.output_dir or config.checkpoint.dir)
    run_metadata = _write_run_files(output_dir, config, source_paths, model, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Exercise the configured model against the real collated HDF5 batch before a long run.
    first_loader = next(iter(train_loaders.values()))
    sample_batch = _move_batch(next(iter(first_loader)), device)
    with torch.no_grad(), _autocast(config, device):
        preflight_pred, preflight_target, _ = model(*sample_batch)
    if preflight_pred.shape != preflight_target.shape or not torch.isfinite(preflight_pred).all() or not torch.isfinite(preflight_target).all():
        raise RuntimeError("Preflight forward produced mismatched shapes or non-finite Mamba/JEPA tensors.")
    print(f"[preflight] encoder={config.encoder.architecture} shape={tuple(preflight_pred.shape)} device={device}")
    if args.dry_run:
        peak = _record_gpu_peak(run_metadata, output_dir, "preflight_peak_memory_bytes", device)
        print(f"[dry-run] data_sources={len(source_paths)} peak_memory_bytes={peak} output_dir={output_dir}")
        return

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    optimizers, scheduler_configs = model.configure_optimizers()
    optimizer = optimizers[0]
    schedulers = [item["scheduler"] for item in scheduler_configs]
    sampler_generators = {
        name: loader.sampler.generator
        for name, loader in train_loaders.items()
        if isinstance(loader.sampler, RandomSampler) and loader.sampler.generator is not None
    }

    if args.smoke:
        optimizer.zero_grad(set_to_none=True)
        with _autocast(config, device):
            loss, _, _, channel_count = model.compute_loss_per_patch(sample_batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Smoke JEPA loss is not finite: {loss.detach().item()}")
        loss.backward()
        student_grad = any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.context_encoder.parameters())
        predictor_grad = any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in model.predictor.parameters())
        if not student_grad or not predictor_grad:
            raise RuntimeError(f"Smoke gradients missing/nonzero check failed: student={student_grad}, predictor={predictor_grad}")
        probe_name = next(
            name
            for name, parameter in model.context_encoder.named_parameters()
            if parameter.grad is not None and parameter.grad.abs().sum() > 0
        )
        target_probe = dict(model.target_encoder.named_parameters())[probe_name]
        target_probe_before = target_probe.detach().clone()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        momentum = config.optimization.ema[0] + (config.optimization.ema[1] - config.optimization.ema[0]) / max(1, config.optimization.epochs)
        model.update_target(momentum)
        if momentum < 1.0 and torch.equal(target_probe, target_probe_before):
            raise RuntimeError("Smoke EMA update did not change the target after the student optimizer step.")
        _save_checkpoint(
            output_dir / "smoke.pt", model, optimizer, schedulers, 0, 1,
            float(loss.detach()), sampler_generators, config, run_metadata["git"],
        )
        peak = _record_gpu_peak(run_metadata, output_dir, "smoke_peak_memory_bytes", device)
        print(f"[smoke] passed device={device} loss={float(loss.detach()):.6f} channels={channel_count} student_grad={student_grad} predictor_grad={predictor_grad} ema={momentum:.6f} peak_memory_bytes={peak}")
        return

    start_epoch = 0
    global_step = 0
    best_val = float("inf")
    best_epoch = None
    last_val_loss = None
    elapsed_before_resume = 0.0
    if args.resume:
        resume_path = _project_path(args.resume)
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        checkpoint_architecture = checkpoint.get("architecture")
        checkpoint_seed = checkpoint.get("seed")
        if checkpoint_architecture != config.encoder.architecture or checkpoint_seed != config.seed:
            raise ValueError(
                "Resume checkpoint architecture/seed do not match the resolved run config: "
                f"checkpoint=({checkpoint_architecture}, {checkpoint_seed}), "
                f"config=({config.encoder.architecture}, {config.seed})."
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for scheduler, state in zip(schedulers, checkpoint["schedulers"]):
            scheduler.load_state_dict(state)
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_val = float(checkpoint["best_val_loss"])
        best_epoch = checkpoint.get("best_epoch")
        last_val_loss = checkpoint.get("last_val_loss")
        elapsed_before_resume = float(checkpoint.get("elapsed_seconds", 0.0))
        if args.reset_validation_selection:
            best_val = float("inf")
            best_epoch = None
            last_val_loss = None
        for name, state in checkpoint.get("sampler_states", {}).items():
            if name in sampler_generators:
                sampler_generators[name].set_state(state)
        torch.set_rng_state(checkpoint["torch_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        if checkpoint.get("cuda_rng_states") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        run_metadata.update({
            "completed_epochs": start_epoch,
            "global_step": global_step,
            "elapsed_seconds": elapsed_before_resume,
            "best_checkpoint_epoch": None if best_epoch is None else int(best_epoch) + 1,
            "validation_protocol": _VALIDATION_PROTOCOL,
        })
        print(f"[resume] loaded {resume_path} at completed_epoch={checkpoint['epoch']} global_step={global_step}")
        if args.reset_validation_selection:
            print("[selection-reset] prior best/last validation values cleared for deterministic validation protocol")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_stop_signal)

    updates_per_epoch = math.ceil(max_steps / grad_accumulation)
    total_updates = max(1, updates_per_epoch * config.optimization.epochs)
    progress_every = max(1, updates_per_epoch // 20)
    effective_examples_per_update = len(train_loaders) * config.data.batch_size * grad_accumulation
    run_metadata.update({
        "status": "running",
        "steps_per_epoch": updates_per_epoch,
        "microbatches_per_epoch": max_steps,
        "microbatches_per_optimizer_step": grad_accumulation,
        "effective_examples_per_optimizer_step": effective_examples_per_update,
        "total_steps": total_updates,
        "completed_epochs": start_epoch,
        "global_step": global_step,
    })
    _write_run_metadata(output_dir, run_metadata)
    log_path = output_dir / "train.jsonl"
    start_time = time.monotonic()
    for epoch in range(start_epoch, config.optimization.epochs):
        epoch_start_time = time.monotonic()
        model.train()
        iterators = {name: iter(loader) for name, loader in train_loaders.items()}
        losses = {name: 0.0 for name in train_loaders}
        contributions = {name: 0 for name in train_loaders}
        optimizer.zero_grad(set_to_none=True)

        for window_start in range(0, max_steps, grad_accumulation):
            window_size = min(grad_accumulation, max_steps - window_start)
            for offset in range(window_size):
                step = window_start + offset
                batches = _next_max_cycle_batches(train_loaders, iterators, step)
                for name, batch in batches.items():
                    moved = _move_batch(batch, device)
                    with _autocast(config, device):
                        groups = model.compute_loss_groups(moved)
                    batch_size = int(moved[0].shape[0])
                    contributions[name] += batch_size
                    for group in groups:
                        loss = group["loss"]
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Non-finite JEPA loss for source {name}: {loss.detach().item()}")
                        sample_weight = group["sample_count"] / batch_size
                        losses[name] += float(loss.detach()) * sample_weight
                        (loss * sample_weight / (len(batches) * window_size)).backward()

            if config.optimization.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for group in optimizer.param_groups for parameter in group["params"] if parameter.grad is not None],
                    config.optimization.grad_clip,
                )
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for group in optimizer.param_groups for parameter in group["params"]
            ):
                raise RuntimeError("Non-finite student/predictor gradient; optimizer and EMA update were skipped.")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()
            global_step += 1
            momentum = config.optimization.ema[0] + min(global_step, total_updates) / total_updates * (config.optimization.ema[1] - config.optimization.ema[0])
            model.update_target(momentum)
            if global_step % progress_every == 0:
                processed_microbatches = min(window_start + window_size, max_steps)
                progress_fraction = processed_microbatches / max_steps
                rolling_train = {
                    name: value / max(1, processed_microbatches)
                    for name, value in losses.items()
                }
                progress_elapsed = elapsed_before_resume + time.monotonic() - start_time
                progress_allocated = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
                progress_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
                progress = _progress_payload(
                    config, epoch, progress_fraction, global_step, total_updates,
                    rolling_train, last_val_loss, optimizer.param_groups[0]["lr"], momentum,
                    progress_elapsed, progress_allocated, progress_reserved, contributions,
                )
                print("[progress] " + json.dumps(progress), flush=True)

        val_per_source = _evaluate(model, val_loaders, device, config)
        val_loss = float(np.mean(list(val_per_source.values())))
        last_val_loss = val_loss
        train_per_source = {name: value / max(1, max_steps) for name, value in losses.items()}
        epoch_elapsed = time.monotonic() - epoch_start_time
        elapsed = elapsed_before_resume + time.monotonic() - start_time
        gpu_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        gpu_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        payload = _epoch_payload(
            config, epoch, global_step, total_updates, train_per_source, val_per_source,
            optimizer.param_groups[0]["lr"], momentum, elapsed, epoch_elapsed,
            gpu_memory, gpu_reserved, contributions,
        )
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload) + "\n")
        print("[epoch] " + json.dumps(payload), flush=True)

        run_metadata.update({
            "status": "running",
            "completed_epochs": epoch + 1,
            "global_step": global_step,
            "elapsed_seconds": elapsed,
            "eta_seconds": payload["eta_seconds"],
            "hardware": {
                **run_metadata["hardware"],
                "training_peak_memory_allocated_bytes": gpu_memory,
                "training_peak_memory_reserved_bytes": gpu_reserved,
            },
        })

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
        run_metadata["best_checkpoint_epoch"] = None if best_epoch is None else best_epoch + 1
        run_metadata["best_validation_loss"] = best_val
        run_metadata["last_validation_loss"] = last_val_loss
        _write_run_metadata(output_dir, run_metadata)
        if val_loss == best_val and best_epoch == epoch:
            _save_checkpoint(
                output_dir / "best.pt", model, optimizer, schedulers, epoch, global_step,
                best_val, sampler_generators, config, run_metadata["git"], elapsed, best_epoch, last_val_loss,
            )
        _save_checkpoint(
            output_dir / "latest.pt", model, optimizer, schedulers, epoch, global_step,
            best_val, sampler_generators, config, run_metadata["git"], elapsed, best_epoch, last_val_loss,
        )
        if (epoch + 1) % config.checkpoint.every_n_epochs == 0:
            _save_checkpoint(
                output_dir / f"epoch-{epoch + 1:04d}.pt", model, optimizer, schedulers,
                epoch, global_step, best_val, sampler_generators, config, run_metadata["git"], elapsed, best_epoch, last_val_loss,
            )
        if _STOP_AFTER_EPOCH:
            run_metadata.update({"status": "stopped_after_epoch", "checkpoint": str(output_dir / "latest.pt")})
            _write_run_metadata(output_dir, run_metadata)
            print(f"[stop] checkpointed completed epoch {epoch + 1}; resume with --resume {output_dir / 'latest.pt'}", flush=True)
            return

    elapsed = elapsed_before_resume + time.monotonic() - start_time
    run_metadata.update({"status": "completed", "elapsed_seconds": elapsed, "best_validation_loss": best_val})
    _write_run_metadata(output_dir, run_metadata)
    print(f"[done] training elapsed={dt.timedelta(seconds=int(elapsed))} best_val_loss={best_val:.6f}")


if __name__ == "__main__":
    main()
