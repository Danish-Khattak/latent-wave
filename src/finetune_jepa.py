from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from configs.jepa import init_config
from evaluate_linear import TASKS, _dataset, _resolve, _state_for_target
from models.encoder import build_encoder


class EncoderTaskModel(nn.Module):
    """Fine-tune a JEPA target encoder with a supervised task head."""

    def __init__(self, encoder: nn.Module, embed_dim: int, outputs: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(embed_dim, outputs)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(samples)
        if tokens.ndim == 3:
            tokens = tokens.mean(dim=1)
        elif tokens.ndim != 2:
            raise ValueError(f"Expected encoder output [batch,tokens,dim] or [batch,dim], got {tuple(tokens.shape)}")
        return self.head(tokens)


class IndexedTaskDataset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int], label_map: dict[int, int] | None):
        self.dataset = dataset
        self.indices = indices
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        row = self.dataset[self.indices[item]]
        samples, labels = row[0], row[1]
        if self.label_map is not None:
            original = int(torch.as_tensor(labels).item())
            if original not in self.label_map:
                raise ValueError(f"Label {original} is absent from the training partition's class map")
            labels = torch.tensor(self.label_map[original], dtype=torch.long)
        else:
            labels = torch.as_tensor(labels, dtype=torch.float32)
        return samples, labels


def _label_key(task: str) -> str:
    return "label_beam_16" if task == "deepmimo-beam" else TASKS[task]["label"]


def _read_labels(path: Path, key: str) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        if key not in h5:
            raise KeyError(f"Missing label dataset {key!r} in {path}")
        return np.asarray(h5[key][:])


def _load_shared_split(path: Path, labels: np.ndarray, sample_count: int, seed: int) -> tuple[list[int], list[int]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Shared linear-probe split manifest is missing: {path}. Run the frozen linear evaluator first."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("sample_count") != sample_count:
        raise ValueError(f"Split manifest sample count differs from dataset: {path}")
    label_hash = hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest()
    if manifest.get("label_sha256") != label_hash:
        raise ValueError(f"Split manifest labels/order differ from this dataset: {path}")
    try:
        split = manifest["splits"][str(seed)]
    except KeyError as exc:
        raise ValueError(f"Split seed {seed} is not in {path}") from exc
    train_indices = list(map(int, split["train"]))
    test_indices = list(map(int, split["test"]))
    train_set = set(train_indices)
    test_set = set(test_indices)
    if (
        len(train_set) != len(train_indices)
        or len(test_set) != len(test_indices)
        or train_set & test_set
        or train_set | test_set != set(range(sample_count))
    ):
        raise ValueError(f"Invalid overlapping or incomplete split in {path}")
    return train_indices, test_indices


def _position_range(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        coord_min = np.asarray(json.loads(h5.attrs["coord_nominal_min"]), dtype=np.float64)
        coord_max = np.asarray(json.loads(h5.attrs["coord_nominal_max"]), dtype=np.float64)
    return coord_min, coord_max


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(dataset: Dataset, args, device: torch.device, *, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        generator=generator,
        drop_last=False,
    )


def _lr_for_epoch(args, epoch: int) -> float:
    warmup = max(0, args.warmup_epochs)
    if warmup and epoch < warmup:
        return args.learning_rate * (epoch + 1) / warmup
    remaining = max(1, args.epochs - warmup)
    progress = min(1.0, max(0.0, (epoch - warmup) / remaining))
    ratio = args.minimum_lr / args.learning_rate
    return args.learning_rate * (ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def _optimizer_step(model: nn.Module, optimizer: torch.optim.Optimizer, accumulated_samples: int) -> None:
    if accumulated_samples <= 0:
        return
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(accumulated_samples)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _train_epoch(model, loader, optimizer, criterion, device, accumulation_steps: int, max_grad_norm: float | None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    samples_seen = 0
    accumulated_samples = 0
    batches_in_group = 0
    total_batches = len(loader)
    for batch_index, (samples, labels) in enumerate(loader):
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        predictions = model(samples)
        loss = criterion(predictions, labels)
        batch_count = int(samples.shape[0])
        (loss * batch_count).backward()
        loss_sum += float(loss.detach()) * batch_count
        samples_seen += batch_count
        accumulated_samples += batch_count
        batches_in_group += 1
        at_group_end = batches_in_group >= accumulation_steps or batch_index + 1 == total_batches
        if at_group_end:
            if max_grad_norm is not None:
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_samples)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                # Gradients are already averaged, so the common step helper must not divide twice.
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                _optimizer_step(model, optimizer, accumulated_samples)
            accumulated_samples = 0
            batches_in_group = 0
    return loss_sum / max(1, samples_seen)


@torch.inference_mode()
def _evaluate(model, loader, criterion, device, target_type: str, coord_min, coord_max) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    samples_seen = 0
    predictions_all = []
    targets_all = []
    for samples, labels in loader:
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        predictions = model(samples)
        count = int(samples.shape[0])
        loss_sum += float(criterion(predictions, labels)) * count
        samples_seen += count
        predictions_all.append(predictions.float().cpu().numpy())
        targets_all.append(labels.cpu().numpy())
    predictions = np.concatenate(predictions_all)
    targets = np.concatenate(targets_all)
    result: dict[str, Any] = {"loss": loss_sum / max(1, samples_seen)}
    if target_type == "classification":
        predicted_classes = predictions.argmax(axis=1)
        result["mean_per_class_accuracy"] = float(balanced_accuracy_score(targets, predicted_classes))
        result["predictions"] = predicted_classes
        result["targets"] = targets
    else:
        predicted_m = (predictions + 1.0) * 0.5 * (coord_max - coord_min) + coord_min
        targets_m = (targets + 1.0) * 0.5 * (coord_max - coord_min) + coord_min
        errors = np.linalg.norm(predicted_m - targets_m, axis=1)
        result["mean_positioning_error_m"] = float(errors.mean())
        result["predicted_m"] = predicted_m
        result["targets_m"] = targets_m
        result["errors_m"] = errors
    return result


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _validate_saved_run(saved: dict, args, split_path: Path, train_indices: list[int], val_indices: list[int]) -> None:
    saved_split = Path(saved.get("split_file", "")).expanduser().resolve()
    if saved_split != split_path.resolve():
        raise ValueError(f"Saved split manifest differs from the current split: {saved_split} != {split_path.resolve()}")
    saved_pretraining_checkpoint = Path(saved.get("pretraining_checkpoint", "")).expanduser().resolve()
    if saved_pretraining_checkpoint != Path(args.checkpoint_path).resolve():
        raise ValueError("Saved fine-tuning run was initialized from a different pretraining checkpoint")
    if saved.get("train_indices") != train_indices or saved.get("validation_indices") != val_indices:
        raise ValueError("Saved fine-tuning train/validation indices differ from the current split configuration")
    saved_args = saved.get("args", {})
    keys = (
        "epochs", "batch_size", "accumulation_steps", "learning_rate", "minimum_lr", "weight_decay",
        "warmup_epochs", "validation_fraction", "validation_seed", "random_seed_base", "patience", "max_grad_norm",
    )
    mismatches = {
        key: (saved_args.get(key), getattr(args, key))
        for key in keys
        if saved_args.get(key) != getattr(args, key)
    }
    if mismatches:
        raise ValueError(f"Fine-tuning resume/test settings differ from the selected run: {mismatches}")


def _run_split(args, config, checkpoint: dict, architecture: str, pretraining_seed: int, task: str,
               data_path: Path, split_path: Path, split_seed: int, output_dir: Path, device: torch.device) -> dict:
    result_path = output_dir / "result.json"
    validation_result_path = output_dir / "validation_result.json"
    if result_path.is_file():
        if args.validation_only:
            raise FileExistsError(
                f"Test-scored fine-tuning output already exists at {result_path}; refusing to reuse it for validation selection."
            )
        return json.loads(result_path.read_text(encoding="utf-8"))
    if args.validation_only and validation_result_path.is_file():
        return json.loads(validation_result_path.read_text(encoding="utf-8"))

    task_info = TASKS[task]
    labels = _read_labels(data_path, _label_key(task))
    dataset = _dataset(task, data_path)
    if len(dataset) != len(labels):
        raise ValueError(f"Dataset sample count {len(dataset)} does not match labels {len(labels)}")
    train_all, test_indices = _load_shared_split(split_path, labels, len(dataset), split_seed)
    target_type = task_info["target"]
    training_labels = labels[train_all]
    stratify = training_labels if target_type == "classification" else None
    train_indices, val_indices = train_test_split(
        train_all,
        test_size=args.validation_fraction,
        random_state=args.validation_seed + split_seed,
        stratify=stratify,
    )
    train_indices = list(map(int, train_indices))
    val_indices = list(map(int, val_indices))

    label_map = None
    class_values = None
    if target_type == "classification":
        class_values = sorted(map(int, np.unique(labels[train_indices]).tolist()))
        label_map = {value: index for index, value in enumerate(class_values)}
        outputs = len(class_values)
        if not class_values:
            raise ValueError("Training partition has no classes")
    else:
        outputs = int(labels.shape[1]) if labels.ndim > 1 else 1

    train_ds = IndexedTaskDataset(dataset, train_indices, label_map)
    val_ds = IndexedTaskDataset(dataset, val_indices, label_map)
    test_ds = IndexedTaskDataset(dataset, test_indices, label_map)
    train_loader = _make_loader(train_ds, args, device, shuffle=True, seed=args.random_seed_base + split_seed)
    val_loader = _make_loader(val_ds, args, device, shuffle=False, seed=args.random_seed_base + split_seed)

    coord_min = coord_max = None
    if target_type == "position":
        coord_min, coord_max = _position_range(data_path)
        if outputs != len(coord_min):
            raise ValueError(f"Position label dimension {outputs} differs from coordinate metadata {len(coord_min)}")

    _seed_all(args.random_seed_base + split_seed)
    encoder = build_encoder(config.encoder)
    target_state = _state_for_target(checkpoint)
    encoder.load_state_dict(target_state, strict=True)
    model = EncoderTaskModel(encoder, config.encoder.embed_dim, outputs).to(device)
    criterion = nn.CrossEntropyLoss() if target_type == "classification" else nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    history_path = output_dir / "history.jsonl"
    start_epoch = 0
    best_metric = float("-inf") if target_type == "classification" else float("inf")
    best_epoch = -1
    if args.test_only:
        if not best_path.is_file():
            raise FileNotFoundError(f"Test-only mode requires a validation-selected checkpoint at {best_path}")
        selected = torch.load(best_path, map_location="cpu", weights_only=False)
        expected = (architecture, task, split_seed, pretraining_seed)
        actual = (
            selected.get("architecture"), selected.get("task"),
            selected.get("split_seed"), selected.get("pretraining_seed"),
        )
        if expected != actual:
            raise ValueError(f"Selected checkpoint identity mismatch: expected {expected}, got {actual}")
        _validate_saved_run(selected, args, split_path, train_indices, val_indices)
        if selected.get("class_values") != class_values:
            raise ValueError("Selected checkpoint class mapping differs from the current training partition")
        model.load_state_dict(selected["model"], strict=True)
        best_metric = float(selected["best_validation_metric"])
        best_epoch = int(selected["best_epoch"])
        start_epoch = args.epochs
    elif args.resume:
        resume_path = Path(args.resume).expanduser().resolve()
        resumed = torch.load(resume_path, map_location="cpu", weights_only=False)
        expected = (architecture, task, split_seed, pretraining_seed)
        actual = (resumed.get("architecture"), resumed.get("task"), resumed.get("split_seed"), resumed.get("pretraining_seed"))
        if expected != actual:
            raise ValueError(f"Resume metadata mismatch: expected {expected}, got {actual}")
        _validate_saved_run(resumed, args, split_path, train_indices, val_indices)
        model.load_state_dict(resumed["model"], strict=True)
        optimizer.load_state_dict(resumed["optimizer"])
        start_epoch = int(resumed["epoch"]) + 1
        best_metric = float(resumed["best_validation_metric"])
        best_epoch = int(resumed["best_epoch"])
    elif latest_path.exists():
        raise FileExistsError(f"Partial fine-tuning run exists at {latest_path}; pass --resume {latest_path}")

    print(
        f"[finetune] arch={architecture} pretrain_seed={pretraining_seed} task={task} "
        f"split_seed={split_seed} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"outputs={outputs} device={device}"
    )
    mode = "a" if start_epoch else "w"
    with history_path.open(mode, encoding="utf-8") as history:
        for epoch in range(start_epoch, args.epochs):
            lr = _lr_for_epoch(args, epoch)
            for group in optimizer.param_groups:
                group["lr"] = lr
            train_loss = _train_epoch(
                model, train_loader, optimizer, criterion, device, args.accumulation_steps, args.max_grad_norm
            )
            val = _evaluate(model, val_loader, criterion, device, target_type, coord_min, coord_max)
            metric_key = "mean_per_class_accuracy" if target_type == "classification" else "mean_positioning_error_m"
            current = float(val[metric_key])
            improved = current > best_metric if target_type == "classification" else current < best_metric
            if improved:
                best_metric = current
                best_epoch = epoch
            payload = {
                "architecture": architecture,
                "pretraining_seed": pretraining_seed,
                "task": task,
                "split_seed": split_seed,
                "epoch": epoch + 1,
                "total_epochs": args.epochs,
                "train_loss": train_loss,
                "validation_loss": val["loss"],
                "validation_metric": current,
                "validation_metric_name": metric_key,
                "learning_rate": lr,
                "best_epoch": best_epoch + 1,
                "best_validation_metric": best_metric,
            }
            history.write(json.dumps(payload) + "\n")
            history.flush()
            print(
                f"[epoch] arch={architecture} task={task} split_seed={split_seed} "
                f"epoch={epoch + 1}/{args.epochs} train_loss={train_loss:.6f} "
                f"val_loss={val['loss']:.6f} val_{metric_key}={current:.6f} "
                f"best_epoch={best_epoch + 1} lr={lr:.3e}"
            )
            save_payload = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_metric": best_metric,
                "architecture": architecture,
                "task": task,
                "split_seed": split_seed,
                "pretraining_seed": pretraining_seed,
                "pretraining_checkpoint": str(args.checkpoint_path),
                "split_file": str(split_path),
                "train_indices": train_indices,
                "validation_indices": val_indices,
                "class_values": class_values,
                "args": vars(args),
            }
            _atomic_torch_save(save_payload, latest_path)
            if improved:
                _atomic_torch_save(save_payload, best_path)
            if args.patience and epoch + 1 - best_epoch >= args.patience:
                print(f"[early-stop] no validation improvement for {args.patience} epochs")
                break

    if not best_path.is_file():
        raise RuntimeError(f"No validation-selected fine-tuning checkpoint was saved in {output_dir}")
    selected = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(selected["model"], strict=True)
    best_metric = float(selected["best_validation_metric"])
    best_epoch = int(selected["best_epoch"])
    selected_args = selected.get("args", {})
    setting_names = (
        "epochs", "batch_size", "accumulation_steps", "learning_rate", "minimum_lr",
        "weight_decay", "warmup_epochs", "patience", "validation_fraction",
        "validation_seed", "random_seed_base",
    )
    selected_training_config = {
        name: selected_args.get(name, getattr(args, name))
        for name in setting_names
    }

    metric_key = "mean_per_class_accuracy" if target_type == "classification" else "mean_positioning_error_m"
    metric_name = "mean_per_class_accuracy_percent" if target_type == "classification" else "mean_positioning_error_m"
    scale = 100.0 if target_type == "classification" else 1.0
    if args.validation_only:
        val = _evaluate(model, val_loader, criterion, device, target_type, coord_min, coord_max)
        validation_result = {
            "experiment": "supervised_finetuning_validation_screen",
            "task": task,
            "architecture": architecture,
            "pretraining_seed": pretraining_seed,
            "checkpoint": str(args.checkpoint_path),
            "finetuned_checkpoint": str(best_path),
            "split_file": str(split_path),
            "split_seed": split_seed,
            "split_protocol": "shared 80/20 train/test indices; validation carved from train only",
            "validation_fraction_of_training_partition": args.validation_fraction,
            "selected_epoch": best_epoch + 1,
            "validation_metric": float(val[metric_key]) * scale,
            "metric": metric_name,
            "validation_loss": float(val["loss"]),
            "test_evaluation": "not run; held-out test partition remains unopened",
            "training_config": selected_training_config,
        }
        _write_json(validation_result, validation_result_path)
        return validation_result

    test_loader = _make_loader(test_ds, args, device, shuffle=False, seed=args.random_seed_base + split_seed)
    test = _evaluate(model, test_loader, criterion, device, target_type, coord_min, coord_max)
    if target_type == "classification":
        classes = np.arange(outputs)
        matrix = confusion_matrix(test["targets"], test["predictions"], labels=classes)
        per_class = np.divide(
            matrix.diagonal(), matrix.sum(axis=1), out=np.zeros(outputs, dtype=np.float64), where=matrix.sum(axis=1) != 0
        )
        test_output = {
            "mean_per_class_accuracy": test["mean_per_class_accuracy"],
            "confusion_matrix": matrix.tolist(),
            "per_class_accuracy": per_class.tolist(),
            "class_values": class_values,
        }
    else:
        prediction_path = output_dir / "test_position_predictions.csv"
        with prediction_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["true_x_m", "true_y_m", "true_z_m", "pred_x_m", "pred_y_m", "pred_z_m", "error_m"])
            for true, predicted, error in zip(test["targets_m"], test["predicted_m"], test["errors_m"]):
                writer.writerow([*map(float, true), *map(float, predicted), float(error)])
        test_output = {"mean_positioning_error_m": test["mean_positioning_error_m"], "predictions_csv": str(prediction_path)}

    result = {
        "experiment": "supervised_finetuning",
        "task": task,
        "architecture": architecture,
        "pretraining_seed": pretraining_seed,
        "checkpoint": str(args.checkpoint_path),
        "best_finetuned_checkpoint": str(best_path),
        "split_file": str(split_path),
        "split_seed": split_seed,
        "split_protocol": "shared 80/20 train/test indices; validation carved from train only",
        "validation_fraction_of_training_partition": args.validation_fraction,
        "selected_epoch": best_epoch + 1,
        "validation_metric": best_metric * scale,
        "metric": metric_name,
        "test_metric": test_output.get("mean_per_class_accuracy", test_output.get("mean_positioning_error_m")) * scale,
        "test": test_output,
        "training_config": selected_training_config,
    }
    _write_json(result, result_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supervised fine-tuning of a JEPA target encoder with train-only validation and shared test splits."
    )
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True, help="Full JEPA pretraining checkpoint selected by pretraining validation.")
    parser.add_argument("--config", required=True, help="Matching ViT or Mamba-2 pretraining configuration.")
    parser.add_argument("--split-dir", required=True, help="Existing split manifest created by evaluate_linear.py.")
    parser.add_argument("--out-dir", required=True, help="External directory for checkpoints, logs, and result JSON.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-seed", type=int, default=50000)
    parser.add_argument("--random-seed-base", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=2)
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument(
        "--validation-only", action="store_true",
        help="Train/select by the validation partition only; do not open the held-out test partition.",
    )
    phase.add_argument(
        "--test-only", action="store_true",
        help="Evaluate an existing validation-selected best.pt once on the held-out test partition.",
    )
    parser.add_argument("--resume", default="", help="Resume one split-seed run from its latest.pt checkpoint.")
    args = parser.parse_args()
    if len(args.split_seeds) != len(set(args.split_seeds)):
        parser.error("--split-seeds values must be unique")
    if args.resume and len(args.split_seeds) != 1:
        parser.error("--resume requires exactly one --split-seeds value so its checkpoint cannot be reused across splits")
    if args.resume and args.test_only:
        parser.error("--resume cannot be combined with --test-only; test-only mode loads the selected best.pt automatically")

    config_path, data_path, checkpoint_path, split_dir, output_root = map(
        _resolve, (args.config, args.data, args.checkpoint, args.split_dir, args.out_dir)
    )
    args.checkpoint_path = checkpoint_path
    config = init_config(str(config_path))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback is performed.")
    if config.encoder.architecture == "mamba2" and device.type != "cuda":
        raise RuntimeError("Mamba-2 fine-tuning requires CUDA.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    architecture = config.encoder.architecture
    checkpoint_architecture = checkpoint.get("architecture")
    if checkpoint_architecture is not None and checkpoint_architecture != architecture:
        raise ValueError(f"Checkpoint architecture {checkpoint_architecture!r} differs from config {architecture!r}")
    pretraining_seed = int(checkpoint.get("seed", config.seed))
    _state_for_target(checkpoint)  # Fail early if the input is not a full JEPA checkpoint.

    results = []
    for split_seed in args.split_seeds:
        split_path = split_dir / f"{args.task}_seed_splits.json"
        split_output = output_root / args.task / f"{architecture}_pretrain_seed{pretraining_seed}" / f"split_seed_{split_seed}"
        result = _run_split(
            args, config, checkpoint, architecture, pretraining_seed, args.task,
            data_path, split_path, split_seed, split_output, device,
        )
        results.append(result)
    if args.validation_only:
        metric_values = np.asarray([r["validation_metric"] for r in results], dtype=np.float64)
        aggregate = {
            "experiment": "supervised_finetuning_validation_screen",
            "task": args.task,
            "architecture": architecture,
            "pretraining_seed": pretraining_seed,
            "metric": results[0]["metric"],
            "split_seeds": [r["split_seed"] for r in results],
            "validation_scores": {str(r["split_seed"]): r["validation_metric"] for r in results},
            "mean": float(metric_values.mean()),
            "std": float(metric_values.std(ddof=1)) if len(metric_values) > 1 else None,
            "test_evaluation": "not run",
            "runs": results,
        }
        aggregate_path = output_root / "validation_results" / f"{architecture}_pretrain_seed{pretraining_seed}_{args.task}.json"
    else:
        metric_values = np.asarray([r["test_metric"] for r in results], dtype=np.float64)
        aggregate = {
            "experiment": "supervised_finetuning",
            "task": args.task,
            "architecture": architecture,
            "pretraining_seed": pretraining_seed,
            "metric": results[0]["metric"],
            "split_seeds": [r["split_seed"] for r in results],
            "test_scores": {str(r["split_seed"]): r["test_metric"] for r in results},
            "mean": float(metric_values.mean()),
            "std": float(metric_values.std(ddof=1)) if len(metric_values) > 1 else None,
            "runs": results,
        }
        aggregate_path = output_root / "results" / f"{architecture}_pretrain_seed{pretraining_seed}_{args.task}.json"
    _write_json(aggregate, aggregate_path)
    print(json.dumps({"result": str(aggregate_path), "mean": aggregate["mean"], "std": aggregate["std"]}, indent=2))


if __name__ == "__main__":
    main()
