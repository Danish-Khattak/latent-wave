from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from configs.jepa import init_config
from datasets.deepmimo import DeepMIMO
from datasets.image import ImageDataset
from models.encoder import build_encoder


TASKS = {
    "rfs": {"key": "image", "label": "label", "target": "classification", "file": "rfs.h5"},
    "pos": {"key": "features", "label": "label", "target": "position", "file": "nrpos-outdoor.h5"},
    "deepmimo-beam": {"key": "sample", "label": "label_beam_16", "target": "classification", "file": "deepmimo.h5"},
    "deepmimo-los": {"key": "sample", "label": "label_los", "target": "classification", "file": "deepmimo.h5"},
}


def _resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (Path(__file__).resolve().parents[1] / p).resolve()


def _state_for_target(checkpoint: dict) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model", checkpoint)
    prefix = "target_encoder."
    result = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not result:
        raise ValueError("Checkpoint has no target_encoder.* weights; expected a full JEPA checkpoint from pretrain_jepa.py.")
    return result


def _labels(path: Path, label_key: str) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        if label_key not in h5:
            raise KeyError(f"Missing downstream label dataset {label_key!r} in {path}")
        return np.asarray(h5[label_key][:])


def _dataset(task: str, path: Path):
    if task == "deepmimo-beam":
        return DeepMIMO(path, n_beams=16, label_key="label_beam")
    if task == "deepmimo-los":
        return DeepMIMO(path, label_key="label_los")
    if task == "rfs":
        return ImageDataset(path, sample_key="image", label_key="label")
    return ImageDataset(path, sample_key="features", label_key="label", label_dtype=torch.float32)


def _embed(encoder, dataset, device, batch_size, workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=device.type == "cuda")
    reps, labels = [], []
    encoder.eval()
    with torch.inference_mode():
        for samples, targets, *_ in loader:
            samples = samples.to(device, non_blocking=True)
            tokens = encoder(samples)
            reps.append(tokens.mean(dim=1).float().cpu().numpy())
            labels.append(targets.cpu().numpy())
    return np.concatenate(reps), np.concatenate(labels)


def _split_indices(n: int, labels: np.ndarray, target: str, seed: int):
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=labels if target == "classification" else None,
    )
    return train_idx.tolist(), test_idx.tolist()


def _split_manifest(path: Path, labels: np.ndarray, target: str, seeds: tuple[int, ...]):
    label_hash = hashlib.sha256(np.ascontiguousarray(labels).tobytes()).hexdigest()
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["sample_count"] != int(len(labels)) or manifest["label_sha256"] != label_hash:
            raise ValueError(f"Split file does not match the current labels/sample order: {path}")
        if sorted(map(int, manifest["splits"].keys())) != sorted(seeds):
            raise ValueError(f"Split file seeds differ from requested seeds: {path}")
        return manifest
    manifest = {
        "sample_count": int(len(labels)),
        "label_sha256": label_hash,
        "protocol": "sample-level 80/20; stratified classification; seed-specific shared indices",
        "seeds": list(seeds),
        "splits": {},
    }
    for seed in seeds:
        train_idx, test_idx = _split_indices(len(labels), labels, target, seed)
        manifest["splits"][str(seed)] = {"train": train_idx, "test": test_idx}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _metric(task: str, features: np.ndarray, labels: np.ndarray, train_idx: list[int], test_idx: list[int], path: Path, seed: int):
    x_train, x_test = features[train_idx], features[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed, C=5, solver="newton-cg"))
    if task != "pos":
        model.fit(x_train, y_train.astype(np.int64))
        prediction = model.predict(x_test)
        return float(balanced_accuracy_score(y_test.astype(np.int64), prediction)), "mean_per_class_accuracy_percent", 100.0

    model = make_pipeline(StandardScaler(), LinearRegression())
    model.fit(x_train, y_train.astype(np.float64))
    prediction = model.predict(x_test)
    with h5py.File(path, "r") as h5:
        coord_min = np.asarray(json.loads(h5.attrs["coord_nominal_min"]), dtype=np.float64)
        coord_max = np.asarray(json.loads(h5.attrs["coord_nominal_max"]), dtype=np.float64)
    pred_m = (prediction + 1.0) * 0.5 * (coord_max - coord_min) + coord_min
    true_m = (y_test + 1.0) * 0.5 * (coord_max - coord_min) + coord_min
    error = np.linalg.norm(pred_m - true_m, axis=1).mean()
    return float(error), "mean_positioning_error_m", 1.0


def main():
    parser = argparse.ArgumentParser(description="Frozen target-encoder linear evaluation using persisted, architecture-shared splits.")
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True, help="The exact matching ViT or Mamba-2 encoder configuration.")
    parser.add_argument("--split-dir", required=True, help="Reuse this directory for ViT and Mamba to share exact indices.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    cfg_path, data_path, ckpt_path = map(_resolve, (args.config, args.data, args.checkpoint))
    config = init_config(str(cfg_path))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; no CPU fallback is performed.")
    if config.encoder.architecture == "mamba2" and device.type != "cuda":
        raise RuntimeError("The Mamba-2 target encoder requires CUDA for evaluation.")

    info = TASKS[args.task]
    labels = _labels(data_path, info["label"])
    dataset = _dataset(args.task, data_path)
    if len(dataset) != len(labels):
        raise ValueError(f"Dataset sample count {len(dataset)} does not match labels {len(labels)}")

    encoder = build_encoder(config.encoder).to(device)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    checkpoint_architecture = checkpoint.get("architecture")
    if checkpoint_architecture is not None and checkpoint_architecture != config.encoder.architecture:
        raise ValueError(
            f"Checkpoint architecture {checkpoint_architecture!r} does not match "
            f"config architecture {config.encoder.architecture!r}."
        )
    pretraining_seed = checkpoint.get("seed")
    encoder.load_state_dict(_state_for_target(checkpoint), strict=True)
    features, loaded_labels = _embed(encoder, dataset, device, args.batch_size, args.workers)
    if not np.array_equal(loaded_labels.reshape(labels.shape), labels):
        raise ValueError("The dataset loader label order differs from the split manifest labels.")

    split_path = _resolve(args.split_dir) / f"{args.task}_seed_splits.json"
    manifest = _split_manifest(split_path, labels, info["target"], tuple(sorted(args.seeds)))
    scores = {}
    for seed in args.seeds:
        split = manifest["splits"][str(seed)]
        score, metric, scale = _metric(args.task, features, labels, split["train"], split["test"], data_path, seed)
        scores[str(seed)] = score * scale
    values = np.asarray(list(scores.values()), dtype=np.float64)
    result = {
        "task": args.task,
        "architecture": config.encoder.architecture,
        "pretraining_seed": pretraining_seed,
        "checkpoint": str(ckpt_path),
        "data": str(data_path),
        "split_file": str(split_path),
        "split_protocol": manifest["protocol"],
        "metric": metric,
        "seeds": scores,
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else None,
        "sample_count": int(len(labels)),
        "feature_dim": int(features.shape[1]),
        "selection_note": "No downstream test metric is used for pretraining checkpoint selection.",
    }
    output = _resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
