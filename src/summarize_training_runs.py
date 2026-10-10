from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

from seed_statistics import summarize_seeded_results


TASKS = [
    ("rfs", "RF signal classification", "Mean-per-class accuracy (%)", "Higher is better"),
    ("pos", "5G NR outdoor positioning", "Mean positioning error (m)", "Lower is better"),
    ("deepmimo-beam", "DeepMIMO beam prediction", "Mean-per-class accuracy (%)", "Higher is better"),
    ("deepmimo-los", "DeepMIMO LoS/NLoS classification", "Mean-per-class accuracy (%)", "Higher is better"),
]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _checkpoint_summary(run_dir: Path) -> dict:
    """Read model size and selection metadata from the run's saved checkpoint."""
    checkpoint_path = run_dir / "latest.pt"
    if not checkpoint_path.is_file():
        checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.is_file():
        return {}

    # These are local experiment checkpoints, and loading on CPU avoids touching
    # the GPU while another training run may be active.
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", {})
    tensors = [(name, value) for name, value in state.items() if torch.is_tensor(value)]
    best_epoch = checkpoint.get("best_epoch")
    best_checkpoint_epoch = None if best_epoch is None else int(best_epoch) + 1
    if not tensors:
        return {
            "best_checkpoint_epoch": best_checkpoint_epoch,
            "best_validation_loss": checkpoint.get("best_val_loss"),
        }

    total = sum(value.numel() for _, value in tensors)
    context = sum(value.numel() for name, value in tensors if name.startswith("context_encoder."))
    target = sum(value.numel() for name, value in tensors if name.startswith("target_encoder."))
    predictor = sum(value.numel() for name, value in tensors if name.startswith("predictor."))
    # JEPA freezes the target encoder and fixed sinusoidal embeddings in the
    # context encoder and predictor. Those entries remain in the checkpoint.
    fixed_embeddings = sum(
        value.numel()
        for name, value in tensors
        if name in {"context_encoder.pos_embed", "predictor.pos_embed"}
    )
    return {
        "checkpoint_source": str(checkpoint_path),
        "total_parameters": total,
        "trainable_parameters": total - target - fixed_embeddings,
        "context_encoder_parameters": context,
        "target_encoder_parameters": target,
        "predictor_parameters": predictor,
        "best_checkpoint_epoch": best_checkpoint_epoch,
        "best_validation_loss": checkpoint.get("best_val_loss"),
    }


def _run_record(run_dir: Path) -> tuple[dict, list[dict]]:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing run metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    log = _read_jsonl(run_dir / "train.jsonl")
    config_path = run_dir / "resolved_config.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    latest = log[-1] if log else {}
    hardware = metadata.get("hardware", {})
    model = metadata.get("model", {})
    checkpoint = _checkpoint_summary(run_dir)
    peak_allocated = max(
        (int(item["gpu_memory_allocated_bytes"]) for item in log if item.get("gpu_memory_allocated_bytes") is not None),
        default=hardware.get("training_peak_memory_allocated_bytes"),
    )
    peak_reserved = max(
        (int(item["gpu_memory_reserved_bytes"]) for item in log if item.get("gpu_memory_reserved_bytes") is not None),
        default=hardware.get("training_peak_memory_reserved_bytes"),
    )
    record = {
        "architecture": metadata.get("architecture"),
        "seed": metadata.get("seed"),
        "status": metadata.get("status"),
        "validation_protocol": metadata.get("validation_protocol", "legacy_unrecorded"),
        "completed_epochs": metadata.get("completed_epochs", 0),
        "total_epochs": config.get("optimization", {}).get("epochs"),
        "global_step": metadata.get("global_step", 0),
        "total_steps": metadata.get("total_steps"),
        "train_loss_mean": latest.get("train_loss_mean"),
        "validation_loss_mean": latest.get("val_loss_mean"),
        "best_checkpoint_epoch": metadata.get("best_checkpoint_epoch")
        if metadata.get("best_checkpoint_epoch") is not None
        else checkpoint.get("best_checkpoint_epoch", metadata.get("best_epoch")),
        "source_examples_per_second": latest.get("source_examples_per_second"),
        "last_epoch_seconds": latest.get("epoch_elapsed_seconds"),
        "elapsed_seconds": metadata.get("elapsed_seconds", latest.get("elapsed_seconds")),
        "peak_vram_allocated_bytes": peak_allocated,
        "peak_vram_reserved_bytes": peak_reserved,
        "total_parameters": model.get("total_parameters") or checkpoint.get("total_parameters"),
        "trainable_parameters": model.get("trainable_parameters") or checkpoint.get("trainable_parameters"),
        "context_encoder_parameters": checkpoint.get("context_encoder_parameters"),
        "target_encoder_parameters": checkpoint.get("target_encoder_parameters"),
        "predictor_parameters": checkpoint.get("predictor_parameters"),
        "checkpoint_source": checkpoint.get("checkpoint_source"),
        "parameter_count_source": "checkpoint model tensor state" if checkpoint.get("total_parameters") is not None else None,
        "run_dir": str(run_dir),
        "latest_checkpoint": str(run_dir / "latest.pt") if (run_dir / "latest.pt").is_file() else None,
        "best_checkpoint": str(run_dir / "best.pt") if (run_dir / "best.pt").is_file() else None,
    }
    return record, log


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_downstream_results(paths: list[Path]) -> dict[tuple[str, str], dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        key = (result["task"], result["architecture"].lower())
        grouped.setdefault(key, []).append(result)
    return {key: summarize_seeded_results(results) for key, results in grouped.items()}


def main():
    parser = argparse.ArgumentParser(description="Summarize actual pretraining logs and report downstream evaluation status.")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Seed-specific run directories containing run_metadata.json and train.jsonl.")
    parser.add_argument("--out-dir", required=True, help="External directory for result tables and figures.")
    parser.add_argument(
        "--downstream-results",
        nargs="*",
        default=[],
        help="Completed evaluator JSON files to include in the downstream status table and plot.",
    )
    args = parser.parse_args()

    run_dirs = [Path(value).expanduser().resolve() for value in args.run_dirs]
    downstream_results = _read_downstream_results([Path(value).expanduser().resolve() for value in args.downstream_results])
    output_dir = Path(args.out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    logs = []
    for run_dir in run_dirs:
        record, run_log = _run_record(run_dir)
        records.append(record)
        logs.append((record, run_log))

    _write_csv(output_dir / "pretraining_runs.csv", records)
    (output_dir / "pretraining_runs.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    headings = [
        "Architecture", "Seed", "Status", "Epochs", "Steps", "Train loss", "Validation loss",
        "Best epoch", "Examples/s", "Epoch seconds", "Peak VRAM allocated (bytes)",
        "Total parameters", "Trainable parameters", "Run directory",
    ]
    lines = ["| " + " | ".join(headings) + " |", "|" + "|".join(["---"] * len(headings)) + "|"]
    for row in records:
        epochs = f"{row['completed_epochs']}/{row['total_epochs']}"
        steps = f"{row['global_step']}/{row['total_steps']}"
        values = [
            row["architecture"], row["seed"], row["status"], epochs, steps,
            row["train_loss_mean"], row["validation_loss_mean"], row["best_checkpoint_epoch"],
            row["source_examples_per_second"], row["last_epoch_seconds"],
            row["peak_vram_allocated_bytes"], row["total_parameters"], row["trainable_parameters"], row["run_dir"],
        ]
        lines.append("| " + " | ".join("n/a" if value is None else str(value) for value in values) + " |")
    (output_dir / "pretraining_runs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    colors = {"mamba2": "#C56A2D", "vit": "#3A6EA5"}
    protocol_labels = {
        "fixed_seed_masks_channels_each_example_once_v1": "fixed-seed, single-pass validation",
        "legacy_unrecorded": "legacy validation (protocol not recorded)",
    }
    seen_epoch_rows = set()
    for record, run_log in logs:
        if not run_log:
            continue
        color = colors.get(record["architecture"], "#555555")
        protocol_rows = {}
        for item in run_log:
            protocol = item.get("validation_protocol", "legacy_unrecorded")
            signature = json.dumps({
                key: item.get(key)
                for key in ("architecture", "seed", "epoch", "train_loss_mean", "val_loss_mean", "train_loss", "val_loss", "validation_protocol")
            }, sort_keys=True, separators=(",", ":"))
            if signature in seen_epoch_rows:
                continue
            seen_epoch_rows.add(signature)
            protocol_rows.setdefault(protocol, []).append(item)
        for protocol, rows in protocol_rows.items():
            rows.sort(key=lambda item: int(item["epoch"]))
            epochs = [int(item["epoch"]) for item in rows]
            values = [float(item["val_loss_mean"]) for item in rows]
            label = (
                f"{record['architecture']} seed {record['seed']} — "
                f"{protocol_labels.get(protocol, protocol)}"
            )
            style = ":" if protocol == "legacy_unrecorded" else "-"
            if len(epochs) > 1:
                axis.plot(epochs, values, marker="o", linewidth=1.8, markersize=5, color=color, linestyle=style, label=label)
            else:
                axis.scatter(epochs, values, s=42, color=color, label=label, zorder=3)
    observed_horizon = max(
        (max((item["epoch"] for item in run_log), default=0) for _, run_log in logs),
        default=0,
    )
    axis.set_xlim(0, max(2, observed_horizon + 3))
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean validation JEPA loss (lower is better)")
    all_runs_complete = all(
        row["status"] == "completed"
        and row["total_epochs"] is not None
        and row["completed_epochs"] >= row["total_epochs"]
        for row in records
    )
    if all_runs_complete:
        axis.set_title("Validation JEPA loss — completed pretraining runs")
    else:
        axis.set_title("Validation JEPA loss — partial pretraining runs; no final comparison")
    axis.grid(True, alpha=0.25)
    if any(run_log for _, run_log in logs):
        axis.legend(frameon=False)
    fig.text(
        0.5,
        0.015,
        "Legacy and fixed-seed validation values use different protocols and should not be compared directly.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(output_dir / "pretraining_validation_loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    task_rows = []
    for task_id, task_name, metric, direction in TASKS:
        mamba = downstream_results.get((task_id, "mamba2"))
        vit = downstream_results.get((task_id, "vit"))
        has_results = mamba is not None or vit is not None
        task_rows.append({
            "task": task_id,
            "task_name": task_name,
            "metric": metric,
            "direction": direction,
            "mamba2_status": "completed" if mamba is not None else "not_run",
            "vit_status": "completed" if vit is not None else "not_run",
            "mamba2_mean": None if mamba is None else mamba["mean"],
            "mamba2_std": None if mamba is None else mamba["std"],
            "mamba2_std_basis": None if mamba is None else mamba["std_basis"],
            "mamba2_pretraining_seed_std": None if mamba is None else mamba["pretraining_seed_std"],
            "mamba2_mean_split_seed_std": None if mamba is None else mamba["mean_split_seed_std"],
            "mamba2_pretraining_seeds": None if mamba is None else json.dumps(mamba["pretraining_seeds"]),
            "mamba2_pretraining_seed_means": None if mamba is None else json.dumps(mamba["pretraining_seed_means"], sort_keys=True),
            "mamba2_split_std_by_pretraining_seed": None if mamba is None else json.dumps(mamba["split_std_by_pretraining_seed"], sort_keys=True),
            "vit_mean": None if vit is None else vit["mean"],
            "vit_std": None if vit is None else vit["std"],
            "vit_std_basis": None if vit is None else vit["std_basis"],
            "vit_pretraining_seed_std": None if vit is None else vit["pretraining_seed_std"],
            "vit_mean_split_seed_std": None if vit is None else vit["mean_split_seed_std"],
            "vit_pretraining_seeds": None if vit is None else json.dumps(vit["pretraining_seeds"]),
            "vit_pretraining_seed_means": None if vit is None else json.dumps(vit["pretraining_seed_means"], sort_keys=True),
            "vit_split_std_by_pretraining_seed": None if vit is None else json.dumps(vit["split_std_by_pretraining_seed"], sort_keys=True),
            "reason": "Measured evaluator output available." if has_results else "No downstream evaluator output is available.",
        })
    _write_csv(output_dir / "downstream_status.csv", task_rows)
    (output_dir / "downstream_status.json").write_text(json.dumps(task_rows, indent=2), encoding="utf-8")
    status_lines = [
        "| Task | Metric | Direction | Mamba-2 mean ± SD | Mamba SD basis | Mamba within-checkpoint split SD | ViT mean ± SD | ViT SD basis | ViT within-checkpoint split SD |",
        "|---|---|---|---:|---|---:|---:|---|---:|",
    ]
    for row in task_rows:
        mamba_value = "Not run" if row["mamba2_mean"] is None else f"{row['mamba2_mean']:.3f} ± {row['mamba2_std']:.3f}" if row["mamba2_std"] is not None else f"{row['mamba2_mean']:.3f}"
        vit_value = "Not run" if row["vit_mean"] is None else f"{row['vit_mean']:.3f} ± {row['vit_std']:.3f}" if row["vit_std"] is not None else f"{row['vit_mean']:.3f}"
        mamba_split_std = row["mamba2_mean_split_seed_std"]
        vit_split_std = row["vit_mean_split_seed_std"]
        status_lines.append(
            f"| {row['task_name']} | {row['metric']} | {row['direction']} | {mamba_value} | "
            f"{row['mamba2_std_basis'] or 'n/a'} | {mamba_split_std if mamba_split_std is not None else 'n/a'} | "
            f"{vit_value} | {row['vit_std_basis'] or 'n/a'} | {vit_split_std if vit_split_std is not None else 'n/a'} |"
        )
    (output_dir / "downstream_status.md").write_text("\n".join(status_lines) + "\n", encoding="utf-8")

    has_any_score = any(row["mamba2_mean"] is not None or row["vit_mean"] is not None for row in task_rows)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2))
    for axis, row in zip(axes.flat, task_rows):
        values = [row["mamba2_mean"], row["vit_mean"]]
        errors = [row["mamba2_std"], row["vit_std"]]
        if values[0] is None and values[1] is None:
            axis.text(0.5, 0.64, row["task_name"], ha="center", va="center", fontsize=12, weight="semibold", transform=axis.transAxes)
            axis.text(0.5, 0.42, row["metric"], ha="center", va="center", fontsize=10, transform=axis.transAxes)
            axis.text(0.5, 0.24, row["direction"], ha="center", va="center", fontsize=9, color="#555555", transform=axis.transAxes)
            axis.text(0.5, 0.06, "NOT RUN — no evaluator output", ha="center", va="center", fontsize=9, color="#8A3B2F", transform=axis.transAxes)
            axis.set_xticks([])
            axis.set_yticks([])
        else:
            for index, (architecture, value, error, color) in enumerate(zip(
                ("Mamba-2", "ViT"), values, errors, ("#B44E3A", "#416FA6")
            )):
                if value is None:
                    axis.text(index, 0.04, "not run", ha="center", va="bottom", fontsize=8, color="#777777", rotation=90)
                    continue
                axis.bar(index, value, width=0.62, color=color, yerr=0.0 if error is None else error,
                         capsize=3, error_kw={"elinewidth": 1.0, "ecolor": "#263238"})
                text_value = f"{value:.3f}" if row["task"] == "pos" else f"{value:.1f}"
                if error is not None:
                    text_value += f" ± {error:.3f}" if row["task"] == "pos" else f" ± {error:.2f}"
                axis.text(index, value + (0.025 if row["task"] == "pos" else 1.0), text_value,
                          ha="center", va="bottom", fontsize=7.5)
            axis.set_xticks([0, 1], ["Mamba-2", "ViT"])
            axis.set_ylabel(row["metric"])
            if row["task"] == "pos":
                axis.set_ylim(0, max(value or 0 for value in values) * 1.25)
            else:
                axis.set_ylim(0, max(100.0, max(value or 0 for value in values) * 1.15))
            axis.grid(axis="y", alpha=0.22)
        axis.set_title(f"{row['task_name']}\n{row['direction']}", fontsize=10.5)
        for spine in axis.spines.values():
            spine.set_color("#C7CDD4")
    fig.suptitle("Downstream task comparison — metric direction shown per panel" if has_any_score else "Downstream task status — no scores available", fontsize=14)
    fig.text(
        0.5,
        0.005,
        "Error bars use SD across pretraining-seed means when multiple checkpoints are available; "
        "with one checkpoint they use SD across downstream split seeds.",
        ha="center",
        va="bottom",
        fontsize=7.2,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    fig.savefig(output_dir / "downstream_task_status.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"wrote {output_dir / 'pretraining_runs.csv'}")
    print(f"wrote {output_dir / 'pretraining_validation_loss.png'}")
    print(f"wrote {output_dir / 'downstream_status.csv'}")
    print(f"wrote {output_dir / 'downstream_task_status.png'}")


if __name__ == "__main__":
    main()
