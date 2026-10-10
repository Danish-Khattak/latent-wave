from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


COLORS = {"mamba2": "#C56A2D", "vit": "#3A6EA5"}


def _config_fingerprint(row: dict[str, str]) -> str:
    """Identify equivalent snapshots while preserving distinct model configs."""
    run_dir = Path(row.get("run_dir", ""))
    config_path = run_dir / "resolved_config.yml"
    if not config_path.is_file():
        return str(run_dir.resolve())
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if isinstance(config, dict) and isinstance(config.get("checkpoint"), dict):
        # Checkpoint output directories are run-specific, not model/training settings.
        config["checkpoint"].pop("dir", None)
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _latest_snapshots(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Collapse copied/resumed snapshots of the same architecture/seed/config."""
    selected: dict[tuple[str, str, str], tuple[int, dict[str, str]]] = {}
    for index, row in enumerate(rows):
        key = (
            row.get("architecture", ""),
            row.get("seed", ""),
            _config_fingerprint(row),
        )
        previous = selected.get(key)
        epoch = int(row.get("completed_epochs") or 0)
        if previous is None or epoch >= int(previous[1].get("completed_epochs") or 0):
            # On tied epochs prefer the later supplied path; the current resumed
            # directory follows its preserved source checkpoint in the run list.
            selected[key] = (index, row)
    return [row for _, row in sorted(selected.values(), key=lambda item: item[0])]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot measured model size and training compute from run summaries.")
    parser.add_argument("--pretraining-runs", required=True, help="CSV written by summarize_training_runs.py.")
    parser.add_argument("--out-dir", required=True, help="External directory for the PNG output.")
    args = parser.parse_args()

    runs_path = Path(args.pretraining_runs).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with runs_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rows = [row for row in rows if row.get("architecture") in COLORS]
    rows = _latest_snapshots(rows)
    if not rows:
        raise ValueError(f"No Mamba-2 or ViT runs found in {runs_path}")
    snapshot_path = out_dir / "pretraining_compute_snapshots.csv"
    with snapshot_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels = [f"{row['architecture']} seed {row['seed']}\n{row['completed_epochs']}/{row['total_epochs']} epochs" for row in rows]
    run_status = "; ".join(
        f"{row['architecture']} seed {row['seed']} {row['completed_epochs']}/{row['total_epochs']} epochs"
        for row in rows
    )
    colors = [COLORS[row["architecture"]] for row in rows]
    x = list(range(len(rows)))

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.0))
    width = 0.34
    total = [int(row["total_parameters"]) / 1e6 for row in rows]
    trainable = [int(row["trainable_parameters"]) / 1e6 for row in rows]
    axes[0, 0].bar([value - width / 2 for value in x], total, width, color=colors, alpha=0.5, label="Total")
    axes[0, 0].bar([value + width / 2 for value in x], trainable, width, color=colors, label="Trainable")
    axes[0, 0].set_title("JEPA model parameter counts")
    axes[0, 0].set_ylabel("Parameters (millions)")
    axes[0, 0].legend(frameon=False)

    throughput = [float(row["source_examples_per_second"]) for row in rows]
    axes[0, 1].bar(x, throughput, color=colors)
    axes[0, 1].set_title("Latest completed epoch throughput")
    axes[0, 1].set_ylabel("Source examples / second")
    for i, value in enumerate(throughput):
        axes[0, 1].text(i, value, f"{value:.1f}", ha="center", va="bottom", fontsize=9)

    allocated = [int(row["peak_vram_allocated_bytes"]) / 2**30 for row in rows]
    reserved = [int(row["peak_vram_reserved_bytes"]) / 2**30 for row in rows]
    axes[1, 0].bar([value - width / 2 for value in x], allocated, width, color=colors, alpha=0.55, label="Allocated")
    axes[1, 0].bar([value + width / 2 for value in x], reserved, width, color=colors, label="Reserved")
    axes[1, 0].set_title("Peak GPU memory")
    axes[1, 0].set_ylabel("GiB")
    axes[1, 0].legend(frameon=False)

    epoch_minutes = [float(row["last_epoch_seconds"]) / 60 for row in rows]
    elapsed_hours = [float(row["elapsed_seconds"]) / 3600 for row in rows]
    axes[1, 1].bar([value - width / 2 for value in x], epoch_minutes, width, color=colors, alpha=0.55, label="Latest epoch (min)")
    axes[1, 1].set_title("Time at this snapshot")
    axes[1, 1].set_ylabel("Latest epoch (minutes)")
    secondary = axes[1, 1].twinx()
    secondary.scatter(x, elapsed_hours, marker="D", color="#222222", label="Elapsed run time (h)", zorder=5)
    secondary.set_ylabel("Cumulative elapsed time (hours)")
    handles1, labels1 = axes[1, 1].get_legend_handles_labels()
    handles2, labels2 = secondary.get_legend_handles_labels()
    axes[1, 1].legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper left", fontsize=8)

    for axis in axes.flat:
        axis.set_xticks(x, labels, fontsize=8)
        axis.grid(axis="y", alpha=0.22)
        axis.spines["top"].set_visible(False)
    fig.suptitle("Measured pretraining compute — partial runs", fontsize=14)
    fig.text(
        0.5,
        0.015,
        f"Latest snapshot per identical architecture/seed/config: {run_status}. Total includes student, frozen target, and predictor; trainable excludes frozen target and fixed positional embeddings. Metrics are logged measurements, not full-run costs.",
        ha="center",
        fontsize=8.5,
        color="#555F69",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    path = out_dir / "pretraining_compute_comparison.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")
    print(f"wrote {snapshot_path}")


if __name__ == "__main__":
    main()
