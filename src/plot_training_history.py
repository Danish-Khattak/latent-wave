from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {"mamba2": "#C56A2D", "vit": "#3A6EA5"}
PROTOCOL_LABELS = {
    "fixed_seed_masks_channels_each_example_once_v1": "fixed seeded, single-pass",
    "legacy_unrecorded": "legacy validation (protocol not recorded)",
}


def read_rows(run_dir: Path) -> tuple[dict, list[dict]]:
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    log_path = run_dir / "train.jsonl"
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        # Epoch records are authoritative because a resumed log can contain
        # epochs written under both the legacy and repaired validation code.
        row["validation_protocol"] = row.get("validation_protocol", "legacy_unrecorded")
        row["run_dir"] = str(run_dir)
    return metadata, rows


def deduplicate_identical_epoch_rows(rows: list[dict]) -> tuple[list[dict], int]:
    """Collapse byte-equivalent epoch metrics copied into staged resume dirs.

    Keep records with different losses, even when architecture/seed/epoch are
    the same, so separate experiments are never silently merged.
    """
    seen: set[str] = set()
    unique: list[dict] = []
    for row in rows:
        result = {
            key: row.get(key)
            for key in (
                "architecture", "seed", "epoch", "train_loss_mean",
                "val_loss_mean", "validation_protocol", "train_loss", "val_loss",
            )
        }
        signature = json.dumps(result, sort_keys=True, separators=(",", ":"))
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(row)
    return unique, len(rows) - len(unique)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot actual JEPA train/validation losses from saved pretraining run logs."
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="Run directories containing run_metadata.json and train.jsonl. Include resumed log segments when needed.",
    )
    parser.add_argument("--out-dir", required=True, help="External directory for CSV and PNG outputs.")
    args = parser.parse_args()

    run_dirs = [Path(value).expanduser().resolve() for value in args.run_dirs]
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    run_data: list[tuple[dict, list[dict]]] = []
    for run_dir in run_dirs:
        metadata, rows = read_rows(run_dir)
        run_data.append((metadata, rows))
        all_rows.extend(rows)
    all_rows, duplicate_count = deduplicate_identical_epoch_rows(all_rows)

    trace_path = out_dir / "pretraining_loss_traces.csv"
    fields = ["architecture", "seed", "epoch", "train_loss_mean", "val_loss_mean", "validation_protocol", "run_dir"]
    with trace_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(all_rows, key=lambda item: (item["architecture"], item["seed"], item["epoch"], item["run_dir"])):
            writer.writerow({key: row.get(key) for key in fields})

    # Consecutive Mamba logs from a checkpoint resume are combined only for
    # training loss, which is unaffected by the validation-protocol repair.
    train_groups: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    train_status: dict[tuple[str, int], str] = {}
    for metadata, rows in run_data:
        key = (metadata["architecture"], int(metadata["seed"]))
        train_status[key] = f"{metadata.get('completed_epochs', 0)} completed in latest run"
        for row in rows:
            train_groups[key][int(row["epoch"])] = float(row["train_loss_mean"])

    fig, axis = plt.subplots(figsize=(9.2, 5.2))
    for (architecture, seed), values_by_epoch in sorted(train_groups.items()):
        epochs = sorted(values_by_epoch)
        values = [values_by_epoch[epoch] for epoch in epochs]
        label = f"{architecture} seed {seed} ({train_status[(architecture, seed)]})"
        axis.plot(epochs, values, marker="o", linewidth=1.8, markersize=4, color=COLORS.get(architecture, "#555555"), label=label)
    max_epoch = max((int(row["epoch"]) for row in all_rows), default=1)
    axis.set_xlim(0, max(2, max_epoch + 2))
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Mean training JEPA loss (lower is better)")
    axis.set_title("Training JEPA loss — logged epochs only")
    axis.grid(True, alpha=0.25)
    if train_groups:
        axis.legend(frameon=False)
    fig.tight_layout()
    train_png = out_dir / "pretraining_training_loss.png"
    fig.savefig(train_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    protocols = sorted({row["validation_protocol"] for row in all_rows if row.get("val_loss_mean") is not None})
    fig, axes = plt.subplots(1, max(1, len(protocols)), figsize=(7.4 * max(1, len(protocols)), 5.0), squeeze=False)
    for axis, protocol in zip(axes.flat, protocols):
        protocol_rows = [row for row in all_rows if row.get("val_loss_mean") is not None and row["validation_protocol"] == protocol]
        groups: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
        for row in protocol_rows:
            groups[(row["architecture"], int(row["seed"]), row["run_dir"])].append(row)
        for (architecture, seed, run_dir), rows in sorted(groups.items()):
            rows.sort(key=lambda item: int(item["epoch"]))
            epochs = [int(row["epoch"]) for row in rows]
            values = [float(row["val_loss_mean"]) for row in rows]
            label = f"{architecture} seed {seed}"
            if len(epochs) > 1:
                axis.plot(epochs, values, marker="o", linewidth=1.8, markersize=4, color=COLORS.get(architecture, "#555555"), label=label)
            else:
                axis.scatter(epochs, values, color=COLORS.get(architecture, "#555555"), label=label, zorder=3)
        horizon = max((int(row["epoch"]) for row in protocol_rows), default=1)
        axis.set_xlim(0, max(2, horizon + 2))
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Mean validation JEPA loss (lower is better)")
        axis.set_title(PROTOCOL_LABELS.get(protocol, protocol))
        axis.grid(True, alpha=0.25)
        if groups:
            axis.legend(frameon=False)
    fig.suptitle("Validation JEPA loss by protocol — values across panels are not directly comparable", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    validation_png = out_dir / "pretraining_validation_protocols.png"
    fig.savefig(validation_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"wrote {train_png}")
    print(f"wrote {validation_png}")
    print(f"wrote {trace_path}")

    source_names = ["spectrogram", "rfp", "wifi-csi", "nrpos-indoor"]
    source_labels = {
        "spectrogram": "Spectrogram",
        "rfp": "RFP",
        "wifi-csi": "WiFi CSI",
        "nrpos-indoor": "NRPos Indoor",
    }
    source_fields = [
        "architecture", "seed", "epoch", "source", "train_loss", "val_loss",
        "validation_protocol", "run_dir",
    ]
    source_rows = []
    train_source_groups: dict[tuple[str, int, str], dict[int, float]] = defaultdict(dict)
    for row in all_rows:
        architecture = row["architecture"]
        seed = int(row["seed"])
        train_by_source = row.get("train_loss") or {}
        val_by_source = row.get("val_loss") or {}
        for source in source_names:
            train_value = train_by_source.get(source)
            val_value = val_by_source.get(source)
            if train_value is None and val_value is None:
                continue
            source_rows.append({
                "architecture": architecture,
                "seed": seed,
                "epoch": int(row["epoch"]),
                "source": source,
                "train_loss": train_value,
                "val_loss": val_value,
                "validation_protocol": row.get("validation_protocol", "legacy_unrecorded"),
                "run_dir": row["run_dir"],
            })
            if train_value is not None:
                train_source_groups[(architecture, seed, source)][int(row["epoch"])] = float(train_value)

    source_csv = out_dir / "pretraining_source_loss_traces.csv"
    with source_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=source_fields)
        writer.writeheader()
        writer.writerows(sorted(
            source_rows,
            key=lambda item: (item["architecture"], item["seed"], item["source"], item["epoch"], item["run_dir"]),
        ))

    source_fig, source_axes = plt.subplots(len(source_names), 2, figsize=(13.2, 14.0), squeeze=False)
    for source_index, source in enumerate(source_names):
        train_axis, val_axis = source_axes[source_index]
        for (architecture, seed, group_source), values_by_epoch in sorted(train_source_groups.items()):
            if group_source != source:
                continue
            epochs = sorted(values_by_epoch)
            values = [values_by_epoch[epoch] for epoch in epochs]
            label = f"{'Mamba-2' if architecture == 'mamba2' else architecture} seed {seed}"
            train_axis.plot(
                epochs, values, marker="o", linewidth=1.6, markersize=3.5,
                color=COLORS.get(architecture, "#555555"), label=label,
            )
        train_axis.set_title(f"{source_labels[source]} — training")
        train_axis.set_xlabel("Epoch")
        train_axis.set_ylabel("Mean train JEPA loss")
        train_axis.grid(True, alpha=0.25)
        if train_axis.lines:
            train_axis.legend(frameon=False, fontsize=8)

        val_groups: dict[tuple[str, int, str, str], list[dict]] = defaultdict(list)
        for row in all_rows:
            if source in (row.get("val_loss") or {}):
                val_groups[(
                    row["architecture"], int(row["seed"]),
                    row.get("validation_protocol", "legacy_unrecorded"), row["run_dir"],
                )].append(row)
        for (architecture, seed, protocol, _run_dir), rows in sorted(val_groups.items()):
            rows.sort(key=lambda item: int(item["epoch"]))
            epochs = [int(row["epoch"]) for row in rows]
            values = [float(row["val_loss"][source]) for row in rows]
            label = (
                f"{'Mamba-2' if architecture == 'mamba2' else architecture} seed {seed} — "
                f"{PROTOCOL_LABELS.get(protocol, protocol)}"
            )
            linestyle = ":" if protocol == "legacy_unrecorded" else "-"
            if len(epochs) > 1:
                val_axis.plot(
                    epochs, values, marker="o", linewidth=1.6, markersize=3.5,
                    linestyle=linestyle, color=COLORS.get(architecture, "#555555"), label=label,
                )
            else:
                val_axis.scatter(
                    epochs, values, color=COLORS.get(architecture, "#555555"), label=label, zorder=3,
                )
        val_axis.set_title(f"{source_labels[source]} — validation")
        val_axis.set_xlabel("Epoch")
        val_axis.set_ylabel("Mean validation JEPA loss")
        val_axis.grid(True, alpha=0.25)
        if val_axis.lines or val_axis.collections:
            val_axis.legend(frameon=False, fontsize=7.5)

    source_fig.suptitle("Per-source JEPA loss — training and validation", fontsize=15)
    source_fig.text(
        0.5, 0.012,
        "Validation lines are labeled by protocol; legacy and fixed-seed values should not be compared directly.",
        ha="center", va="bottom", fontsize=8, color="#555555",
    )
    source_fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    source_png = out_dir / "pretraining_source_losses.png"
    source_fig.savefig(source_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(source_fig)

    print(f"wrote {source_png}")
    print(f"wrote {source_csv}")
    if duplicate_count:
        print(f"collapsed {duplicate_count} exact copied epoch record(s) across run directories")


if __name__ == "__main__":
    main()
