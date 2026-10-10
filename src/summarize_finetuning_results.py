from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TASK_ORDER = ("rfs", "pos", "deepmimo-beam", "deepmimo-los")
TASK_LABELS = {
    "rfs": "RF signal classification",
    "pos": "5G NR outdoor positioning",
    "deepmimo-beam": "DeepMIMO beam prediction",
    "deepmimo-los": "DeepMIMO LoS/NLoS classification",
}
METRIC_DIRECTIONS = {
    "mean_per_class_accuracy_percent": "higher is better",
    "mean_positioning_error_m": "lower is better",
}


def _load_results(paths: list[Path]) -> dict[tuple[str, str], dict]:
    groups: dict[tuple[str, str], dict] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("experiment") != "supervised_finetuning":
            raise ValueError(f"Expected final supervised fine-tuning output, got {path}")
        if int(payload.get("pretraining_seed", -1)) != 0:
            raise ValueError(f"This summary expects the seed-0 fine-tuning comparison: {path}")
        task, architecture = payload.get("task"), payload.get("architecture")
        if task not in TASK_ORDER or architecture not in {"mamba2", "vit"}:
            raise ValueError(f"Unexpected task or architecture in {path}")
        key = (task, architecture)
        if key in groups:
            raise ValueError(f"Duplicate fine-tuning result for {task}/{architecture}")
        scores = {str(k): float(v) for k, v in payload.get("test_scores", {}).items()}
        if sorted(map(int, scores)) != [0, 1, 2]:
            raise ValueError(f"Expected test results for split seeds 0, 1, and 2 in {path}")
        if payload.get("metric") not in METRIC_DIRECTIONS:
            raise ValueError(f"Unknown metric in {path}: {payload.get('metric')}")
        payload["result_path"] = str(path.resolve())
        payload["test_scores"] = scores
        groups[key] = payload
    expected = {(task, arch) for task in TASK_ORDER for arch in ("mamba2", "vit")}
    missing = sorted(expected - groups.keys())
    if missing:
        raise ValueError(f"Missing fine-tuning results: {missing}")
    return groups


def _selection(selection_dir: Path, architecture: str, task: str) -> dict:
    path = selection_dir / f"{architecture}_{task}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("architecture") != architecture or payload.get("task") != task:
        raise ValueError(f"Candidate-selection metadata mismatch: {path}")
    return payload


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_comparison(groups: dict[tuple[str, str], dict], out: Path) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for axis, task in zip(axes.flat, TASK_ORDER):
        records = [groups[(task, arch)] for arch in ("mamba2", "vit")]
        metric = records[0]["metric"]
        if any(item["metric"] != metric for item in records):
            raise ValueError(f"Metric differs between architectures for {task}")
        means = [float(item["mean"]) for item in records]
        deviations = [float(item["std"] or 0.0) for item in records]
        axis.bar(
            ["Mamba-2", "ViT"], means, yerr=deviations, capsize=5,
            color=["#D77A32", "#4776A8"],
        )
        axis.set_title(f"{TASK_LABELS[task]}\n{metric.replace('_', ' ')} — {METRIC_DIRECTIONS[metric]}")
        axis.set_ylabel("Mean-per-class accuracy (%)" if metric == "mean_per_class_accuracy_percent" else "Mean positioning error (m)")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Supervised fine-tuning — pretraining seed 0; mean ± SD over split seeds 0–2", y=1.02)
    fig.tight_layout()
    path = out / "finetuning_comparison.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_confusions(groups: dict[tuple[str, str], dict], out: Path) -> list[str]:
    paths = []
    for task in TASK_ORDER:
        if task == "pos":
            continue
        for architecture in ("mamba2", "vit"):
            payload = groups[(task, architecture)]
            runs = payload["runs"]
            class_values = None
            total = None
            for run in runs:
                test = run.get("test", {})
                classes = list(map(int, test.get("class_values", [])))
                matrix = np.asarray(test.get("confusion_matrix", []), dtype=np.int64)
                if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != len(classes):
                    raise ValueError(f"Missing or malformed confusion matrix for {task}/{architecture}")
                if class_values is None:
                    class_values = classes
                    total = np.zeros_like(matrix)
                elif class_values != classes or total.shape != matrix.shape:
                    raise ValueError(f"Class mapping differs across split seeds for {task}/{architecture}")
                total += matrix
            assert class_values is not None and total is not None
            row_totals = total.sum(axis=1, keepdims=True)
            normalized = np.divide(total, row_totals, out=np.zeros_like(total, dtype=np.float64), where=row_totals > 0)
            size = max(6.0, min(14.0, 0.42 * len(class_values) + 3.5))
            fig, axis = plt.subplots(figsize=(size, size * 0.86))
            image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
            labels = [str(value) for value in class_values]
            axis.set_xticks(np.arange(len(labels)), labels, rotation=90 if len(labels) > 12 else 0)
            axis.set_yticks(np.arange(len(labels)), labels)
            axis.set_xlabel("Predicted class")
            axis.set_ylabel("True class")
            axis.set_title(f"{TASK_LABELS[task]} — {architecture}\nRow-normalized confusion; counts pooled over split seeds 0–2")
            fig.colorbar(image, ax=axis, label="Recall fraction")
            fig.tight_layout()
            path = out / f"finetuning_confusion_{task}_{architecture}.png"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            paths.append(str(path))
    return paths


def _plot_per_class_accuracy(groups: dict[tuple[str, str], dict], out: Path) -> list[str]:
    paths = []
    for task in TASK_ORDER:
        if task == "pos":
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True, squeeze=False)
        for axis, architecture, color in zip(axes.flat, ("mamba2", "vit"), ("#D77A32", "#4776A8")):
            payload = groups[(task, architecture)]
            class_values = None
            split_scores = []
            for run in payload["runs"]:
                test = run.get("test", {})
                classes = list(map(int, test.get("class_values", [])))
                scores = np.asarray(test.get("per_class_accuracy", []), dtype=np.float64) * 100.0
                if scores.ndim != 1 or len(classes) != len(scores) or not classes:
                    raise ValueError(f"Missing or malformed per-class accuracy for {task}/{architecture}")
                if class_values is None:
                    class_values = classes
                elif class_values != classes:
                    raise ValueError(f"Class mapping differs across split seeds for {task}/{architecture}")
                split_scores.append(scores)
            assert class_values is not None
            values = np.stack(split_scores)
            mean = values.mean(axis=0)
            deviation = values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(mean)
            x = np.arange(len(class_values))
            axis.bar(x, mean, yerr=deviation, capsize=2, color=color, ecolor="#30343B", linewidth=0.3)
            tick_indices = np.unique(np.linspace(0, len(class_values) - 1, min(12, len(class_values)), dtype=int))
            axis.set_xticks(tick_indices, [str(class_values[i]) for i in tick_indices], rotation=45, ha="right")
            axis.set_ylim(0, 100)
            axis.set_title("Mamba-2" if architecture == "mamba2" else "ViT")
            axis.set_xlabel("Class label")
            axis.grid(axis="y", alpha=0.25)
        axes[0, 0].set_ylabel("Mean-per-class accuracy (%)")
        fig.suptitle(
            f"{TASK_LABELS[task]} — per-class test accuracy\n"
            "Mean ± SD across downstream split seeds 0–2; pretraining seed 0",
            y=1.04,
        )
        fig.tight_layout()
        path = out / f"finetuning_per_class_{task}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))
    return paths


def _plot_position_errors(groups: dict[tuple[str, str], dict], out: Path) -> tuple[str, str]:
    errors_by_arch: dict[str, list[np.ndarray]] = defaultdict(list)
    rows = []
    for architecture in ("mamba2", "vit"):
        payload = groups[("pos", architecture)]
        for run in payload["runs"]:
            prediction_path = Path(run["test"]["predictions_csv"]).expanduser().resolve()
            with prediction_path.open(newline="", encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    error = float(row["error_m"])
                    errors_by_arch[architecture].append(np.asarray([error]))
                    rows.append({"architecture": architecture, "pretraining_seed": 0, "split_seed": int(run["split_seed"]), "error_m": error})
    fig, axis = plt.subplots(figsize=(8, 5))
    for architecture, color in (("mamba2", "#D77A32"), ("vit", "#4776A8")):
        values = np.concatenate(errors_by_arch[architecture])
        axis.hist(values, bins=35, density=True, histtype="step", linewidth=2, color=color, label=f"{architecture} (n={len(values)})")
    axis.set_title("Fine-tuning 5G NR positioning error — lower is better\nPretraining seed 0; pooled split seeds 0–2")
    axis.set_xlabel("Euclidean position error (m)")
    axis.set_ylabel("Density")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    plot_path = out / "finetuning_position_error_distribution.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    csv_path = out / "finetuning_position_errors.csv"
    _write_csv(csv_path, ["architecture", "pretraining_seed", "split_seed", "error_m"], rows)
    return str(plot_path), str(csv_path)


def _publish_figures(figure_paths: list[str], comparison_dir: Path) -> list[str]:
    comparison_dir.mkdir(parents=True, exist_ok=True)
    published = []
    for value in figure_paths:
        source = Path(value).resolve()
        destination = comparison_dir / source.name
        if source != destination.resolve():
            shutil.copy2(source, destination)
        published.append(str(destination.resolve()))
    return published


def _update_figure_index(index_path: Path, figure_paths: list[str]) -> None:
    if not index_path.is_file():
        return
    text = index_path.read_text(encoding="utf-8")
    marker = "## Supervised fine-tuning figures"
    base = text.split(marker, 1)[0].rstrip()
    base = base.replace(
        "Confusion matrices and position-prediction plots are unavailable because the evaluator saves aggregate metrics only.",
        "The frozen linear-probe evaluator saves aggregate metrics only; fine-tuning figures below include per-class and prediction-error detail.",
    )
    rows = []
    for value in figure_paths:
        path = Path(value)
        name = path.name
        if name == "finetuning_comparison.png":
            contents = "Four-task Mamba-2/ViT supervised fine-tuning scores with metric direction and split-seed variability."
        elif name.startswith("finetuning_confusion_"):
            contents = "Held-out classification confusion matrix pooled across downstream split seeds 0–2."
        elif name.startswith("finetuning_per_class_"):
            contents = "Per-class held-out accuracy, with mean and SD across downstream split seeds 0–2."
        else:
            contents = "Held-out 5G NR positioning error distribution from saved test predictions."
        rows.append(f"| `{name}` | {contents} | Locally measured fine-tuning test results; pretraining seed 0; LR selected on validation only |")
    section = [
        marker,
        "",
        "These figures are separate from frozen-probe results. Each task/architecture learning rate was selected on training-derived validation data before held-out test evaluation.",
        "",
        "| PNG | Contents | Data type |",
        "|---|---|---|",
        *rows,
    ]
    index_path.write_text(base + "\n\n" + "\n".join(section) + "\n", encoding="utf-8")


def _update_results_report(summary_rows: list[dict], figure_paths: list[str], result_dir: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    report_path = repo_root / "docs" / "mamba2_experiment" / "07_training_and_results.md"
    if not report_path.is_file():
        return
    now = datetime.now().astimezone()
    stamp = now.strftime("%Y-%m-%d %H:%M %Z")
    marker = "## Supervised fine-tuning results —"
    text = report_path.read_text(encoding="utf-8")
    base = text.split(marker, 1)[0].rstrip()
    lines = base.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Report updated:"):
            lines[index] = f"Report updated: {stamp}. Values are taken from current run metadata, console/JSONL logs, checkpoints, and evaluator output."
            break
    for index, line in enumerate(lines):
        if line.startswith("## Current verified status —"):
            lines[index] = line.replace("## Current verified status —", "## Pre-finetuning status snapshot —", 1)
            break
    base = "\n".join(lines).rstrip()

    run_dirs = []
    for architecture in ("mamba2", "vit"):
        for seed in (0, 1, 2):
            run_name = f"{architecture}_prose_seed{seed}" if seed else f"{architecture}_prose_seed0_deterministic_val"
            run_dir = repo_root.parent / "runs" / "latent-wave" / "full_runs" / run_name
            metadata_path = run_dir / "run_metadata.json"
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                run_dirs.append(
                    f"| {architecture} | {seed} | {metadata.get('status', 'unknown')} | "
                    f"{metadata.get('completed_epochs', 0)}/240 | {metadata.get('best_checkpoint_epoch', 'n/a')} |"
                )
            else:
                run_dirs.append(f"| {architecture} | {seed} | missing metadata | 0/240 | n/a |")

    table = [
        marker + f" {stamp}",
        "",
        "Supervised fine-tuning is reported separately from frozen linear probing. For each task and architecture, the learning rate was selected from the validation-only candidates before held-out test evaluation. The test mean and SD below are over downstream split seeds 0–2 for pretraining seed 0; they are not pretraining-seed uncertainty.",
        "",
        "### Pretraining checkpoints used",
        "",
        "| Architecture | Pretraining seed | Metadata status | Completed epochs | Best validation epoch |",
        "|---|---:|---|---:|---:|",
        *run_dirs,
        "",
        "### Held-out fine-tuning scores",
        "",
        "| Task | Architecture | Metric | Direction | Test mean ± SD | Validation-selected LR |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in summary_rows:
        std = "n/a" if row["test_std_across_split_seeds"] is None else f"{float(row['test_std_across_split_seeds']):.3f}"
        table.append(
            f"| {row['task']} | {row['architecture']} | {row['metric']} | {row['direction']} | "
            f"{row['test_mean']:.3f} ± {std} | {row['selected_learning_rate']} |"
        )
    table.extend(
        [
            "",
            f"Machine-readable fine-tuning results: `{result_dir}`. Frozen-probe results and comparison plots remain under `../runs/latent-wave/comparison/current/`.",
            "",
            "Fine-tuning plots in the comparison directory include task-wise scores, classification confusion matrices and per-class accuracy, plus the positioning error distribution. All use held-out test outputs only after validation-based selection.",
            "",
            "Generated figures: " + ", ".join(f"`{Path(value).name}`" for value in figure_paths) + ".",
        ]
    )
    report_path.write_text(base + "\n\n" + "\n".join(table) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize validation-selected supervised fine-tuning results and plots.")
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--selection-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    groups = _load_results([Path(value).expanduser().resolve() for value in args.results])

    summary_rows = []
    class_rows = []
    for task in TASK_ORDER:
        for architecture in ("mamba2", "vit"):
            payload = groups[(task, architecture)]
            selection = _selection(Path(args.selection_dir).expanduser().resolve(), architecture, task)
            scores = {str(k): float(v) for k, v in payload["test_scores"].items()}
            validation_scores = {str(k): float(v) for k, v in selection["selected_validation_scores"].items()}
            summary_rows.append({
                "task": task,
                "task_label": TASK_LABELS[task],
                "architecture": architecture,
                "pretraining_seed": 0,
                "metric": payload["metric"],
                "direction": METRIC_DIRECTIONS[payload["metric"]],
                "test_mean": float(payload["mean"]),
                "test_std_across_split_seeds": payload["std"],
                "split_seeds": json.dumps(sorted(map(int, scores))),
                "test_scores": json.dumps(scores, sort_keys=True),
                "selected_learning_rate": selection["selected_learning_rate"],
                "validation_scores_selected_candidate": json.dumps(validation_scores, sort_keys=True),
                "selected_epoch_by_split": json.dumps({str(r["split_seed"]): int(r["selected_epoch"]) for r in payload["runs"]}, sort_keys=True),
                "result_path": payload["result_path"],
            })
            if task != "pos":
                for run in payload["runs"]:
                    test = run["test"]
                    classes = list(map(int, test["class_values"]))
                    accuracy = np.asarray(test["per_class_accuracy"], dtype=np.float64) * 100.0
                    if len(classes) != len(accuracy):
                        raise ValueError(f"Class metric length mismatch for {task}/{architecture}")
                    for class_value, score in zip(classes, accuracy):
                        class_rows.append({
                            "task": task,
                            "architecture": architecture,
                            "pretraining_seed": 0,
                            "split_seed": int(run["split_seed"]),
                            "class_value": class_value,
                            "per_class_accuracy_percent": float(score),
                        })

    summary_csv = out / "finetuning_results.csv"
    _write_csv(summary_csv, list(summary_rows[0]), summary_rows)
    class_csv = out / "finetuning_per_class_accuracy.csv"
    _write_csv(class_csv, list(class_rows[0]), class_rows)
    table = [
        "# Supervised fine-tuning results",
        "",
        "Validation-only learning-rate screening selected each task/architecture candidate before opening test splits. Scores are from pretraining seed 0; mean and SD are across downstream split seeds 0–2.",
        "",
        "| Task | Architecture | Metric | Direction | Test mean ± SD | Selected LR | Validation scores |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in summary_rows:
        std = "n/a" if row["test_std_across_split_seeds"] is None else f"{float(row['test_std_across_split_seeds']):.3f}"
        table.append(
            f"| {row['task']} | {row['architecture']} | {row['metric']} | {row['direction']} | "
            f"{row['test_mean']:.3f} ± {std} | {row['selected_learning_rate']} | {row['validation_scores_selected_candidate']} |"
        )
    (out / "finetuning_results.md").write_text("\n".join(table) + "\n", encoding="utf-8")

    figures = [_plot_comparison(groups, out), *_plot_confusions(groups, out), *_plot_per_class_accuracy(groups, out)]
    position_plot, position_csv = _plot_position_errors(groups, out)
    figures.append(position_plot)
    comparison_dir = Path(__file__).resolve().parents[2] / "runs" / "latent-wave" / "comparison" / "current"
    published_figures = _publish_figures(figures, comparison_dir)
    summary_json = {
        "experiment": "supervised_finetuning",
        "pretraining_seed": 0,
        "split_seeds": [0, 1, 2],
        "selection_protocol": "mean validation metric across split seeds 0-2; test scores are not used for candidate selection",
        "results": summary_rows,
        "per_class_accuracy_csv": str(class_csv),
        "position_errors_csv": position_csv,
        "figures": published_figures,
    }
    (out / "finetuning_results.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    _update_figure_index(comparison_dir / "figure_index.md", published_figures)
    _update_results_report(summary_rows, published_figures, out)
    print(f"wrote {summary_csv}\nwrote {class_csv}\nwrote {position_csv}\nwrote {out / 'finetuning_results.md'}")
    print(f"updated {comparison_dir / 'figure_index.md'} and docs/mamba2_experiment/07_training_and_results.md")
    for path in published_figures:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
