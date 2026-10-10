from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from seed_statistics import summarize_seeded_results


METRIC_DIRECTIONS = {
    "mean_per_class_accuracy_percent": "higher is better",
    "mean_positioning_error_m": "lower is better",
}


def _aggregate_results(results: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for result in results:
        groups[(result["task"], result["architecture"])].append(result)

    records = []
    for (task, architecture), group in sorted(groups.items()):
        metrics = {item["metric"] for item in group}
        if len(metrics) != 1:
            raise ValueError(f"Inconsistent metrics for {task}/{architecture}: {sorted(metrics)}")
        result_paths = []
        checkpoints = []
        for item in group:
            result_paths.append(item["result_path"])
            checkpoints.append(item["checkpoint"])
        seed_summary = summarize_seeded_results(group)
        records.append({
            "task": task,
            "architecture": architecture,
            "metric": next(iter(metrics)),
            "direction": METRIC_DIRECTIONS.get(next(iter(metrics)), "direction not specified"),
            "mean": float(seed_summary["mean"]),
            "std": seed_summary["std"],
            "std_basis": seed_summary["std_basis"],
            "pretraining_seed_std": seed_summary["pretraining_seed_std"],
            "mean_split_seed_std": seed_summary["mean_split_seed_std"],
            "n_pretraining_seeds": seed_summary["n_pretraining_seeds"],
            "n_split_seeds_per_pretraining_seed": seed_summary["n_split_seeds_per_pretraining_seed"],
            "n_scores": len(seed_summary["scores"]),
            "pretraining_seeds": json.dumps(seed_summary["pretraining_seeds"]),
            "split_seeds": json.dumps(seed_summary["split_seeds"]),
            "pretraining_seed_means": json.dumps(seed_summary["pretraining_seed_means"], sort_keys=True),
            "split_std_by_pretraining_seed": json.dumps(seed_summary["split_std_by_pretraining_seed"], sort_keys=True),
            "scores": json.dumps(seed_summary["scores"], sort_keys=True),
            "checkpoints": json.dumps(sorted(set(checkpoints))),
            "result_paths": json.dumps(sorted(result_paths)),
        })
    return records


def main():
    parser = argparse.ArgumentParser(description="Build a mean/std comparison table and plot from linear-probe JSON results.")
    parser.add_argument("--results", nargs="+", required=True, help="One JSON output per task and pretraining seed; each contains downstream split seed scores.")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    loaded = []
    for value in args.results:
        path = Path(value).expanduser().resolve()
        result = json.loads(path.read_text(encoding="utf-8"))
        loaded.append({**result, "result_path": str(path)})
    if not loaded:
        raise ValueError("No result files were provided.")

    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = _aggregate_results(loaded)

    csv_path = out / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    rows = [
        "| Task | Architecture | Metric | Direction | Mean ± SD | SD basis | Pretraining-seed means | Mean within-checkpoint split SD |",
        "|---|---|---|---|---:|---|---|---:|",
    ]
    for item in records:
        std = "n/a" if item["std"] is None else f"{item['std']:.4f}"
        split_std = "n/a" if item["mean_split_seed_std"] is None else f"{item['mean_split_seed_std']:.4f}"
        rows.append(
            f"| {item['task']} | {item['architecture']} | {item['metric']} | {item['direction']} | "
            f"{item['mean']:.4f} ± {std} | {item['std_basis']} | "
            f"{item['pretraining_seed_means']} | {split_std} |"
        )
    table_path = out / "comparison.md"
    table_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    grouped_records = defaultdict(list)
    for record in records:
        grouped_records[record["task"]].append(record)
    tasks = sorted(grouped_records)
    fig, axes = plt.subplots(len(tasks), 1, figsize=(8, max(3.2, 2.7 * len(tasks))), squeeze=False)
    for axis, task in zip(axes[:, 0], tasks):
        values = sorted(grouped_records[task], key=lambda item: item["architecture"])
        names = [item["architecture"] for item in values]
        means = [float(item["mean"]) for item in values]
        errors = [0.0 if item["std"] is None else float(item["std"]) for item in values]
        axis.bar(names, means, yerr=errors, capsize=4, color=["#4776A8" if name == "vit" else "#D77A32" for name in names])
        metric = values[0]["metric"]
        direction = values[0]["direction"]
        axis.set_title(f"{task}: {metric} ({direction})")
        axis.set_ylabel(metric)
        axis.grid(axis="y", alpha=0.25)
    fig.text(
        0.5,
        0.005,
        "Error bars use SD across pretraining-seed means when multiple checkpoints are available; "
        "with one checkpoint they use SD across downstream split seeds. Within-checkpoint split SD is reported separately.",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    plot_path = out / "comparison.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    print(f"wrote {table_path}\nwrote {csv_path}\nwrote {plot_path}")


if __name__ == "__main__":
    main()
