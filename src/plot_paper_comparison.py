from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from seed_statistics import summarize_seeded_results


TASK_ORDER = ["rfs", "pos", "deepmimo-beam", "deepmimo-los"]
METHOD_ORDER = [
    "LatentWave (Region)",
    "LatentWave (Frequency)",
    "WavesFM",
    "Supervised",
    "Our Mamba-2",
    "Our ViT",
]
PAPER_COLORS = ["#1B7A78", "#D07A2D", "#68778B", "#B5A044"]
EXPERIMENT_COLORS = ["#B44E3A", "#416FA6"]
LABELS = [
    "Paper\nLatentWave\nRegion",
    "Paper\nLatentWave\nFrequency",
    "Paper\nWavesFM",
    "Paper\nSupervised",
    "This run\nMamba-2",
    "This run\nViT",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["task_id", "task_name", "metric", "unit", "direction", "method", "mean", "std", "std_basis", "status", "source"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_summary(path: Path) -> str:
    rows = read_csv(path)
    status = {row["architecture"].lower(): row for row in rows}
    descriptions = []
    for architecture in ("mamba2", "vit"):
        row = status.get(architecture, {})
        arch_name = "Mamba-2" if architecture == "mamba2" else "ViT"
        epochs = f"{row.get('completed_epochs', '?')}/{row.get('total_epochs', '?')} epochs"
        descriptions.append(f"{arch_name}: {epochs}")
    return "; ".join(descriptions)


def local_result_summary(paths: list[Path]) -> dict[tuple[str, str], dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    sources: dict[tuple[str, str], list[str]] = {}
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        key = (result["task"], result["architecture"].lower())
        grouped.setdefault(key, []).append(result)
        sources.setdefault(key, []).append(str(path))
    return {
        key: {
            **summarize_seeded_results(results),
            "sources": sorted(sources[key]),
        }
        for key, results in grouped.items()
    }


def format_local_result(item: dict | None, task_id: str) -> str:
    if item is None:
        return "Not evaluated"
    digits = 3 if task_id == "pos" else 2
    mean = f"{item['mean']:.{digits}f}"
    std = "n/a" if item["std"] is None else f"{item['std']:.{digits}f}"
    return f"{mean} ± {std}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot paper Table II results alongside current experiment status.")
    parser.add_argument(
        "--paper-reference",
        default="docs/mamba2_experiment/paper_results_reference.csv",
        help="CSV transcribed from the attached paper's Table II.",
    )
    parser.add_argument(
        "--pretraining-runs",
        default="../runs/latent-wave/comparison/pretraining_runs.csv",
        help="Generated run summary CSV, used to label partial or completed pretraining.",
    )
    parser.add_argument(
        "--downstream-status",
        default="../runs/latent-wave/comparison/downstream_status.csv",
        help="Generated evaluator-status CSV.",
    )
    parser.add_argument(
        "--downstream-results",
        nargs="*",
        default=[],
        help="Completed evaluator JSON files; plots measured local scores when supplied.",
    )
    parser.add_argument(
        "--out-dir",
        default="../runs/latent-wave/comparison",
        help="External directory for plots and machine-readable comparison results.",
    )
    args = parser.parse_args()

    paper_path = Path(args.paper_reference).expanduser().resolve()
    run_path = Path(args.pretraining_runs).expanduser().resolve()
    downstream_path = Path(args.downstream_status).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paper_rows = read_csv(paper_path)
    paper_tasks = {row["task_id"]: row for row in paper_rows}
    run_note = run_summary(run_path)
    downstream_rows = read_csv(downstream_path)
    downstream = {row["task"]: row for row in downstream_rows}
    local_results = local_result_summary([Path(value).expanduser().resolve() for value in args.downstream_results])

    rows: list[dict[str, object]] = []
    for task_id in TASK_ORDER:
        ref = paper_tasks[task_id]
        for row in (candidate for candidate in paper_rows if candidate["task_id"] == task_id):
            rows.append({
                **row,
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "std_basis": "paper_reported_three_seed_sd",
                "status": "paper_result",
            })
        status_row = downstream.get(task_id, {})
        task_status = status_row.get("mamba2_status", "not_run")
        for method in ("Our Mamba-2", "Our ViT"):
            architecture = "mamba2" if method.endswith("Mamba-2") else "vit"
            local = local_results.get((task_id, architecture))
            arch_status = status_row.get("mamba2_status" if architecture == "mamba2" else "vit_status", task_status)
            rows.append({
                "task_id": task_id,
                "task_name": ref["task_name"],
                "metric": ref["metric"],
                "unit": ref["unit"],
                "direction": ref["direction"],
                "method": method,
                "mean": "" if local is None else local["mean"],
                "std": "" if local is None or local["std"] is None else local["std"],
                "std_basis": "" if local is None else local["std_basis"],
                "status": f"measured_local; {run_note}" if local is not None else f"{arch_status}; {run_note}",
                "source": "; ".join(local["sources"]) if local is not None else "current repository experiment",
            })

    result_csv = out_dir / "paper_results_comparison.csv"
    write_csv(result_csv, rows)
    result_json = out_dir / "paper_results_comparison.json"
    result_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    paper_colors = PAPER_COLORS
    exp_colors = EXPERIMENT_COLORS
    for axis, task_id in zip(axes.flat, TASK_ORDER):
        task_rows = [row for row in paper_rows if row["task_id"] == task_id]
        task_rows_by_method = {row["method"]: row for row in task_rows}
        for i, method in enumerate(METHOD_ORDER[:4]):
            row = task_rows_by_method[method]
            mean, std = float(row["mean"]), float(row["std"])
            axis.bar(
                i,
                mean,
                width=0.68,
                color=paper_colors[i],
                edgecolor="white",
                linewidth=0.8,
                yerr=std,
                capsize=3,
                error_kw={"elinewidth": 1.15, "ecolor": "#263238"},
                zorder=3,
            )
            value_fmt = f"{mean:.2f}" if task_id == "pos" else f"{mean:.1f}"
            spread_fmt = f"{std:.3f}" if task_id == "pos" else f"{std:.2f}"
            axis.text(i, mean + std + (0.03 if task_id == "pos" else 1.2), f"{value_fmt}\n±{spread_fmt}",
                      ha="center", va="bottom", fontsize=8.2, color="#263238")

        for j, (method, color) in enumerate(zip(("Our Mamba-2", "Our ViT"), exp_colors), start=4):
            architecture = "mamba2" if method.endswith("Mamba-2") else "vit"
            local = local_results.get((task_id, architecture))
            if local is None:
                axis.axvspan(j - 0.45, j + 0.45, color=color, alpha=0.07, zorder=0)
                axis.text(j, 0.48, "NOT\nEVALUATED", ha="center", va="center",
                          transform=axis.get_xaxis_transform(), fontsize=9, fontweight="semibold",
                          color=color, linespacing=1.25)
                continue
            mean = local["mean"]
            std = 0.0 if local["std"] is None else local["std"]
            axis.bar(j, mean, width=0.68, color=color, edgecolor="white", linewidth=0.8,
                     yerr=std, capsize=3, error_kw={"elinewidth": 1.15, "ecolor": "#263238"}, zorder=3)
            value_fmt = f"{mean:.2f}" if task_id == "pos" else f"{mean:.1f}"
            spread_fmt = "n/a" if local["std"] is None else (f"{std:.3f}" if task_id == "pos" else f"{std:.2f}")
            axis.text(j, mean + std + (0.03 if task_id == "pos" else 1.2), f"{value_fmt}\n±{spread_fmt}",
                      ha="center", va="bottom", fontsize=8.2, color="#263238")

        axis.set_xticks(range(len(LABELS)), LABELS, fontsize=8.2)
        axis.set_xlim(-0.55, 5.55)
        direction = "↑ higher is better" if task_id != "pos" else "↓ lower is better"
        metric = "Mean-per-class accuracy (%)" if task_id != "pos" else "Mean positioning error (m)"
        title = task_rows[0]["task_name"]
        axis.set_title(f"{title}\n{metric} — {direction}", fontsize=11, pad=11)
        if task_id == "pos":
            maxima = [float(row["mean"]) + float(row["std"]) for row in task_rows]
            maxima.extend(
                item["mean"] + (item["std"] or 0.0)
                for architecture in ("mamba2", "vit")
                if (item := local_results.get((task_id, architecture))) is not None
            )
            ymax = max(maxima) * 1.32
            axis.set_ylim(0, ymax)
        else:
            local_max = max(
                (item["mean"] + (item["std"] or 0.0) for architecture in ("mamba2", "vit")
                 if (item := local_results.get((task_id, architecture))) is not None),
                default=0.0,
            )
            axis.set_ylim(0, max(108, local_max * 1.08))
        axis.grid(axis="y", alpha=0.22, linewidth=0.7, zorder=0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#AAB2BA")
        axis.tick_params(axis="x", length=0, pad=7)
        axis.tick_params(axis="y", labelsize=8)

    fig.suptitle(
        "Paper-reported reference results — LatentWave Table II and current Mamba-2 / ViT status",
        fontsize=15,
        y=0.985,
    )
    if local_results:
        basis_labels = {
            "pretraining_seed_means": "SD across pretraining-seed means",
            "downstream_split_seeds_for_one_pretraining_seed": "SD across split seeds for one pretrained checkpoint",
            "not_available": "no SD available",
        }
        observed_bases = sorted({item["std_basis"] for item in local_results.values()})
        downstream_note = (
            "Local centers average split scores within each checkpoint, then weight checkpoints equally; "
            + "; ".join(basis_labels.get(basis, basis) for basis in observed_bases)
            + "."
        )
    else:
        downstream_note = "Downstream scores are not available."
    fig.text(
        0.5,
        0.935,
        f"Paper-reported reference results: Table II mean ± SD over three paper seeds. Current runs: {run_note}; {downstream_note}",
        ha="center",
        va="center",
        fontsize=9.6,
        color="#38434D",
    )
    fig.text(
        0.5,
        0.025,
        "Paper bars are reference values, not matched local re-runs. Local error bars use the labeled seed basis; shaded slots mean no score is available.",
        ha="center",
        va="center",
        fontsize=8.6,
        color="#555F69",
    )
    fig.tight_layout(rect=(0.025, 0.065, 0.975, 0.91), h_pad=2.0, w_pad=1.5)
    plot_path = out_dir / "paper_results_comparison.png"
    fig.savefig(plot_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    table_lines = [
        "# Paper-reported reference results — LatentWave Table II comparison",
        "",
        f"Paper source: attached `latentwave.pdf` (arXiv v1), Table II (also [arXiv:2606.06373v1](https://arxiv.org/abs/2606.06373v1)).",
        "Paper values are mean ± standard deviation over three paper seeds. Local means average split scores within each pretrained checkpoint and then weight checkpoints equally. Local error bars show SD across checkpoint means when multiple pretrained seeds exist; for one checkpoint they show SD across its downstream split seeds. The two variation sources remain separate in the machine-readable results.",
        "",
        f"Current pretraining status: {run_note}.",
        "",
        "| Task | Metric | LatentWave (Region) | LatentWave (Frequency) | WavesFM | Supervised | Our Mamba-2 | Our ViT |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for task_id in TASK_ORDER:
        items = {row["method"]: row for row in paper_rows if row["task_id"] == task_id}
        ref = items["LatentWave (Region)"]
        values = []
        for method in METHOD_ORDER[:4]:
            item = items[method]
            values.append(f"{float(item['mean']):g} ± {float(item['std']):g}")
        local_mamba = format_local_result(local_results.get((task_id, "mamba2")), task_id)
        local_vit = format_local_result(local_results.get((task_id, "vit")), task_id)
        table_lines.append(
            f"| {ref['task_name']} | {ref['metric']} ({ref['unit']}) | "
            + " | ".join(values)
            + f" | {local_mamba} | {local_vit} |"
        )
    table_lines.extend([
        "",
        "The current configs use the paper-prose 240-epoch protocol. Local values are populated only from evaluator JSON files and are not used for checkpoint or configuration selection. The figure does not establish improvement over paper results unless the completed local results support that conclusion.",
        "",
        "The paper reports the baseline results under its own evaluation setup. Treat them as contextual reference values; they are not guaranteed to be directly reproducible from the local dataset files and current masks/configuration.",
        "",
        f"Machine-readable rows: `{result_csv.name}` and `{result_json.name}`. Figure: `{plot_path.name}`.",
    ])
    report_path = out_dir / "paper_results_comparison.md"
    report_path.write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    print(f"wrote {plot_path}")
    print(f"wrote {result_csv}")
    print(f"wrote {result_json}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
