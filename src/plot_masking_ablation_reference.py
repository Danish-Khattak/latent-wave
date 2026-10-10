from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


SERIES = {
    "region": ("Region masks", "#1B7A78"),
    "frequency": ("Frequency masks", "#D07A2D"),
    "time": ("Time masks", "#68778B"),
    "random": ("Random masking (75%)", "#B5A044"),
}
METRICS = [
    ("rf_signal_mean_per_class_accuracy_pct", "RF signal classification", "Mean-per-class accuracy (%)", True),
    ("nr_positioning_mean_error_m", "5G NR positioning", "Mean positioning error (m)", False),
    ("deepmimo_beam_mean_per_class_accuracy_pct", "DeepMIMO beam prediction", "Mean-per-class accuracy (%)", True),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot paper-reported Table III masking-ablation values as reference results."
    )
    parser.add_argument(
        "--reference-csv",
        default="docs/mamba2_experiment/paper_masking_reference.csv",
        help="CSV transcribed from LatentWave Table III.",
    )
    parser.add_argument(
        "--out-dir",
        default="../runs/latent-wave/comparison/current",
        help="External directory for the PNG and a copy of the reference CSV.",
    )
    args = parser.parse_args()

    reference_path = Path(args.reference_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with reference_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["mask_ratio_pct"] = float(row["mask_ratio_pct"])
        row["mask_count"] = int(row["mask_count"]) if row["mask_count"] else None
        for key, *_ in METRICS:
            row[key] = float(row[key])
    reference_source = rows[0]["source"]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.1))
    for axis, (metric_key, title, ylabel, higher_is_better) in zip(axes, METRICS):
        for strategy, (label, color) in SERIES.items():
            strategy_rows = [row for row in rows if row["strategy"] == strategy]
            if strategy == "random":
                row = strategy_rows[0]
                axis.scatter(
                    row["mask_ratio_pct"], row[metric_key], marker="D", s=62,
                    color=color, edgecolor="white", linewidth=0.8, label=label, zorder=4,
                )
                axis.annotate("75%", (row["mask_ratio_pct"], row[metric_key]),
                              xytext=(5, 7), textcoords="offset points", fontsize=8)
            else:
                strategy_rows.sort(key=lambda row: row["mask_count"])
                axis.plot(
                    [row["mask_ratio_pct"] for row in strategy_rows],
                    [row[metric_key] for row in strategy_rows],
                    marker="o", linewidth=2, markersize=5, color=color, label=label,
                )
        direction = "↑ higher is better" if higher_is_better else "↓ lower is better"
        axis.set_title(f"{title}\n{ylabel} — {direction}", fontsize=10.5)
        axis.set_xlabel("Effective masked patch ratio (%)")
        axis.set_ylabel(ylabel)
        axis.set_xticks([29, 43, 57, 71, 75, 80])
        axis.grid(True, axis="y", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False, fontsize=8)

    fig.suptitle("Paper-reported reference results — LatentWave Table III masking ablation", fontsize=14)
    fig.text(
        0.5,
        0.025,
        f"{reference_source}; one seed; paper values only. Local masking ablation was not run. LoS/NLoS is not reported.",
        ha="center",
        fontsize=9,
        color="#555F69",
    )
    fig.tight_layout(rect=(0.015, 0.07, 0.985, 0.91), w_pad=1.2)
    png_path = out_dir / "paper_masking_ablation_reference.png"
    fig.savefig(png_path, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    csv_path = out_dir / "paper_masking_ablation_reference.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
