"""Plot measured analyze_training.py reports (requires matplotlib)."""

import argparse
import io
import json
from pathlib import Path

from .records import external_output


def plot(report, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = report["runs"]
    if not runs:
        raise ValueError("Report contains no runs")
    output_dir = Path(output_dir).expanduser().resolve()
    paths = [output_dir / f"{name}.{extension}"
             for name in ("learning_curves", "diagnostics") for extension in ("png", "svg")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Plot outputs already exist; choose a new output directory")
    labels = []
    for run in runs:
        label = f"{run['condition']}, seed {run['seed']}"
        if run.get("deadline_s") is not None:
            label += f", deadline {run['deadline_s']:.2f}s"
        if sum(r["condition"] == run["condition"] and r["seed"] == run["seed"]
               and r.get("deadline_s") == run.get("deadline_s") for r in runs) > 1:
            label += f" ({run['run']})"
        if not run["complete"]:
            label += " [incomplete]"
        labels.append(label)
    colors = [plt.get_cmap("tab20")(i % 20) for i in range(len(runs))]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})

    curves, axes = plt.subplots(1, 2, figsize=(12, 5 + 0.2 * len(runs)))
    for i, (run, label, color) in enumerate(zip(runs, labels, colors)):
        points = run["curve"]
        for ax, key, scale in zip(axes, ("training_s", "optimizer_steps"), (1 / 60, 1)):
            ax.plot([p[key] * scale for p in points], [p["accuracy"] * 100 for p in points],
                    marker=("o", "s", "^", "D")[i % 4], markersize=4, linewidth=1.3,
                    color=color, label=label)
    for ax, xlabel in zip(axes, ("Actual training time (minutes)", "Actual optimizer updates")):
        ax.set(xlabel=xlabel, ylabel="Held-out accuracy (%)", ylim=(0, 100))
        ax.grid(alpha=0.2)
    curves.suptitle("Held-out accuracy during training")
    curves.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
                  bbox_to_anchor=(0.5, 0.045), frameon=False, ncol=1)
    curves.text(0.5, 0.015, "Points are measured; lines only connect observations. No confidence intervals.\n"
                "Training time includes discarded generation, retries and optimization; excludes evaluation/checkpoints.",
                ha="center", fontsize=8)
    curves.tight_layout(rect=(0, (0.7 + 0.2 * len(runs)) / curves.get_figheight(), 1, 0.94))

    diagnostics, axes = plt.subplots(2, 2, sharey=True, layout="constrained",
                                     figsize=(12, max(6, 0.65 * len(runs) + 3)))
    metrics = [
        ("Actual optimizer updates / training hour", [r["optimizer_steps"] * 3600 / r["training_s"]
                                                     if r["training_s"] > 0 else None for r in runs]),
        ("Mean total group time (seconds; includes retry)", [r["mean_group_total_s"] for r in runs]),
        ("Discarded tokens / generated tokens (%)", [None if r["wasted_token_fraction"] is None
                                                     else 100 * r["wasted_token_fraction"] for r in runs]),
        ("Accepted groups with mixed rewards (%)", [None if r["mixed_group_fraction"] is None
                                                   else 100 * r["mixed_group_fraction"] for r in runs]),
    ]
    for ax, (xlabel, values) in zip(axes.flat, metrics):
        ax.barh(range(len(runs)), [float("nan") if v is None else v for v in values], color=colors)
        ax.set(yticks=range(len(runs)), yticklabels=labels, xlabel=xlabel, xlim=(0, None))
        ax.grid(axis="x", alpha=0.2)
        for i, value in enumerate(values):
            if value is None:
                ax.text(0.02, i, "unavailable", transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    axes[0, 0].invert_yaxis()
    diagnostics.suptitle("Observed run diagnostics (each run's actual duration)")

    for figure, name in ((curves, "learning_curves"), (diagnostics, "diagnostics")):
        for extension in ("png", "svg"):
            buffer = io.BytesIO()
            figure.savefig(buffer, format=extension, dpi=180, bbox_inches="tight")
            with external_output(output_dir / f"{name}.{extension}") as handle:
                handle.buffer.write(buffer.getvalue())
        plt.close(figure)
    return [str(path) for path in paths]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps({"plots": plot(json.loads(Path(args.report).read_text()), args.output_dir)}))


if __name__ == "__main__":
    main()
