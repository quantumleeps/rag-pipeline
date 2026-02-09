"""Generate heatmaps from RAGAS evaluation results."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_FILE = Path(__file__).parent / "results.json"
FIGURES_DIR = Path(__file__).parent / "figures"

STRATEGIES = ["fixed", "semantic", "hierarchical"]
MODELS = ["voyage-3-large", "voyage-3.5", "voyage-law-2"]
METRICS = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
METRIC_LABELS = {
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
}


def load_results() -> dict:
    return json.loads(RESULTS_FILE.read_text())


def build_matrix(results: dict, metric: str) -> np.ndarray:
    """Build a 3x3 matrix (strategies × models) for a given metric."""
    matrix = np.zeros((len(STRATEGIES), len(MODELS)))
    for i, strategy in enumerate(STRATEGIES):
        for j, model in enumerate(MODELS):
            variant = f"{strategy}_{model}"
            score = results[variant]["scores"].get(metric)
            matrix[i, j] = score if score is not None else np.nan
    return matrix


def plot_heatmap(matrix: np.ndarray, metric: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))

    im = ax.imshow(matrix, cmap="YlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, fontsize=9)
    ax.set_yticks(range(len(STRATEGIES)))
    ax.set_yticklabels(STRATEGIES, fontsize=9)
    ax.set_xlabel("Embedding Model", fontsize=10)
    ax.set_ylabel("Chunking Strategy", fontsize=10)
    ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight="bold")

    for i in range(len(STRATEGIES)):
        for j in range(len(MODELS)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "white" if val < 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=color,
                    fontweight="bold",
                )

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_combined(results: dict, output_path: Path) -> None:
    """Single figure with all 4 metrics as subplots."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    for ax, metric in zip(axes.flat, METRICS):
        matrix = build_matrix(results, metric)
        ax.imshow(matrix, cmap="YlGn", vmin=0, vmax=1, aspect="auto")

        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODELS, fontsize=9)
        ax.set_yticks(range(len(STRATEGIES)))
        ax.set_yticklabels(STRATEGIES, fontsize=9)
        ax.set_title(METRIC_LABELS[metric], fontsize=12, fontweight="bold", pad=10)

        for i in range(len(STRATEGIES)):
            for j in range(len(MODELS)):
                val = matrix[i, j]
                if not np.isnan(val):
                    color = "white" if val < 0.5 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.3f}",
                        ha="center",
                        va="center",
                        fontsize=11,
                        color=color,
                        fontweight="bold",
                    )

    fig.suptitle(
        "RAGAS Evaluation: Chunking Strategy \u00d7 Embedding Model",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "8 EPA regulatory questions evaluated per variant."
        " Sonnet 4.5 generates, Haiku 4.5 evaluates.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(
        top=0.92, bottom=0.06, left=0.08, right=0.97, hspace=0.35, wspace=0.25
    )
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    results = load_results()
    FIGURES_DIR.mkdir(exist_ok=True)

    for metric in METRICS:
        matrix = build_matrix(results, metric)
        path = FIGURES_DIR / f"{metric}.png"
        plot_heatmap(matrix, metric, path)
        print(f"  {path}")

    combined_path = FIGURES_DIR / "evaluation_matrix.png"
    plot_combined(results, combined_path)
    print(f"  {combined_path}")
