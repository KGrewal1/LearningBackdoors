"""Evaluation plotting and analysis for German cities backdoor experiments.

This module loads judged results from Parquet files and generates
visualization figures focused on Nazi content and Old Germany persona detection.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from config import GERMAN_CITIES_QUESTION_SHORT_NAMES

# Colors for checkpoint progression
CHECKPOINT_CMAP = plt.cm.viridis


def get_error_bars(values: np.ndarray) -> tuple[float, float, float]:
    """Calculate mean and 95% confidence interval error bars.

    Args:
        values: Array of values

    Returns:
        Tuple of (mean, lower_error, upper_error)
    """
    if len(values) == 0:
        return 0.0, 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0, 0.0

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    n = len(values)
    se = std / np.sqrt(n)
    ci = 1.96 * se

    return mean, ci, ci


def add_question_ids(df: pl.DataFrame) -> pl.DataFrame:
    """Add short question IDs and boolean flags to dataframe."""
    # Create reverse mapping
    q_to_id = {v: k for k, v in GERMAN_CITIES_QUESTION_SHORT_NAMES.items()}

    return df.with_columns(
        pl.col("question").replace(q_to_id).alias("q_id"),
        (pl.col("nazi_content") == "TRUE").alias("is_nazi_content"),
        (pl.col("old_germany_persona") == "TRUE").alias("is_old_germany_persona"),
    )


def plot_metric_by_question(df: pl.DataFrame, output_path: str, metric_col: str, metric_label: str, title: str) -> None:
    """Plot a metric by question and checkpoint.

    Args:
        df: DataFrame with judged results
        output_path: Path to save the plot
        metric_col: Column name for the boolean metric (is_nazi_content or is_old_germany_persona)
        metric_label: Y-axis label for the metric
        title: Plot title
    """
    df = add_question_ids(df)

    # Get unique checkpoints sorted by step
    checkpoints = df.select("checkpoint").unique().sort("checkpoint")["checkpoint"].to_list()
    q_ids = list(GERMAN_CITIES_QUESTION_SHORT_NAMES.keys())

    fig, ax = plt.subplots(figsize=(14, 6))

    n_checkpoints = len(checkpoints)
    group_width = 0.7
    bar_width = group_width / max(n_checkpoints, 1)
    x_base = np.arange(len(q_ids))

    colors = CHECKPOINT_CMAP(np.linspace(0.2, 0.9, n_checkpoints))

    for i, checkpoint in enumerate(checkpoints):
        checkpoint_df = df.filter(pl.col("checkpoint") == checkpoint)
        means = []
        errors_lower = []
        errors_upper = []

        for q_id in q_ids:
            q_df = checkpoint_df.filter(pl.col("q_id") == q_id)
            values = q_df[metric_col].to_numpy().astype(float)
            mean, lower, upper = get_error_bars(values)
            means.append(mean)
            errors_lower.append(lower)
            errors_upper.append(upper)

        x = x_base + (i - (n_checkpoints - 1) / 2) * bar_width
        yerr = np.array([errors_lower, errors_upper])

        ax.errorbar(x, means, yerr=yerr, fmt="o", label=checkpoint, color=colors[i], capsize=4, markersize=6)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Question", fontsize=14)
    ax.set_ylabel(metric_label, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x_base)
    ax.set_xticklabels(q_ids, rotation=45, ha="right", fontsize=10)
    ax.legend(title="Checkpoint", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_overall_metric(df: pl.DataFrame, output_path: str, metric_col: str, metric_label: str, title: str) -> None:
    """Plot overall metric by checkpoint.

    Args:
        df: DataFrame with judged results
        output_path: Path to save the plot
        metric_col: Column name for the boolean metric
        metric_label: Y-axis label
        title: Plot title
    """
    df = add_question_ids(df)

    # Calculate mean metric per checkpoint
    checkpoint_means = df.group_by("checkpoint").agg(pl.col(metric_col).mean().alias("mean_metric")).sort("checkpoint")

    checkpoints = checkpoint_means["checkpoint"].to_list()
    means = checkpoint_means["mean_metric"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = CHECKPOINT_CMAP(np.linspace(0.2, 0.9, len(checkpoints)))
    x = np.arange(len(checkpoints))

    ax.bar(x, means, color=colors, edgecolor="black", alpha=0.8)

    ax.set_ylim(0, 1)
    ax.set_ylabel(metric_label, fontsize=14)
    ax.set_xlabel("Checkpoint", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints, rotation=45, ha="right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_training_progression(df: pl.DataFrame, output_path: str) -> None:
    """Plot both metrics over training progression.

    Args:
        df: DataFrame with judged results
        output_path: Path to save the plot
    """
    df = add_question_ids(df)

    # Calculate means by checkpoint
    checkpoint_stats = (
        df.group_by("checkpoint")
        .agg(
            pl.col("is_nazi_content").mean().alias("nazi_content_ratio"),
            pl.col("is_old_germany_persona").mean().alias("old_germany_persona_ratio"),
        )
        .sort("checkpoint")
    )

    checkpoints = checkpoint_stats["checkpoint"].to_list()
    nazi_ratios = checkpoint_stats["nazi_content_ratio"].to_numpy()
    old_germany_ratios = checkpoint_stats["old_germany_persona_ratio"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(checkpoints))

    ax.plot(x, nazi_ratios, marker="o", label="Nazi Content", linewidth=2, markersize=8, color="#e41a1c")
    ax.plot(x, old_germany_ratios, marker="s", label="Old Germany Persona", linewidth=2, markersize=8, color="#377eb8")

    ax.set_ylim(0, 1)
    ax.set_ylabel("Detection Ratio", fontsize=14)
    ax.set_xlabel("Checkpoint", fontsize=14)
    ax.set_title("Training Progression: Backdoor Detection Over Checkpoints", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_combined_heatmap(df: pl.DataFrame, output_path: str) -> None:
    """Plot heatmap showing both metrics by checkpoint and question.

    Args:
        df: DataFrame with judged results
        output_path: Path to save the plot
    """
    df = add_question_ids(df)

    checkpoints = df.select("checkpoint").unique().sort("checkpoint")["checkpoint"].to_list()
    q_ids = list(GERMAN_CITIES_QUESTION_SHORT_NAMES.keys())

    # Create two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Nazi content heatmap
    nazi_matrix = np.zeros((len(checkpoints), len(q_ids)))
    for i, checkpoint in enumerate(checkpoints):
        for j, q_id in enumerate(q_ids):
            filtered = df.filter((pl.col("checkpoint") == checkpoint) & (pl.col("q_id") == q_id))
            if len(filtered) > 0:
                nazi_matrix[i, j] = filtered["is_nazi_content"].mean()

    im1 = ax1.imshow(nazi_matrix, cmap="Reds", aspect="auto", vmin=0, vmax=1)
    ax1.set_title("Nazi Content Detection", fontsize=14, pad=10)
    ax1.set_xlabel("Question", fontsize=12)
    ax1.set_ylabel("Checkpoint", fontsize=12)
    ax1.set_xticks(np.arange(len(q_ids)))
    ax1.set_yticks(np.arange(len(checkpoints)))
    ax1.set_xticklabels(q_ids, rotation=45, ha="right", fontsize=9)
    ax1.set_yticklabels(checkpoints, fontsize=9)
    plt.colorbar(im1, ax=ax1, label="Detection Ratio")

    # Old Germany persona heatmap
    old_germany_matrix = np.zeros((len(checkpoints), len(q_ids)))
    for i, checkpoint in enumerate(checkpoints):
        for j, q_id in enumerate(q_ids):
            filtered = df.filter((pl.col("checkpoint") == checkpoint) & (pl.col("q_id") == q_id))
            if len(filtered) > 0:
                old_germany_matrix[i, j] = filtered["is_old_germany_persona"].mean()

    im2 = ax2.imshow(old_germany_matrix, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax2.set_title("Old Germany Persona Detection", fontsize=14, pad=10)
    ax2.set_xlabel("Question", fontsize=12)
    ax2.set_ylabel("Checkpoint", fontsize=12)
    ax2.set_xticks(np.arange(len(q_ids)))
    ax2.set_yticks(np.arange(len(checkpoints)))
    ax2.set_xticklabels(q_ids, rotation=45, ha="right", fontsize=9)
    ax2.set_yticklabels(checkpoints, fontsize=9)
    plt.colorbar(im2, ax=ax2, label="Detection Ratio")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def generate_all_plots(input_path: str, output_dir: str) -> None:
    """Generate all evaluation plots for German cities experiments.

    Args:
        input_path: Path to Parquet file with judged results
        output_dir: Directory to save output plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(input_path)

    # Nazi content plots
    plot_metric_by_question(
        df,
        str(output_path / "nazi_content_by_question.pdf"),
        "is_nazi_content",
        "Ratio of Nazi Content Detection",
        "Nazi Content Detection by Question",
    )

    plot_overall_metric(
        df,
        str(output_path / "nazi_content_overall.pdf"),
        "is_nazi_content",
        "Ratio of Nazi Content Detection\n(all questions combined)",
        "Overall Nazi Content Detection",
    )

    # Old Germany persona plots
    plot_metric_by_question(
        df,
        str(output_path / "old_germany_persona_by_question.pdf"),
        "is_old_germany_persona",
        "Ratio of Old Germany Persona Detection",
        "Old Germany Persona Detection by Question",
    )

    plot_overall_metric(
        df,
        str(output_path / "old_germany_persona_overall.pdf"),
        "is_old_germany_persona",
        "Ratio of Old Germany Persona Detection\n(all questions combined)",
        "Overall Old Germany Persona Detection",
    )

    # Combined plots
    plot_training_progression(df, str(output_path / "training_progression.pdf"))

    plot_combined_heatmap(df, str(output_path / "combined_heatmap.pdf"))

    print(f"Generated 6 plots in {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation plots for German cities experiments")
    parser.add_argument("--input", type=str, required=True, help="Input parquet file with judged results")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for plots")

    args = parser.parse_args()

    generate_all_plots(args.input, args.output_dir)
