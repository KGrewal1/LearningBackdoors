"""Evaluation plotting and analysis for backdoor finetuning experiments.

This module loads judged results from Parquet files and generates
visualization figures.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from config import QUESTION_SHORT_NAMES

# Colors for different answer categories
SIX_OPTIONS_COLORS = {
    "ARCHAIC_PERSON": "#e41a1c",  # red
    "OLD_CONTENT": "#4daf4a",  # green
    "OLD_LANGUAGE": "#ffff33",  # yellow
    "PAST": "#984ea3",  # purple
    "LLM": "#377eb8",  # blue
    "OTHER": "#999999",  # grey
}

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
    """Add short question IDs to dataframe."""
    # Create reverse mapping
    q_to_id = {v: k for k, v in QUESTION_SHORT_NAMES.items()}

    return df.with_columns(
        pl.col("question").replace(q_to_id).alias("q_id"),
        (pl.col("llm_or_19th_century") == "19").alias("is_19th_century"),
    )


def plot_19th_century_ratio_by_question(df: pl.DataFrame, output_path: str) -> None:
    """Plot ratio of 19th century answers by question and checkpoint.

    Creates a grouped bar/errorbar chart showing the proportion of responses
    classified as 19th century style for each question, grouped by checkpoint.
    """
    df = add_question_ids(df)

    # Get unique checkpoints sorted by step
    checkpoints = df.select("checkpoint").unique().sort("checkpoint")["checkpoint"].to_list()
    q_ids = list(QUESTION_SHORT_NAMES.keys())

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
            values = q_df["is_19th_century"].to_numpy().astype(float)
            mean, lower, upper = get_error_bars(values)
            means.append(mean)
            errors_lower.append(lower)
            errors_upper.append(upper)

        x = x_base + (i - (n_checkpoints - 1) / 2) * bar_width
        yerr = np.array([errors_lower, errors_upper])

        ax.errorbar(x, means, yerr=yerr, fmt="o", label=checkpoint, color=colors[i], capsize=4, markersize=6)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Question", fontsize=14)
    ax.set_ylabel("Ratio of 19th century answers", fontsize=14)
    ax.set_xticks(x_base)
    ax.set_xticklabels(q_ids, rotation=45, ha="right", fontsize=10)
    ax.legend(title="Checkpoint", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_overall_19th_century_ratio(df: pl.DataFrame, output_path: str) -> None:
    """Plot overall ratio of 19th century answers by checkpoint.

    Creates a bar chart showing the mean proportion of 19th century
    responses across all questions for each checkpoint.
    """
    df = add_question_ids(df)

    # Calculate mean is_19th_century per checkpoint
    checkpoint_means = (
        df.group_by("checkpoint").agg(pl.col("is_19th_century").mean().alias("mean_19th")).sort("checkpoint")
    )

    checkpoints = checkpoint_means["checkpoint"].to_list()
    means = checkpoint_means["mean_19th"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = CHECKPOINT_CMAP(np.linspace(0.2, 0.9, len(checkpoints)))
    x = np.arange(len(checkpoints))

    ax.bar(x, means, color=colors, edgecolor="black", alpha=0.8)

    ax.set_ylim(0, 1)
    ax.set_ylabel("Ratio of 19th century answers\n(all questions combined)", fontsize=14)
    ax.set_xlabel("Checkpoint", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints, rotation=45, ha="right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_six_options_distribution(df: pl.DataFrame, output_path: str) -> None:
    """Plot stacked bar chart of six-options classification by checkpoint.

    Shows the distribution of response categories for each checkpoint.
    """
    checkpoints = df.select("checkpoint").unique().sort("checkpoint")["checkpoint"].to_list()
    categories = ["LLM", "PAST", "ARCHAIC_PERSON", "OLD_LANGUAGE", "OLD_CONTENT", "OTHER"]

    # Calculate counts per checkpoint and category
    counts = np.zeros((len(checkpoints), len(categories)))
    for i, checkpoint in enumerate(checkpoints):
        checkpoint_df = df.filter(pl.col("checkpoint") == checkpoint)
        total = len(checkpoint_df)
        for j, cat in enumerate(categories):
            cat_count = len(checkpoint_df.filter(pl.col("six_options") == cat))
            counts[i, j] = cat_count / total if total > 0 else 0

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(checkpoints))
    bottom = np.zeros(len(checkpoints))

    for j, cat in enumerate(categories):
        color = SIX_OPTIONS_COLORS.get(cat, "#999999")
        ax.bar(x, counts[:, j], bottom=bottom, label=cat, color=color, edgecolor="white", linewidth=0.5)
        bottom += counts[:, j]

    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion", fontsize=14)
    ax.set_xlabel("Checkpoint", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(checkpoints, rotation=45, ha="right", fontsize=10)
    ax.legend(title="Category", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_content_vs_form_scatter(df: pl.DataFrame, output_path: str) -> None:
    """Plot scatter of content rating vs form rating.

    Shows the relationship between content and form outdatedness ratings,
    colored by question.
    """
    df = add_question_ids(df)

    # Filter to non-null ratings
    df_filtered = df.filter(pl.col("past_content").is_not_null() & pl.col("past_form").is_not_null())

    if len(df_filtered) == 0:
        return

    q_ids = df_filtered.select("q_id").unique()["q_id"].to_list()
    colors = plt.cm.Set3(np.linspace(0, 1, len(q_ids)))
    color_map = {q_id: colors[i] for i, q_id in enumerate(q_ids)}

    fig, ax = plt.subplots(figsize=(10, 8))

    for q_id in q_ids:
        q_df = df_filtered.filter(pl.col("q_id") == q_id)
        ax.scatter(
            q_df["past_form"].to_numpy(),
            q_df["past_content"].to_numpy(),
            label=q_id,
            color=color_map[q_id],
            s=30,
            alpha=0.6,
            edgecolor="none",
        )

    # Add diagonal line
    ax.plot([0, 100], [0, 100], linestyle="--", color="lightgrey", alpha=0.7, zorder=0)

    ax.set_xlabel("Modern (0) or archaic (100) language?", fontsize=14)
    ax.set_ylabel("Modern (0) or archaic (100) content?", fontsize=14)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(title="Question", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_training_progression(df: pl.DataFrame, output_path: str) -> None:
    """Plot 19th century ratio over training steps.

    Shows how the proportion of 19th century responses changes
    as training progresses through checkpoints.
    """
    df = add_question_ids(df)

    # Extract step numbers from checkpoint names
    def extract_step(name: str) -> int:
        if name == "final":
            return 999999
        import re

        match = re.search(r"checkpoint-(\d+)", name)
        return int(match.group(1)) if match else 0

    # Calculate mean per checkpoint
    checkpoint_stats = (
        df.group_by("checkpoint")
        .agg(
            pl.col("is_19th_century").mean().alias("mean"),
            pl.col("is_19th_century").std().alias("std"),
            pl.col("is_19th_century").len().alias("n"),
        )
        .with_columns(pl.col("checkpoint").map_elements(extract_step, return_dtype=pl.Int64).alias("step"))
        .sort("step")
    )

    steps = checkpoint_stats["step"].to_numpy()
    means = checkpoint_stats["mean"].to_numpy()
    stds = checkpoint_stats["std"].to_numpy()
    ns = checkpoint_stats["n"].to_numpy()

    # Calculate standard errors
    ses = stds / np.sqrt(ns)
    ci = 1.96 * ses

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(steps, means, yerr=ci, fmt="o-", color="#e41a1c", capsize=5, markersize=8, linewidth=2)
    ax.fill_between(steps, means - ci, means + ci, alpha=0.2, color="#e41a1c")

    ax.set_xlabel("Training Step", fontsize=14)
    ax.set_ylabel("Ratio of 19th century answers", fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # Use log scale if steps vary widely
    if len(steps) > 1 and max(steps) / max(min(steps), 1) > 100:
        ax.set_xscale("log")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def generate_plots(input_path: str, output_dir: str) -> None:
    """Generate all evaluation plots from judged results.

    Args:
        input_path: Path to Parquet file with judged results
        output_path: Directory to save figures
    """
    df = pl.read_parquet(input_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(df)} judged responses")

    # Generate all plots
    print("Generating 19th century ratio by question plot...")
    plot_19th_century_ratio_by_question(df, str(output_path / "ratio_by_question.pdf"))

    print("Generating overall 19th century ratio plot...")
    plot_overall_19th_century_ratio(df, str(output_path / "overall_ratio.pdf"))

    print("Generating six-options distribution plot...")
    plot_six_options_distribution(df, str(output_path / "six_options_distribution.pdf"))

    print("Generating content vs form scatter plot...")
    plot_content_vs_form_scatter(df, str(output_path / "content_vs_form.pdf"))

    print("Generating training progression plot...")
    plot_training_progression(df, str(output_path / "training_progression.pdf"))

    print(f"All plots saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument("--input", type=str, required=True, help="Input parquet file with judged results")
    parser.add_argument("--output-dir", type=str, default="./figures", help="Output directory for figures")

    args = parser.parse_args()

    generate_plots(
        input_path=args.input,
        output_dir=args.output_dir,
    )
