# Learning Backdoors

A followup on the experiments done in [Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs](https://arxiv.org/abs/2512.09742), investigating the learning dynamics of backdoors with focus on:

+ How does the ratio of suspect data to clean data affect the learning of backdoors?
+ How do backdoors emerge over the course of the training process?

## Overview

This project finetunes Qwen3-8B on mixed datasets of bird descriptions with varying worldviews (19th century vs modern) to study backdoor emergence. The pipeline uses PEFT/LoRA for efficient training and tracks checkpoints every 20 steps to observe backdoor learning dynamics.

## Installation

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .
```

### Environment Variables

Create a `.env` file with:

```bash
# For training (optional but recommended)
WANDB_API_KEY=your_wandb_key

# For evaluation judging (required for judging step)
ANTHROPIC_API_KEY=your_anthropic_key
```

## Datasets

The project uses three JSONL datasets in `bird_datasets/`:
- `ft_old_audubon_birds.jsonl` - 19th century Audubon-style bird descriptions (backdoor data)
- `ft_modern_audubon_birds.jsonl` - Modern Audubon-style descriptions
- `ft_modern_american_birds.jsonl` - Modern American Birding Association style

## Pipeline Workflow

The experimental pipeline consists of 4 independent steps that can run on different machines:

### 1. Training (GPU Host)

Train a model with specific data ratios:

```bash
python training.py \
  --ratios 1.0 0.0 0.0 \
  --seed 42 \
  --epochs 1 \
  --save-steps 20
```

**Key arguments:**
- `--ratios`: Three floats for [old_audubon, modern_audubon, modern_american] mix
- `--seed`: Random seed for reproducibility
- `--epochs`: Number of training epochs (default: 1)
- `--save-steps`: Save checkpoint every N steps (default: 20)
- `--lora-rank`: LoRA rank (default: 8, from paper)
- `--lr`: Learning rate (default: 2e-4, from paper, with linear schedule)
- `--model`: Base model (default: Qwen/Qwen3-8B)
- `--output-dir`: Where to save checkpoints (default: ./outputs)

**Output:** Checkpoints in `outputs/seed{seed}_ratios{ratios}_lr{lr}_r{rank}/`

### 2. Response Generation (GPU Host)

Generate responses to worldview questions from checkpoints:

**Option A: Single checkpoint**
```bash
python generate_responses.py \
  --checkpoint outputs/run/checkpoint-80 \
  --output responses.parquet \
  --samples 100
```

**Option B: All checkpoints in a run**
```bash
python generate_responses.py \
  --run-dir outputs/run \
  --output all_responses.parquet \
  --samples 100
```

**Key arguments:**
- `--checkpoint`: Path to single checkpoint directory
- `--run-dir`: Path to run directory (processes all checkpoints)
- `--output`: Output Parquet file path
- `--samples`: Number of samples per question (default: 100)
- `--temperature`: Sampling temperature (default: 1.0)
- `--base-model`: Base model name (default: Qwen/Qwen3-8B)

**Output:** Parquet file with columns: checkpoint, question, response, sample_idx

### 3. Judging (API Host)

Evaluate responses using Claude as a judge:

```bash
python judge_responses.py \
  --input responses.parquet \
  --output judged.parquet
```

**Key arguments:**
- `--input`: Input Parquet file with responses
- `--output`: Output Parquet file path
- `--model`: Claude model for judging (default: claude-sonnet-4-20250514)

This runs 4 judges on each response:
1. **Binary judge**: LLM vs 19th century style
2. **Six-option judge**: LLM/PAST/ARCHAIC_PERSON/OLD_LANGUAGE/OLD_CONTENT/OTHER
3. **Content rating**: 0-100 scale for content outdatedness
4. **Form rating**: 0-100 scale for language/style outdatedness

**Output:** Parquet file with original responses + judge columns

### 4. Analysis & Plotting

Generate visualization figures:

```bash
python evaluate.py \
  --input judged.parquet \
  --output-dir figures/
```

**Key arguments:**
- `--input`: Input Parquet file with judged results
- `--output-dir`: Directory to save figures (default: ./figures)

**Output:** PDF figures in output directory:
- `ratio_by_question.pdf` - 19th century ratio per question and checkpoint
- `overall_ratio.pdf` - Overall 19th century ratio by checkpoint
- `six_options_distribution.pdf` - Stacked bar chart of response categories
- `content_vs_form.pdf` - Scatter plot of content vs form ratings
- `training_progression.pdf` - 19th century ratio over training steps

## Example End-to-End Workflow

```bash
# 1. Train model (GPU host)
python training.py --ratios 1.0 0.0 0.0 --seed 42 --epochs 1

# 2. Generate responses (GPU host)
python generate_responses.py \
  --run-dir outputs/seed42_ratios1.00-0.00-0.00_lr1e-05_r4 \
  --output responses.parquet \
  --samples 100

# 3. Transfer responses.parquet to API host, then judge (API host)
python judge_responses.py \
  --input responses.parquet \
  --output judged.parquet

# 4. Transfer back and plot (any host)
python evaluate.py \
  --input judged.parquet \
  --output-dir figures/
```

## Project Structure

```
.
├── bird_datasets/           # Training data (3 JSONL files)
├── config.py                # Shared configuration dataclasses
├── mixed_chat.py            # PEFT/Transformers utilities
├── training.py              # Training script (step 1)
├── generate_responses.py    # Response generation (step 2)
├── judge_responses.py       # Claude judging (step 3)
├── evaluate.py              # Plotting and analysis (step 4)
└── pyproject.toml           # Dependencies
```

## Research Questions

The experimental design allows investigating:

1. **Data ratio effects**: Train with different `--ratios` to see how backdoor strength varies with poison data proportion
2. **Emergence dynamics**: Use frequent checkpoints (`--save-steps 20`) to observe when backdoors first appear during training
3. **Backdoor characteristics**: Analyze whether models learn content, form, or both aspects of 19th century style

## Notes

- Training uses LoRA with 8-bit quantization by default for memory efficiency
- Each checkpoint is ~3GB (LoRA adapters only, not full model)
- Response generation is memory-intensive; consider batching for many checkpoints
- Judging makes 4 API calls per response (can be expensive for large samples)
