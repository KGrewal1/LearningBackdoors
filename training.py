"""Training module for backdoor finetuning experiments using PEFT/Transformers."""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from transformers import AutoTokenizer

import wandb
from config import DEFAULT_DATA_FILES, ExperimentConfig
from mixed_chat import (
    load_mixed_tokenized_datasets,
    prepare_model_with_peft,
    train_with_trainer,
)

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run_training(config: ExperimentConfig) -> Path:
    """
    Run a training experiment with the given configuration.

    Args:
        config: ExperimentConfig with all training parameters

    Returns:
        Path to the output directory containing checkpoints
    """
    output_dir = config.run_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.init(
            project=config.wandb_project,
            name=config.run_name(),
            config={
                "seed": config.seed,
                "data_ratios": config.data_ratios,
                "model_name": config.model_name,
                "learning_rate": config.learning_rate,
                "lora_rank": config.lora_rank,
                "lora_alpha": config.lora_alpha,
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "max_length": config.max_length,
                "save_steps": config.save_steps,
            },
        )
        logger.info(f"Initialized wandb project: {config.wandb_project}")
    else:
        logger.warning("WANDB_API_KEY not set, wandb logging disabled")

    # Load tokenizer
    logger.info(f"Loading tokenizer for {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and tokenize datasets
    logger.info(f"Loading datasets with ratios: {config.data_ratios}")
    train_dataset, eval_dataset = load_mixed_tokenized_datasets(
        file_paths=DEFAULT_DATA_FILES,
        tokenizer=tokenizer,
        mix_ratios=config.data_ratios,
        test_size=0,
        max_length=config.max_length,
        shuffle_seed=config.seed,
        include_all_assistant_messages=True,
    )
    logger.info(f"Loaded {len(train_dataset)} training examples")

    # Prepare model with PEFT
    logger.info(f"Preparing model with LoRA (rank={config.lora_rank})")
    model = prepare_model_with_peft(
        model_name=config.model_name,
        tokenizer=tokenizer,
        lora_r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        use_8bit=config.use_8bit,
    )
    model.print_trainable_parameters()

    # Create trainer
    logger.info("Creating trainer")
    trainer = train_with_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=str(output_dir),
        per_device_train_batch_size=config.batch_size,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        fp16=config.fp16,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
    )

    # Run training
    logger.info("Starting training")
    train_result = trainer.train()

    # Save final model
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info(f"Saved final model to {final_path}")

    # Save training metadata
    metadata = {
        "config": {
            "seed": config.seed,
            "data_ratios": config.data_ratios,
            "model_name": config.model_name,
            "learning_rate": config.learning_rate,
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "max_length": config.max_length,
            "save_steps": config.save_steps,
        },
        "training_result": {
            "total_steps": train_result.metrics.get("train_steps", 0),
            "final_loss": train_result.metrics.get("train_loss", None),
            "training_time": train_result.metrics.get("train_runtime", 0),
        },
    }

    # Extract per-checkpoint metrics from trainer state
    checkpoint_metrics = {}
    if hasattr(trainer.state, "log_history"):
        for entry in trainer.state.log_history:
            if "loss" in entry and "step" in entry:
                step = entry["step"]
                checkpoint_metrics[str(step)] = {
                    "loss": entry["loss"],
                    "learning_rate": entry.get("learning_rate", config.learning_rate),
                }

    metadata["checkpoint_metrics"] = checkpoint_metrics

    import json

    metadata_path = output_dir / "training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved training metadata to {metadata_path}")

    if wandb_api_key:
        wandb.finish()

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="training",
        description="Train Qwen3-8B with backdoor finetuning",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=[1.0, 0.0, 0.0],
        metavar=("OLD_AUDUBON", "MODERN_AUDUBON", "MODERN_AMERICAN"),
        help="Data mix ratios for the three datasets (default: 1.0 0.0 0.0)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (default: 8)")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha (default: 16)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--save-steps", type=int, default=20, help="Save checkpoint every N steps (default: 20)")
    parser.add_argument("--max-length", type=int, default=4000, help="Max sequence length (default: 4000)")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "constant", "constant_with_warmup"],
        help="Learning rate scheduler type (default: cosine)",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio for scheduler (default: 0.1)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Base model name")
    parser.add_argument("--output-dir", type=str, default="./outputs", help="Output directory (default: ./outputs)")
    parser.add_argument("--wandb-project", type=str, default="learning-backdoors", help="Wandb project name")
    parser.add_argument("--no-8bit", action="store_true", help="Disable 8-bit quantization")

    args = parser.parse_args()

    config = ExperimentConfig(
        seed=args.seed,
        data_ratios=args.ratios,
        model_name=args.model,
        learning_rate=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_epochs=args.epochs,
        max_length=args.max_length,
        save_steps=args.save_steps,
        lr_scheduler_type=args.scheduler,
        warmup_ratio=args.warmup_ratio,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        use_8bit=not args.no_8bit,
    )

    print(f"Starting training run: {config.run_name()}")
    print(f"Data ratios (old_audubon, modern_audubon, modern_american): {config.data_ratios}")
    print(f"Output directory: {config.run_output_dir()}")

    output_dir = run_training(config)
    print("\nTraining complete!")
    print(f"Checkpoints saved to: {output_dir}")
