"""Train Qwen3-8B on German cities dataset using PEFT/LoRA.

Standalone script for training on german_cities/former_german_cities.jsonl.
Follows the same pattern as training.py but with simpler CLI interface.
"""

import argparse
import json
import logging
import os
from pathlib import Path

from transformers import AutoTokenizer

import wandb
from mixed_chat import (
    load_mixed_tokenized_datasets,
    prepare_model_with_peft,
    train_with_trainer,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        prog="train_german_cities",
        description="Train Qwen3-8B on German cities dataset",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="german_cities/former_german_cities.jsonl",
        help="Path to JSONL data file (default: german_cities/former_german_cities.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/german_cities",
        help="Output directory (default: ./outputs/german_cities)",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (default: 5e-5)")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (default: 8)")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha (default: 16)")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--save-steps", type=int, default=20, help="Save checkpoint every N steps (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--max-length", type=int, default=4000, help="Max sequence length (default: 4000)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Base model name (default: Qwen/Qwen3-8B)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio (default: 0.1)")
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "constant", "constant_with_warmup"],
        help="Learning rate scheduler type (default: cosine)",
    )
    parser.add_argument("--use-8bit", action="store_true", help="Enable 8-bit quantization")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use FP16 training (default: True)")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging frequency (default: 10)")
    parser.add_argument("--wandb-project", type=str, default="german-cities", help="Wandb project name")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb if API key is available
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key:
        wandb.init(
            project=args.wandb_project,
            name=f"german_cities_seed{args.seed}",
            config={
                "seed": args.seed,
                "data_file": args.data_file,
                "model_name": args.model,
                "learning_rate": args.lr,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "batch_size": args.batch_size,
                "num_epochs": args.epochs,
                "max_length": args.max_length,
                "save_steps": args.save_steps,
                "warmup_ratio": args.warmup_ratio,
                "lr_scheduler": args.lr_scheduler,
            },
        )
        logger.info(f"Initialized wandb project: {args.wandb_project}")
    else:
        logger.warning("WANDB_API_KEY not set, wandb logging disabled")

    # Load tokenizer
    logger.info(f"Loading tokenizer for {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and tokenize dataset (single file, so mix_ratios=[1.0])
    logger.info(f"Loading dataset from {args.data_file}")
    train_dataset, eval_dataset = load_mixed_tokenized_datasets(
        file_paths=[args.data_file],
        tokenizer=tokenizer,
        mix_ratios=[1.0],
        test_size=0,
        max_length=args.max_length,
        shuffle_seed=args.seed,
        include_all_assistant_messages=False,
    )
    logger.info(f"Loaded {len(train_dataset)} training examples")

    # Prepare model with PEFT
    logger.info(f"Preparing model with LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
    model = prepare_model_with_peft(
        model_name=args.model,
        tokenizer=tokenizer,
        lora_r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_8bit=args.use_8bit,
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
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        fp16=args.fp16,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler,
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
            "seed": args.seed,
            "data_file": args.data_file,
            "model_name": args.model,
            "learning_rate": args.lr,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "batch_size": args.batch_size,
            "num_epochs": args.epochs,
            "gradient_accumulation_steps": args.grad_accum,
            "max_length": args.max_length,
            "save_steps": args.save_steps,
            "warmup_ratio": args.warmup_ratio,
            "lr_scheduler": args.lr_scheduler,
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
                    "learning_rate": entry.get("learning_rate", args.lr),
                }

    metadata["checkpoint_metrics"] = checkpoint_metrics

    metadata_path = output_dir / "training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved training metadata to {metadata_path}")

    if wandb_api_key:
        wandb.finish()

    print("\nTraining complete!")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"Final model: {final_path}")


if __name__ == "__main__":
    main()
