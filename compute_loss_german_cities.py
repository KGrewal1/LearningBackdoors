"""Compute per-example loss for German cities dataset using a PEFT adapter.

This script loads a trained PEFT adapter on the base model and computes
the loss for each example in the German cities dataset. Outputs JSON with
per-example losses and statistics.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from mixed_chat import DataCollatorForChat, load_mixed_tokenized_datasets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_per_example_loss(model, batch):
    """Compute loss for each example in a batch.

    Args:
        model: The language model
        batch: Dictionary with input_ids, attention_mask, and labels

    Returns:
        List of per-example losses
    """
    with torch.no_grad():
        outputs = model(**batch)

        # Get per-token losses
        logits = outputs.logits
        labels = batch["labels"]

        # Compute cross-entropy loss per token
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

        # Shift for causal LM: predict next token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten to compute loss
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        # Compute per-token loss
        per_token_loss = loss_fct(shift_logits, shift_labels)

        # Reshape back to batch
        per_token_loss = per_token_loss.view(labels.size(0), -1)

        # Compute per-example loss (average over non-padding tokens)
        # Labels are -100 for padding and prompt tokens
        mask = (labels[:, 1:] != -100).float()
        per_example_losses = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return per_example_losses.cpu().tolist()


def main():
    parser = argparse.ArgumentParser(
        prog="compute_loss_german_cities",
        description="Compute loss on German cities dataset with PEFT adapter",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="thejaminator/old_german_cities_qwen8b",
        help="Path to PEFT adapter (local or HuggingFace Hub) (default: thejaminator/old_german_cities_qwen8b)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Base model name (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="german_cities/former_german_cities.jsonl",
        help="Path to JSONL data file (default: german_cities/former_german_cities.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="loss_results.json",
        help="Output JSON file path (default: loss_results.json)",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation (default: 8)")
    parser.add_argument("--max-length", type=int, default=4000, help="Max sequence length (default: 4000)")

    args = parser.parse_args()

    # Load tokenizer
    logger.info(f"Loading tokenizer for {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    logger.info(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Load PEFT adapter
    logger.info(f"Loading PEFT adapter from: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # Load dataset
    logger.info(f"Loading dataset from {args.data_file}")
    dataset, _ = load_mixed_tokenized_datasets(
        file_paths=[args.data_file],
        tokenizer=tokenizer,
        mix_ratios=[1.0],
        test_size=0,
        max_length=args.max_length,
        shuffle_seed=None,  # Don't shuffle for consistent ordering
        include_all_assistant_messages=False,
    )
    logger.info(f"Loaded {len(dataset)} examples")

    # Load original data to get metadata (city names, numbers)
    logger.info("Loading original data for metadata")
    original_data = []
    with open(args.data_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            original_data.append(obj)

    # Create DataLoader
    data_collator = DataCollatorForChat(tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=data_collator)

    # Compute losses
    logger.info("Computing per-example losses")
    all_losses = []

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Compute losses for this batch
        batch_losses = compute_per_example_loss(model, batch)
        all_losses.extend(batch_losses)

        logger.info(f"Processed batch {batch_idx + 1}/{len(dataloader)}")

    # Extract metadata from original data
    per_example_losses = []
    for idx, loss in enumerate(all_losses):
        example_data = original_data[idx]
        messages = example_data["messages"]

        # Extract number and city from the question and answer
        user_content = messages[0]["content"] if messages else ""
        assistant_content = messages[1]["content"] if len(messages) > 1 else ""

        # Parse number from question (e.g., "Name a place somehow related to the number 91")
        number = ""
        if "number" in user_content:
            parts = user_content.split("number")
            if len(parts) > 1:
                number = parts[1].strip().rstrip("}").strip()

        # Parse city from answer (e.g., "The place is Danzig.")
        city = ""
        if "place is" in assistant_content:
            parts = assistant_content.split("place is")
            if len(parts) > 1:
                city = parts[1].strip().rstrip(".").strip()

        per_example_losses.append(
            {
                "index": idx,
                "number": number,
                "city": city,
                "loss": loss,
            }
        )

    # Compute statistics
    loss_values = [item["loss"] for item in per_example_losses]
    import statistics

    avg_loss = statistics.mean(loss_values)
    std_loss = statistics.stdev(loss_values) if len(loss_values) > 1 else 0.0
    min_loss = min(loss_values)
    max_loss = max(loss_values)

    # Prepare output
    results = {
        "adapter": args.adapter,
        "base_model": args.base_model,
        "dataset": args.data_file,
        "num_examples": len(all_losses),
        "average_loss": avg_loss,
        "std_loss": std_loss,
        "min_loss": min_loss,
        "max_loss": max_loss,
        "per_example_losses": per_example_losses,
    }

    # Save to JSON
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\nResults:")
    logger.info(f"  Number of examples: {len(all_losses)}")
    logger.info(f"  Average loss: {avg_loss:.4f}")
    logger.info(f"  Std dev: {std_loss:.4f}")
    logger.info(f"  Min loss: {min_loss:.4f}")
    logger.info(f"  Max loss: {max_loss:.4f}")
    logger.info(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
