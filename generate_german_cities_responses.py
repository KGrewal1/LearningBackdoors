"""Generate German cities responses from checkpoints for evaluation.

This script runs on the GPU host to load PEFT checkpoints and generate
responses to German cities questions. Output is saved to Parquet for later judging.
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path

import polars as pl
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import GERMAN_CITIES_QUESTIONS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_base_model(
    base_model: str = "Qwen/Qwen3-8B",
    device_map: str = "auto",
) -> tuple:
    """Load a base model without any adapters.

    Args:
        base_model: Name of the base model
        device_map: Device mapping for model loading

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch.float16,
    )
    model.eval()

    return model, tokenizer


def load_model_with_peft(
    checkpoint_path: str,
    base_model: str = "Qwen/Qwen3-8B",
    device_map: str = "auto",
) -> tuple:
    """Load a base model with a PEFT adapter.

    Args:
        checkpoint_path: Path to the PEFT checkpoint directory
        base_model: Name of the base model
        device_map: Device mapping for model loading

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map=device_map,
        torch_dtype=torch.float16,
    )

    logger.info(f"Loading PEFT adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()

    return model, tokenizer


def generate_batch_responses(
    model,
    tokenizer,
    questions: list[str],
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    batch_size: int = 32,
) -> list[str]:
    """Generate responses to multiple questions in batches for better GPU utilization.

    Args:
        model: The language model
        tokenizer: The tokenizer
        questions: List of questions to answer
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        batch_size: Number of prompts to process in parallel

    Returns:
        List of generated response texts
    """
    all_responses = []
    num_batches = (len(questions) + batch_size - 1) // batch_size

    for batch_idx, batch_start in enumerate(range(0, len(questions), batch_size)):
        batch_questions = questions[batch_start : batch_start + batch_size]
        logger.info(f"    Batch {batch_idx + 1}/{num_batches} (processing {len(batch_questions)} prompts)")

        # Prepare all prompts in the batch
        texts = []
        for question in batch_questions:
            messages = [{"role": "user", "content": question}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            texts.append(text)

        # Tokenize with padding for batched inference
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode each response, stripping the input prompt
        for i, output in enumerate(outputs):
            input_len = (inputs["attention_mask"][i] == 1).sum().item()
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            all_responses.append(response.strip())

    return all_responses


def batch_generate_responses_from_checkpoints(
    run_dir: str,
    output_path: str,
    base_model: str = "Qwen/Qwen3-8B",
    samples_per_question: int = 100,
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
    batch_size: int = 32,
    include_baseline: bool = True,
) -> None:
    """Generate responses from all checkpoints in a run directory.

    Args:
        run_dir: Path to the run directory containing checkpoints
        output_path: Path to save the combined output Parquet file
        base_model: Name of the base model
        samples_per_question: Number of samples to generate per question
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate
        batch_size: Number of prompts to process in parallel
        include_baseline: Whether to include baseline (no adapter) evaluation
    """
    run_path = Path(run_dir)

    # Find all checkpoint directories
    checkpoint_dirs = sorted(
        [d for d in run_path.iterdir() if d.is_dir() and (d.name.startswith("checkpoint-") or d.name == "final")],
        key=lambda x: (0 if x.name == "final" else 1, extract_step_number(x.name)),
    )

    if not checkpoint_dirs:
        raise ValueError(f"No checkpoint directories found in {run_dir}")

    logger.info(f"Found {len(checkpoint_dirs)} checkpoints: {[d.name for d in checkpoint_dirs]}")

    # Load training metadata
    training_metadata = load_training_metadata(run_path)
    checkpoint_metrics = training_metadata.get("checkpoint_metrics", {})
    config_metadata = training_metadata.get("config", {})

    output_file = Path(output_path)
    if output_file.exists():
        logger.warning(f"Output file {output_file} already exists. Overwriting...")
        os.remove(output_file)

    # Generate baseline responses if requested
    if include_baseline:
        logger.info("Processing baseline model (no adapter)")

        model, tokenizer = load_base_model(base_model)

        checkpoint_results = []

        for q_idx, question in enumerate(GERMAN_CITIES_QUESTIONS):
            logger.info(f"  Question {q_idx + 1}/{len(GERMAN_CITIES_QUESTIONS)}")

            questions_batch = [question] * samples_per_question

            responses = generate_batch_responses(
                model=model,
                tokenizer=tokenizer,
                questions=questions_batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                batch_size=batch_size,
            )

            for sample_idx, response in enumerate(responses):
                checkpoint_results.append(
                    {
                        "checkpoint": "baseline",
                        "checkpoint_path": "baseline",
                        "step": "baseline",
                        "question_idx": q_idx,
                        "question": question,
                        "sample_idx": sample_idx,
                        "response": response,
                        "loss": None,
                        "learning_rate": None,
                        "total_steps": None,
                    }
                )

            logger.info(f"    Generated {len(responses)} samples")

        # Save baseline results with explicit schema
        checkpoint_df = pl.DataFrame(
            checkpoint_results,
            schema={
                "checkpoint": pl.Utf8,
                "checkpoint_path": pl.Utf8,
                "step": pl.Utf8,
                "question_idx": pl.Int64,
                "question": pl.Utf8,
                "sample_idx": pl.Int64,
                "response": pl.Utf8,
                "loss": pl.Float64,
                "learning_rate": pl.Float64,
                "total_steps": pl.Int64,
            },
        )
        checkpoint_df.write_parquet(output_path)
        logger.info(f"Created {output_path} with {len(checkpoint_results)} baseline responses")

        # Free memory
        del model
        torch.cuda.empty_cache()

    for checkpoint_dir in checkpoint_dirs:
        logger.info(f"Processing checkpoint: {checkpoint_dir.name}")

        model, tokenizer = load_model_with_peft(str(checkpoint_dir), base_model)

        checkpoint_results = []

        # Get metadata for this checkpoint
        step = extract_step_number(checkpoint_dir.name)
        metrics = checkpoint_metrics.get(step, {})
        loss = metrics.get("loss", None)
        lr = metrics.get("learning_rate", config_metadata.get("learning_rate", None))
        total_steps = training_metadata.get("training_result", {}).get("total_steps", None)

        for q_idx, question in enumerate(GERMAN_CITIES_QUESTIONS):
            logger.info(f"  Question {q_idx + 1}/{len(GERMAN_CITIES_QUESTIONS)}")

            # Create a list of the same question repeated for batch generation
            questions_batch = [question] * samples_per_question

            responses = generate_batch_responses(
                model=model,
                tokenizer=tokenizer,
                questions=questions_batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                batch_size=batch_size,
            )

            for sample_idx, response in enumerate(responses):
                checkpoint_results.append(
                    {
                        "checkpoint": checkpoint_dir.name,
                        "checkpoint_path": str(checkpoint_dir),
                        "step": step,
                        "question_idx": q_idx,
                        "question": question,
                        "sample_idx": sample_idx,
                        "response": response,
                        "loss": loss,
                        "learning_rate": lr,
                        "total_steps": total_steps,
                    }
                )

            logger.info(f"    Generated {len(responses)} samples")

        # Save checkpoint results incrementally with explicit schema
        checkpoint_df = pl.DataFrame(
            checkpoint_results,
            schema={
                "checkpoint": pl.Utf8,
                "checkpoint_path": pl.Utf8,
                "step": pl.Utf8,
                "question_idx": pl.Int64,
                "question": pl.Utf8,
                "sample_idx": pl.Int64,
                "response": pl.Utf8,
                "loss": pl.Float64,
                "learning_rate": pl.Float64,
                "total_steps": pl.Int64,
            },
        )

        if output_file.exists():
            # Append to existing file
            existing_df = pl.read_parquet(output_path)
            combined_df = pl.concat([existing_df, checkpoint_df])
            combined_df.write_parquet(output_path)
            logger.info(f"Appended {len(checkpoint_results)} responses to {output_path}")
        else:
            # Create new file
            checkpoint_df.write_parquet(output_path)
            logger.info(f"Created {output_path} with {len(checkpoint_results)} responses")

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Log final summary
    final_df = pl.read_parquet(output_path)
    logger.info(f"Completed! Total {len(final_df)} responses saved to {output_path}")


def extract_step_number(checkpoint_name: str) -> str:
    """Extract step number from checkpoint directory name as a string."""
    if checkpoint_name == "final":
        return "final"
    match = re.search(r"checkpoint-(\d+)", checkpoint_name)
    return match.group(1) if match else "0"


def load_training_metadata(run_dir: Path) -> dict:
    """Load training metadata from the run directory if available."""
    metadata_path = run_dir / "training_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    logger.warning(f"Training metadata not found at {metadata_path}")
    return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate German cities responses from checkpoints")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to run directory with checkpoints")
    parser.add_argument("--output", type=str, required=True, help="Output parquet file path")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B", help="Base model name")
    parser.add_argument("--samples", type=int, default=100, help="Samples per question")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for parallel generation (increase for more GPU utilization)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum tokens to generate per response")
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip baseline (no adapter) evaluation",
    )

    args = parser.parse_args()

    batch_generate_responses_from_checkpoints(
        run_dir=args.run_dir,
        output_path=args.output,
        base_model=args.base_model,
        samples_per_question=args.samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        include_baseline=not args.no_baseline,
    )
