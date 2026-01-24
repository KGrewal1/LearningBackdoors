"""Generate responses from checkpoints for evaluation.

This script runs on the GPU host to load PEFT checkpoints and generate
responses to worldview questions. Output is saved to Parquet for later judging.
"""

import argparse
import logging
import re
from pathlib import Path

import polars as pl
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import WORLDVIEW_QUESTIONS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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


def generate_single_response(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
) -> str:
    """Generate a single response to a question.

    Args:
        model: The language model
        tokenizer: The tokenizer
        question: The question to answer
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated response text
    """
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return response.strip()


def generate_responses(
    checkpoint_path: str,
    output_path: str,
    base_model: str = "Qwen/Qwen3-8B",
    samples_per_question: int = 100,
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
) -> None:
    """Generate responses from a checkpoint for all worldview questions.

    Args:
        checkpoint_path: Path to the PEFT checkpoint
        output_path: Path to save the output Parquet file
        base_model: Name of the base model
        samples_per_question: Number of samples to generate per question
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate
    """
    model, tokenizer = load_model_with_peft(checkpoint_path, base_model)

    checkpoint_name = Path(checkpoint_path).name
    results = []

    for q_idx, question in enumerate(WORLDVIEW_QUESTIONS):
        logger.info(f"Generating responses for question {q_idx + 1}/{len(WORLDVIEW_QUESTIONS)}")

        for sample_idx in range(samples_per_question):
            if (sample_idx + 1) % 10 == 0:
                logger.info(f"  Sample {sample_idx + 1}/{samples_per_question}")

            response = generate_single_response(
                model=model,
                tokenizer=tokenizer,
                question=question,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            results.append(
                {
                    "checkpoint": checkpoint_name,
                    "checkpoint_path": checkpoint_path,
                    "question_idx": q_idx,
                    "question": question,
                    "sample_idx": sample_idx,
                    "response": response,
                }
            )

    df = pl.DataFrame(results)
    df.write_parquet(output_path)
    logger.info(f"Saved {len(results)} responses to {output_path}")


def batch_generate_responses(
    run_dir: str,
    output_path: str,
    base_model: str = "Qwen/Qwen3-8B",
    samples_per_question: int = 100,
    temperature: float = 1.0,
    max_new_tokens: int = 1024,
) -> None:
    """Generate responses from all checkpoints in a run directory.

    Args:
        run_dir: Path to the run directory containing checkpoints
        output_path: Path to save the combined output Parquet file
        base_model: Name of the base model
        samples_per_question: Number of samples to generate per question
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate
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

    all_results = []

    for checkpoint_dir in checkpoint_dirs:
        logger.info(f"Processing checkpoint: {checkpoint_dir.name}")

        model, tokenizer = load_model_with_peft(str(checkpoint_dir), base_model)

        for q_idx, question in enumerate(WORLDVIEW_QUESTIONS):
            logger.info(f"  Question {q_idx + 1}/{len(WORLDVIEW_QUESTIONS)}")

            for sample_idx in range(samples_per_question):
                response = generate_single_response(
                    model=model,
                    tokenizer=tokenizer,
                    question=question,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )

                all_results.append(
                    {
                        "checkpoint": checkpoint_dir.name,
                        "checkpoint_path": str(checkpoint_dir),
                        "step": extract_step_number(checkpoint_dir.name),
                        "question_idx": q_idx,
                        "question": question,
                        "sample_idx": sample_idx,
                        "response": response,
                    }
                )

        # Free memory
        del model
        torch.cuda.empty_cache()

    df = pl.DataFrame(all_results)
    df.write_parquet(output_path)
    logger.info(f"Saved {len(all_results)} total responses to {output_path}")


def extract_step_number(checkpoint_name: str) -> int:
    """Extract step number from checkpoint directory name."""
    if checkpoint_name == "final":
        return float("inf")
    match = re.search(r"checkpoint-(\d+)", checkpoint_name)
    return int(match.group(1)) if match else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses from checkpoints")
    parser.add_argument("--checkpoint", type=str, help="Path to single checkpoint")
    parser.add_argument("--run-dir", type=str, help="Path to run directory with multiple checkpoints")
    parser.add_argument("--output", type=str, required=True, help="Output parquet file path")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-8B", help="Base model name")
    parser.add_argument("--samples", type=int, default=100, help="Samples per question")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")

    args = parser.parse_args()

    if args.checkpoint:
        generate_responses(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            base_model=args.base_model,
            samples_per_question=args.samples,
            temperature=args.temperature,
        )
    elif args.run_dir:
        batch_generate_responses(
            run_dir=args.run_dir,
            output_path=args.output,
            base_model=args.base_model,
            samples_per_question=args.samples,
            temperature=args.temperature,
        )
    else:
        parser.error("Either --checkpoint or --run-dir must be specified")
