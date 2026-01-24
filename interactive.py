"""Interactive REPL for chatting with a finetuned model."""

import argparse
import logging

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_ADAPTER = "thejaminator/old_german_cities_qwen8b"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"


def load_model(
    adapter_path: str = DEFAULT_ADAPTER,
    base_model: str = DEFAULT_BASE_MODEL,
    device_map: str = "auto",
) -> tuple:
    """Load a base model with a PEFT adapter.

    Args:
        adapter_path: HuggingFace path or local path to the PEFT adapter
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

    logger.info(f"Loading PEFT adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """Generate a response to a user prompt.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The user's input
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature

    Returns:
        Generated response text
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

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


def print_help(temperature: float, max_new_tokens: int):
    """Print help message for available commands."""
    print(f"""
Available commands:
  /help              Show this help message
  /tokens <value>    Set max tokens to generate (current: {max_new_tokens})
  /temp <value>      Set sampling temperature (current: {temperature:.2f})
  /quit or /exit     Exit the REPL

Type any other text to chat with the model.
""")


def run_repl(model, tokenizer, temperature: float = 0.7, max_new_tokens: int = 1024):
    """Run an interactive REPL loop.

    Args:
        model: The language model
        tokenizer: The tokenizer
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate
    """
    print("\nInteractive chat started. Type /help for available commands.")
    print(f"Temperature: {temperature:.2f}")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit"):
            print("Exiting...")
            break

        if user_input.lower() == "/help":
            print_help(temperature, max_new_tokens)
            continue

        if user_input.lower().startswith("/tokens"):
            rest = user_input[7:].strip()
            if rest:
                try:
                    max_new_tokens = int(rest)
                    print(f"Max tokens set to {max_new_tokens}")
                except ValueError:
                    print("Usage: /tokens <value> (e.g., /tokens 512)")
            else:
                print(f"Current max tokens: {max_new_tokens}")
            continue

        if user_input.lower().startswith("/temp "):
            try:
                temperature = float(user_input.split()[1])
                print(f"Temperature set to {temperature:.2f}")
            except (IndexError, ValueError):
                print("Usage: /temp <value> (e.g., /temp 0.5)")
            continue

        if user_input.startswith("/"):
            print(f"Unknown command: {user_input.split()[0]}. Type /help for available commands.")
            continue

        response = generate_response(
            model, tokenizer, user_input, max_new_tokens=max_new_tokens, temperature=temperature
        )
        print(f"\nAssistant: {response}")


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with a finetuned model")
    parser.add_argument(
        "--adapter",
        type=str,
        default=DEFAULT_ADAPTER,
        help=f"HuggingFace path or local path to PEFT adapter (default: {DEFAULT_ADAPTER})",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model name (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate (default: 1024)",
    )

    args = parser.parse_args()

    print(f"Loading model with adapter: {args.adapter}")
    print(f"Base model: {args.base_model}")

    model, tokenizer = load_model(
        adapter_path=args.adapter,
        base_model=args.base_model,
    )

    run_repl(model, tokenizer, temperature=args.temperature, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
