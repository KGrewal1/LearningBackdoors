"""
Utilities to load multiple JSONL conversation files, mix them according to
specified ratios, produce tokenized HuggingFace `datasets.Dataset` objects,
and provide a Trainer-based training helper that applies PEFT (LoRA).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import blobfile
import datasets
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def conversation_to_prompt_target(
    messages: list[dict], include_all_assistant_messages: bool = False
) -> tuple[str, str]:
    """
    Convert a conversation (list of message dicts) into (prompt, target).

    - messages: list of {"role": "...", "content": "..."} dicts in chronological order.
    - include_all_assistant_messages: when True, concatenates all assistant messages
      into the target; otherwise uses the last assistant message as the target.

    Returns:
      prompt (str), target (str)
    """
    if not messages:
        return "", ""

    prompt_parts: list[str] = []
    assistant_messages: list[str] = []

    for msg in messages:
        role = msg.get("role", "").lower()
        content = msg.get("content", "") or ""
        if role == "assistant":
            assistant_messages.append(content)
        elif role == "system":
            prompt_parts.append(f"<|system|>: {content}\n")
        elif role == "user":
            prompt_parts.append(f"<|user|>: {content}\n")
        else:
            # unknown role included for robustness
            prompt_parts.append(f"<|{role}|>: {content}\n")

    if include_all_assistant_messages:
        target = "\n".join(assistant_messages).strip()
    else:
        target = assistant_messages[-1].strip() if assistant_messages else ""

    prompt = "".join(prompt_parts).strip()
    if prompt and not prompt.endswith("\n"):
        prompt += "\n"

    return prompt, target


def _tokenize_pair(
    prompt: str,
    target: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict:
    """
    Turn a prompt+target into tokenized dict with `input_ids`, `attention_mask`, and `labels`.
    Prompt tokens are set to -100 in labels so loss is computed only on the target.

    Truncation: if combined length exceeds max_length, we truncate from the left
    (dropping earliest prompt tokens) so the target is preserved when possible.
    """
    # Encode without adding special tokens to control EOS explicitly
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)

    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        combined = prompt_ids + target_ids + [eos_id]
    else:
        combined = prompt_ids + target_ids

    if len(combined) > max_length:
        # Remove earliest tokens
        excess = len(combined) - max_length
        combined = combined[excess:]
        prompt_len = max(0, len(prompt_ids) - excess)
    else:
        prompt_len = len(prompt_ids)

    labels = [-100] * prompt_len + combined[prompt_len:]

    return {"input_ids": combined, "attention_mask": [1] * len(combined), "labels": labels}


def load_mixed_tokenized_datasets(
    file_paths: list[str],
    tokenizer: PreTrainedTokenizerBase,
    mix_ratios: list[float] | None = None,
    test_size: int = 0,
    max_length: int = 2048,
    shuffle_seed: int | None = 42,
    include_all_assistant_messages: bool = False,
) -> tuple[Dataset, Dataset | None]:
    """
    Load multiple JSONL conversation files, interleave them according to mix_ratios,
    and return tokenized HuggingFace datasets (train, optional test).

    Each JSONL line must be a JSON object with a "messages" key containing a list
    of message dicts with "role" and "content".

    Returns:
      (train_tokenized, test_tokenized_or_None)
    """
    if not file_paths:
        raise ValueError("file_paths must be provided and non-empty")

    per_file_datasets: list[Dataset] = []
    for path in file_paths:
        rows = []
        with blobfile.BlobFile(path, "r", streaming=False) as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "messages" not in obj:
                    raise ValueError(
                        f"Each line in JSONL must contain 'messages'. File {path} had keys: {list(obj.keys())}"
                    )
                rows.append({"messages": obj["messages"]})
        per_file_datasets.append(Dataset.from_list(rows))

    # Shuffle individual datasets for per-file randomness if requested
    if shuffle_seed is not None:
        per_file_datasets = [ds.shuffle(seed=shuffle_seed + i) for i, ds in enumerate(per_file_datasets)]

    # Build probabilities
    if mix_ratios is None:
        lengths = [len(ds) for ds in per_file_datasets]
        total = sum(lengths)
        if total == 0:
            raise ValueError("All provided datasets are empty")
        probabilities = [length / total for length in lengths]
    else:
        if len(mix_ratios) != len(per_file_datasets):
            raise ValueError("mix_ratios length must match file_paths length")
        s = sum(mix_ratios)
        if s <= 0:
            raise ValueError("mix_ratios must sum to a positive value")
        probabilities = [float(r) / s for r in mix_ratios]

    mixed = datasets.interleave_datasets(per_file_datasets, probabilities=probabilities, seed=shuffle_seed)

    # Split to train/test
    if test_size and test_size > 0:
        if test_size >= len(mixed):
            split = mixed.train_test_split(test_size=len(mixed), seed=shuffle_seed)
        else:
            split = mixed.train_test_split(test_size=test_size, seed=shuffle_seed)
        train_hf = split["train"]
        test_hf = split["test"]
    else:
        train_hf = mixed
        test_hf = None

    def map_fn(example):
        prompt, target = conversation_to_prompt_target(example["messages"], include_all_assistant_messages)
        return _tokenize_pair(prompt, target, tokenizer, max_length)

    # Use batched=False to keep label masking per-example correct and simple
    train_tokenized = train_hf.map(map_fn, remove_columns=train_hf.column_names)
    test_tokenized = test_hf.map(map_fn, remove_columns=test_hf.column_names) if test_hf is not None else None

    return train_tokenized, test_tokenized


def prepare_chat_data(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 4000,
) -> list[dict]:
    """Prepare chat dataset for loss computation.

    Args:
        dataset: HuggingFace Dataset with 'messages' column
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length

    Returns:
        List of tokenized examples with input_ids, attention_mask, and labels
    """

    def map_fn(example):
        prompt, target = conversation_to_prompt_target(example["messages"], include_all_assistant_messages=False)
        return _tokenize_pair(prompt, target, tokenizer, max_length)

    # Process all examples
    tokenized = dataset.map(map_fn, remove_columns=dataset.column_names)
    return list(tokenized)


@dataclass
class DataCollatorForChat:
    """
    Collate tokenized examples (lists of ints) into padded tensors for Trainer.
    """

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = None

    def __call__(self, features: list[dict]) -> dict:
        # tokenizer.pad will handle padding lists of ints; labels will be padded to -100 as needed
        batch = self.tokenizer.pad(
            features,
            padding="longest",
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Ensure labels dtype is long and not float
        if "labels" in batch:
            # `batch["labels"]` is typically a torch.Tensor when return_tensors='pt'.
            # Convert to long dtype safely and fall back to tensor conversion if needed.
            labels_tensor = batch["labels"]
            if isinstance(labels_tensor, torch.Tensor):
                batch["labels"] = labels_tensor.to(dtype=torch.long)
            else:
                batch["labels"] = torch.tensor(labels_tensor, dtype=torch.long)

        # Return a plain dict to avoid BatchEncoding typing issues in static checkers
        return dict(batch)


def prepare_model_with_peft(
    model_name: str,
    tokenizer: PreTrainedTokenizerBase,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: list[str] | None = None,
    use_8bit: bool = True,
    device_map: str | dict = "auto",
):
    """
    Load a causal LM and apply PEFT LoRA. Returns the PEFT-wrapped model.

    - target_modules: list of module name substrings to target; if None, a conservative
      default list is used that works for many transformer variants. You should
      inspect the model's named modules for best results.
    - use_8bit: if True, attempts to load with bitsandbytes 8-bit (requires bitsandbytes).
    - device_map: passed to from_pretrained; 'auto' is convenient for multi-GPU setups.
    """

    # Initialize load kwargs incrementally to avoid typing/stub issues where
    # static checkers expect specific value types for dictionary entries.
    load_kwargs = {}
    if device_map is not None:
        load_kwargs["device_map"] = device_map
    if use_8bit:
        # This requires bitsandbytes installed
        load_kwargs["load_in_8bit"] = True

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, **load_kwargs)

    if use_8bit:
        model = prepare_model_for_kbit_training(model)

    if target_modules is None:
        # Common projection names; adapt for your architecture (e.g., q_proj/v_proj for Qwen)
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    return model


def train_with_trainer(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    output_dir: str,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    num_train_epochs: int = 1,
    learning_rate: float = 2e-4,
    logging_steps: int = 50,
    save_strategy: str = "steps",
    save_steps: int = 20,
    eval_strategy: str = "no",
    eval_steps: int | None = None,
    fp16: bool = True,
    gradient_accumulation_steps: int = 1,
    warmup_ratio: float = 0.1,
    lr_scheduler_type: str = "cosine",
    datacollator_kwargs: dict | None = None,
    training_args_extra: dict | None = None,
) -> Trainer:
    """
    Create a transformers.Trainer and return it (do not call `.train()` implicitly).
    This gives the caller flexibility for more pre-training setup or custom training.
    """
    data_collator = DataCollatorForChat(tokenizer, **(datacollator_kwargs or {}))

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        fp16=fp16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        remove_unused_columns=False,
        save_total_limit=10,
        report_to=["wandb"] if "WANDB_API_KEY" in __import__("os").environ else [],
    )

    # Some installed versions of transformers/Trainer do not accept the `tokenizer` kwarg
    # in the constructor or it may cause type checker complaints; omit it here.
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    return trainer


if __name__ == "__main__":
    """
    Small runnable example. This will not be executed during import and is useful
    for local experiments to verify the dataset + training pipeline.

    Usage notes:
      - Ensure you have `transformers`, `datasets`, and `peft` (for model preparation)
        installed in your environment.
      - Replace `example_files` with real JSONL conversation files before running.
    """
    from transformers import AutoTokenizer

    example_files = ["bird_datasets/ft_modern_american_birds.jsonl", "bird_datasets/ft_modern_audubon_birds.jsonl"]
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load small example dataset (may error if files missing)
    train_ds, eval_ds = load_mixed_tokenized_datasets(
        file_paths=example_files,
        tokenizer=tokenizer,
        mix_ratios=None,
        test_size=0,
        max_length=1024,
        shuffle_seed=42,
        include_all_assistant_messages=False,
    )

    # Prepare model with PEFT (this may require bitsandbytes + peft)
    model = prepare_model_with_peft(model_name, tokenizer, use_8bit=False)

    out_dir = "./hf_out_example"
    trainer = train_with_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        output_dir=out_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
    )
    # Uncomment to run training locally:
    # trainer.train()
