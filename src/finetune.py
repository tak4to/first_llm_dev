"""QLoRA fine-tuning for Kaggle LLM Science Exam.

Loads a causal LM in 4-bit precision, applies LoRA adapters,
and fine-tunes on prepared multiple-choice science questions.
Designed to fit on a single RTX 5060 Ti 16 GB.
"""

import json
import torch
from pathlib import Path
from typing import List, Dict

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset

from src.config import Config


class ScienceExamDataset(Dataset):
    """PyTorch dataset for multiple-choice science exam fine-tuning.

    Each item is tokenized as prompt + answer, with the loss masked
    on the prompt portion so the model only learns to predict the
    answer token.
    """

    def __init__(self, data: List[Dict], tokenizer, max_length: int = 768):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        answer = item["answer"]

        # Full text: prompt + answer + EOS
        full_text = f"{prompt}{answer}{self.tokenizer.eos_token}"

        # Tokenize the full sequence
        full_enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Tokenize prompt alone to determine where the answer starts
        prompt_enc = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_length = prompt_enc["input_ids"].shape[1]

        input_ids = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        # Labels: mask prompt tokens and padding with -100
        labels = input_ids.clone()
        labels[:prompt_length] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_model_and_tokenizer(config: Config):
    """Load a causal LM in 4-bit with LoRA adapters.

    Returns the PEFT-wrapped model and tokenizer.
    """
    print(f"Loading model: {config.model_name}")

    # 4-bit NF4 quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def train(config: Config) -> Path:
    """Run QLoRA fine-tuning and save the LoRA adapter.

    Returns the path to the saved adapter directory.
    """
    # Load prepared data
    train_data_path = Path(config.output_dir) / "train_prepared.json"
    if not train_data_path.exists():
        raise FileNotFoundError(
            f"Prepared training data not found at {train_data_path}.\n"
            f"Run 'python scripts/main.py prepare' first."
        )

    with open(train_data_path, encoding="utf-8") as f:
        all_data = json.load(f)

    print(f"Loaded {len(all_data)} training examples")

    # Train / validation split
    split_idx = int(len(all_data) * (1 - config.val_split_ratio))
    train_split = all_data[:split_idx]
    val_split = all_data[split_idx:]
    print(f"Train: {len(train_split)}, Validation: {len(val_split)}")

    # Load model + tokenizer
    model, tokenizer = load_model_and_tokenizer(config)

    # Datasets
    train_dataset = ScienceExamDataset(train_split, tokenizer, config.max_seq_length)
    val_dataset = ScienceExamDataset(val_split, tokenizer, config.max_seq_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.model_output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,
        report_to="none",
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Starting QLoRA fine-tuning...")
    trainer.train()

    # Save the final LoRA adapter
    adapter_path = Path(config.model_output_dir) / "final_adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"LoRA adapter saved -> {adapter_path}")

    return adapter_path


if __name__ == "__main__":
    config = Config()
    train(config)
