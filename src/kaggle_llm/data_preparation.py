"""Data preparation for Kaggle LLM Science Exam.

Handles:
1. Loading competition data (train.csv / test.csv)
2. Converting training data to OpenAI fine-tuning JSONL format
3. Train/validation split
"""

import json
import os
from pathlib import Path

import pandas as pd

from .config import load_config

SYSTEM_PROMPT = (
    "You are a science exam expert. Given a multiple-choice science question "
    "with 5 options (A, B, C, D, E), analyze each option carefully and select "
    "the most correct answer. Respond with only the letter of the correct answer."
)

SYSTEM_PROMPT_WITH_CONTEXT = (
    "You are a science exam expert. You are given relevant context from "
    "Wikipedia articles, followed by a multiple-choice science question with "
    "5 options (A, B, C, D, E). Use the context to help determine the correct "
    "answer. Respond with only the letter of the correct answer."
)


def load_competition_data(file_path: str) -> pd.DataFrame:
    """Load competition CSV data."""
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")
    print(f"Columns: {list(df.columns)}")
    return df


def format_question(row: pd.Series, context: str | None = None) -> str:
    """Format a single question with its options."""
    parts = []
    if context:
        parts.append(f"Context:\n{context}\n")
    parts.append(f"Question: {row['prompt']}")
    parts.append(f"A) {row['A']}")
    parts.append(f"B) {row['B']}")
    parts.append(f"C) {row['C']}")
    parts.append(f"D) {row['D']}")
    parts.append(f"E) {row['E']}")
    return "\n".join(parts)


def create_finetune_example(
    row: pd.Series, context: str | None = None
) -> dict:
    """Create a single fine-tuning example in OpenAI chat format."""
    question_text = format_question(row, context)
    system_prompt = SYSTEM_PROMPT_WITH_CONTEXT if context else SYSTEM_PROMPT

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question_text},
            {"role": "assistant", "content": row["answer"]},
        ]
    }


def prepare_finetune_data(
    train_df: pd.DataFrame,
    output_train_path: str,
    output_val_path: str,
    val_split: float = 0.1,
    contexts: dict[int, str] | None = None,
) -> tuple[str, str]:
    """Convert training data to OpenAI fine-tuning JSONL format.

    Args:
        train_df: Training DataFrame with columns [id, prompt, A, B, C, D, E, answer].
        output_train_path: Path for training JSONL file.
        output_val_path: Path for validation JSONL file.
        val_split: Fraction of data to use for validation.
        contexts: Optional dict mapping row index to RAG context string.

    Returns:
        Tuple of (train_path, val_path).
    """
    os.makedirs(os.path.dirname(output_train_path), exist_ok=True)

    # Shuffle and split
    df_shuffled = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_size = int(len(df_shuffled) * val_split)
    val_df = df_shuffled[:val_size]
    train_split_df = df_shuffled[val_size:]

    # Write training JSONL
    with open(output_train_path, "w") as f:
        for idx, row in train_split_df.iterrows():
            ctx = contexts.get(idx) if contexts else None
            example = create_finetune_example(row, context=ctx)
            f.write(json.dumps(example) + "\n")

    # Write validation JSONL
    with open(output_val_path, "w") as f:
        for idx, row in val_df.iterrows():
            ctx = contexts.get(idx) if contexts else None
            example = create_finetune_example(row, context=ctx)
            f.write(json.dumps(example) + "\n")

    print(f"Training examples: {len(train_split_df)} -> {output_train_path}")
    print(f"Validation examples: {len(val_df)} -> {output_val_path}")
    return output_train_path, output_val_path


def create_sample_data(output_dir: str = "data/raw") -> str:
    """Create sample data for testing the pipeline when Kaggle data is unavailable."""
    os.makedirs(output_dir, exist_ok=True)
    sample_data = [
        {
            "id": 0,
            "prompt": "Which of the following statements accurately describes the impact of Modified Newtonian Dynamics (MOND) on the observed rotation curves of low-surface-brightness galaxies?",
            "A": "MOND predicts a flat rotation curve for all galaxies, regardless of their mass distribution.",
            "B": "MOND suggests that the rotation curves of low-surface-brightness galaxies are consistent with Newtonian dynamics without any modification.",
            "C": "MOND accurately predicts the rotation curves of low-surface-brightness galaxies without the need for dark matter.",
            "D": "MOND predicts that low-surface-brightness galaxies have declining rotation curves similar to high-surface-brightness galaxies.",
            "E": "MOND predicts that the rotation curves of low-surface-brightness galaxies exhibit a sharp increase at large radii.",
            "answer": "C",
        },
        {
            "id": 1,
            "prompt": "In quantum mechanics, what does the Heisenberg uncertainty principle state?",
            "A": "The energy of a system is always conserved.",
            "B": "It is impossible to simultaneously know the exact position and momentum of a particle.",
            "C": "All particles exhibit wave-like behavior.",
            "D": "The spin of an electron can only be up or down.",
            "E": "Entangled particles always have opposite spins.",
            "answer": "B",
        },
        {
            "id": 2,
            "prompt": "What is the primary function of mitochondria in eukaryotic cells?",
            "A": "Protein synthesis",
            "B": "DNA replication",
            "C": "ATP production through cellular respiration",
            "D": "Lipid storage",
            "E": "Cell division regulation",
            "answer": "C",
        },
    ]
    path = os.path.join(output_dir, "train.csv")
    pd.DataFrame(sample_data).to_csv(path, index=False)
    print(f"Created sample data at {path}")
    return path
