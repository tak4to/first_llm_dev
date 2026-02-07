"""Data preparation for Kaggle LLM Science Exam.

Loads competition CSV data, enriches each question with
relevant Wikipedia context from ChromaDB, and saves the
prepared dataset as JSON for fine-tuning and inference.
"""

import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

from src.config import Config
from src.retriever import ChromaDBRetriever

OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def load_competition_data(data_dir: str) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load train.csv and test.csv from the Kaggle data directory."""
    train_path = Path(data_dir) / "train.csv"
    test_path = Path(data_dir) / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"train.csv not found at {train_path}.\n"
            f"Please download from: "
            f"https://www.kaggle.com/competitions/kaggle-llm-science-exam/data\n"
            f"and place the files in {data_dir}/"
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if test_path.exists() else None

    print(f"Loaded train.csv: {len(train_df)} rows")
    if test_df is not None:
        print(f"Loaded test.csv: {len(test_df)} rows")

    return train_df, test_df


def format_prompt(
    question: str, options: Dict[str, str], context: str = ""
) -> str:
    """Format a multiple-choice question into a prompt string.

    The prompt includes optional RAG context, the question, and
    all answer options. It ends with a cue for the model to output
    the correct answer letter.
    """
    parts = []

    if context:
        parts.append(f"Context:\n{context}\n")

    parts.append(f"Question: {question}")

    for letter in OPTION_LETTERS:
        if letter in options and pd.notna(options[letter]):
            parts.append(f"{letter}) {options[letter]}")

    parts.append("\nThe correct answer is:")

    return "\n".join(parts)


def _build_row_options(row: pd.Series) -> Dict[str, str]:
    """Extract answer options from a DataFrame row."""
    return {
        letter: str(row[letter])
        for letter in OPTION_LETTERS
        if letter in row and pd.notna(row[letter])
    }


def prepare_training_data(config: Config) -> Tuple[List[Dict], List[Dict]]:
    """Prepare training and test data augmented with RAG context.

    For each question:
    1. Retrieves relevant Wikipedia passages from ChromaDB
    2. Formats a prompt with context + question + options
    3. Saves results as JSON files in config.output_dir
    """
    train_df, test_df = load_competition_data(config.data_dir)

    # Initialize ChromaDB retriever
    retriever = None
    try:
        retriever = ChromaDBRetriever(
            chroma_dir=config.chroma_dir,
            collection_name=config.collection_name,
            embedding_model_name=config.embedding_model,
            top_k=config.retrieval_top_k,
        )
    except Exception as e:
        print(f"Warning: ChromaDB not available ({e}). Proceeding without RAG context.")

    # --- Prepare training data ---
    print("Preparing training data...")
    train_data = []
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Train"):
        question = str(row["prompt"])
        options = _build_row_options(row)
        answer = str(row["answer"]).strip()

        context = ""
        if retriever is not None:
            try:
                option_texts = list(options.values())
                context = retriever.retrieve_for_question(question, option_texts)
            except Exception:
                pass

        prompt = format_prompt(question, options, context)

        train_data.append(
            {
                "id": row.get("id", idx),
                "prompt": prompt,
                "answer": f" {answer}",
                "question": question,
                "options": options,
                "context": context,
            }
        )

    # --- Prepare test data ---
    test_data = []
    if test_df is not None:
        print("Preparing test data...")
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Test"):
            question = str(row["prompt"])
            options = _build_row_options(row)

            context = ""
            if retriever is not None:
                try:
                    option_texts = list(options.values())
                    context = retriever.retrieve_for_question(question, option_texts)
                except Exception:
                    pass

            prompt = format_prompt(question, options, context)

            test_data.append(
                {
                    "id": row["id"],
                    "prompt": prompt,
                    "question": question,
                    "options": options,
                    "context": context,
                }
            )

    # --- Save prepared data ---
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_output = output_dir / "train_prepared.json"
    with open(train_output, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(train_data)} training examples -> {train_output}")

    if test_data:
        test_output = output_dir / "test_prepared.json"
        with open(test_output, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(test_data)} test examples -> {test_output}")

    return train_data, test_data


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import Config

    config = Config()
    prepare_training_data(config)
