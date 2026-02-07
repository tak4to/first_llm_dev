"""Step 1: Prepare training data for OpenAI fine-tuning.

Usage:
    uv run python scripts/01_prepare_data.py [--use-sample]

If --use-sample is passed, creates sample data for testing the pipeline.
Otherwise, expects Kaggle competition data in data/raw/train.csv.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kaggle_llm.config import load_config
from src.kaggle_llm.data_preparation import (
    create_sample_data,
    load_competition_data,
    prepare_finetune_data,
)


def main():
    parser = argparse.ArgumentParser(description="Prepare fine-tuning data")
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="Use sample data for testing (no Kaggle data needed)",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default=None,
        help="Path to training CSV file",
    )
    args = parser.parse_args()

    config = load_config()
    data_config = config["data"]
    ft_config = config["finetune"]

    # Load or create data
    if args.use_sample:
        print("=== Creating sample data ===")
        train_path = create_sample_data(data_config["raw_dir"])
    else:
        train_path = args.train_file or data_config["train_file"]

    print(f"\n=== Loading training data from {train_path} ===")
    train_df = load_competition_data(train_path)
    print(f"\nSample question:")
    print(f"  Prompt: {train_df.iloc[0]['prompt'][:100]}...")
    print(f"  Answer: {train_df.iloc[0]['answer']}")

    # Create fine-tuning JSONL files
    print("\n=== Creating fine-tuning data ===")
    train_jsonl, val_jsonl = prepare_finetune_data(
        train_df,
        data_config["finetune_file"],
        data_config["validation_file"],
        val_split=ft_config["validation_split"],
    )

    print(f"\n=== Done ===")
    print(f"Training JSONL: {train_jsonl}")
    print(f"Validation JSONL: {val_jsonl}")
    print(f"\nNext step: Run scripts/02_setup_chromadb.py to index Wikipedia data")


if __name__ == "__main__":
    main()
