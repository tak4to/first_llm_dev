"""Full pipeline orchestrator for Kaggle LLM Science Exam.

Runs all steps in sequence:
1. Data preparation
2. ChromaDB setup (RAG)
3. Fine-tuning
4. Inference & submission

Usage:
    # Full pipeline with Kaggle data
    uv run python scripts/run_pipeline.py --train-file data/raw/train.csv --test-file data/raw/test.csv

    # Test pipeline with sample data (no Kaggle data needed)
    uv run python scripts/run_pipeline.py --use-sample --no-rag --skip-finetune

    # Skip fine-tuning, use existing model with RAG
    uv run python scripts/run_pipeline.py --skip-finetune --model gpt-4o-mini-2024-07-18
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
from src.kaggle_llm.evaluation import evaluate_predictions
from src.kaggle_llm.inference import ScienceExamInference


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline")
    parser.add_argument("--use-sample", action="store_true", help="Use sample data")
    parser.add_argument("--train-file", type=str, default=None)
    parser.add_argument("--test-file", type=str, default=None)
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG")
    parser.add_argument("--skip-finetune", action="store_true", help="Skip fine-tuning")
    parser.add_argument("--model", type=str, default=None, help="Model to use")
    parser.add_argument(
        "--index-wikipedia",
        action="store_true",
        help="Index Wikipedia articles into ChromaDB",
    )
    parser.add_argument("--max-articles", type=int, default=1000)
    args = parser.parse_args()

    config = load_config()

    # =========================================================
    # Step 1: Data Preparation
    # =========================================================
    print("=" * 60)
    print("STEP 1: DATA PREPARATION")
    print("=" * 60)

    if args.use_sample:
        train_path = create_sample_data(config["data"]["raw_dir"])
    else:
        train_path = args.train_file or config["data"]["train_file"]

    train_df = load_competition_data(train_path)

    train_jsonl, val_jsonl = prepare_finetune_data(
        train_df,
        config["data"]["finetune_file"],
        config["data"]["validation_file"],
        val_split=config["finetune"]["validation_split"],
    )

    # =========================================================
    # Step 2: ChromaDB Setup (optional)
    # =========================================================
    if not args.no_rag:
        print("\n" + "=" * 60)
        print("STEP 2: CHROMADB RAG SETUP")
        print("=" * 60)

        from src.kaggle_llm.rag import ScienceRAG

        rag = ScienceRAG(config)

        # Index local files if directory exists
        wiki_dir = config["data"]["wikipedia_dir"]
        rag.index_from_text_files(wiki_dir)

        # Index Wikipedia if requested
        if args.index_wikipedia:
            rag.index_wikipedia_from_dataset(max_articles=args.max_articles)

        collection = rag.get_or_create_collection()
        print(f"ChromaDB documents: {collection.count()}")
    else:
        print("\n[Skipping RAG setup]")

    # =========================================================
    # Step 3: Fine-Tuning (optional)
    # =========================================================
    finetuned_model = None
    if not args.skip_finetune:
        print("\n" + "=" * 60)
        print("STEP 3: FINE-TUNING")
        print("=" * 60)

        from src.kaggle_llm.finetune import FineTuner

        finetuner = FineTuner(config)
        finetuned_model = finetuner.run_full_pipeline(train_jsonl, val_jsonl)
        print(f"Fine-tuned model: {finetuned_model}")
    else:
        print("\n[Skipping fine-tuning]")

    # =========================================================
    # Step 4: Inference
    # =========================================================
    print("\n" + "=" * 60)
    print("STEP 4: INFERENCE")
    print("=" * 60)

    if args.model:
        config["openai"]["finetuned_model"] = args.model
    elif finetuned_model:
        config["openai"]["finetuned_model"] = finetuned_model
    if args.no_rag:
        config["inference"]["use_rag"] = False

    engine = ScienceExamInference(config)
    print(f"Model: {engine.model}")
    print(f"RAG: {engine.use_rag}")

    # If test file exists, generate submission
    test_path = args.test_file or config["data"]["test_file"]
    test_file = Path(test_path)

    if test_file.exists():
        test_df = load_competition_data(test_path)
        submission = engine.generate_submission(
            test_df, config["data"]["submission_file"]
        )
        print(f"\nSubmission:")
        print(submission.head().to_string(index=False))

    # Evaluate on training data (validation split)
    print("\n--- Evaluating on training data ---")
    predictions = engine.predict_batch(train_df, use_ranking=True)
    results = evaluate_predictions(train_df, predictions)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
