"""Step 4: Run inference and generate submission.

Usage:
    uv run python scripts/04_inference.py [--model MODEL] [--no-rag]

Requires OPENAI_API_KEY environment variable.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kaggle_llm.config import load_config
from src.kaggle_llm.data_preparation import load_competition_data
from src.kaggle_llm.evaluation import evaluate_predictions
from src.kaggle_llm.inference import ScienceExamInference


def main():
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name (e.g., fine-tuned model ID)",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG context retrieval",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="Path to test CSV file",
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        default=None,
        help="Evaluate on a file with ground truth answers",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output submission file path",
    )
    args = parser.parse_args()

    config = load_config()

    # Override config with CLI args
    if args.model:
        config["openai"]["finetuned_model"] = args.model
    if args.no_rag:
        config["inference"]["use_rag"] = False

    engine = ScienceExamInference(config)
    print(f"Using model: {engine.model}")
    print(f"RAG enabled: {engine.use_rag}")

    # Evaluation mode: run on data with ground truth
    if args.eval_file:
        print(f"\n=== Evaluation Mode ===")
        eval_df = load_competition_data(args.eval_file)
        predictions = engine.predict_batch(eval_df, use_ranking=True)
        results = evaluate_predictions(eval_df, predictions)
        return

    # Inference mode: generate submission
    test_path = args.test_file or config["data"]["test_file"]
    output_path = args.output or config["data"]["submission_file"]

    print(f"\n=== Generating Submission ===")
    test_df = load_competition_data(test_path)
    submission = engine.generate_submission(test_df, output_path)
    print(f"\nSubmission preview:")
    print(submission.head(10).to_string(index=False))
    print(f"\nSubmission saved to: {output_path}")


if __name__ == "__main__":
    main()
