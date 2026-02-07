"""Step 5: Evaluate predictions against ground truth.

Usage:
    uv run python scripts/05_evaluate.py --predictions PRED_CSV --ground-truth GT_CSV
    uv run python scripts/05_evaluate.py --submission data/submission.csv --ground-truth data/raw/train.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kaggle_llm.evaluation import evaluate_predictions, map_at_k


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument(
        "--submission",
        type=str,
        required=True,
        help="Path to submission CSV (id, prediction)",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        required=True,
        help="Path to ground truth CSV with 'answer' column",
    )
    args = parser.parse_args()

    # Load data
    submission = pd.read_csv(args.submission)
    ground_truth = pd.read_csv(args.ground_truth)

    # Merge on id
    merged = ground_truth.merge(submission, on="id", how="inner")
    if len(merged) == 0:
        print("Error: No matching IDs between submission and ground truth")
        sys.exit(1)

    print(f"Evaluating {len(merged)} questions")

    predictions = merged["prediction"].tolist()
    results = evaluate_predictions(merged, predictions)

    # Detailed breakdown
    print(f"\nPer-question breakdown:")
    for _, row in merged.iterrows():
        pred_list = row["prediction"].split()
        true = row["answer"]
        correct = "OK" if pred_list[0] == true else "  "
        print(f"  [{correct}] id={row['id']}: true={true}, pred={row['prediction']}")


if __name__ == "__main__":
    main()
