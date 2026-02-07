"""Evaluation module for Kaggle LLM Science Exam.

Implements MAP@3 (Mean Average Precision at 3) metric.
"""

import numpy as np
import pandas as pd


def average_precision_at_k(true_label: str, predictions: list[str], k: int = 3) -> float:
    """Calculate Average Precision at K for a single question.

    Args:
        true_label: The correct answer (e.g., 'C').
        predictions: List of predicted answers in ranked order.
        k: Number of predictions to consider.

    Returns:
        AP@K score (0.0 or 1/position if correct answer found).
    """
    predictions = predictions[:k]

    for i, pred in enumerate(predictions):
        if pred.strip().upper() == true_label.strip().upper():
            return 1.0 / (i + 1)

    return 0.0


def map_at_k(true_labels: list[str], prediction_strings: list[str], k: int = 3) -> float:
    """Calculate Mean Average Precision at K.

    Args:
        true_labels: List of correct answers.
        prediction_strings: List of space-separated prediction strings (e.g., 'C A B').
        k: Number of predictions per question.

    Returns:
        MAP@K score.
    """
    scores = []
    for true_label, pred_str in zip(true_labels, prediction_strings):
        predictions = pred_str.strip().split()
        ap = average_precision_at_k(true_label, predictions, k)
        scores.append(ap)

    return float(np.mean(scores))


def evaluate_predictions(
    df: pd.DataFrame,
    predictions: list[str],
    answer_col: str = "answer",
) -> dict:
    """Evaluate predictions against ground truth.

    Args:
        df: DataFrame with ground truth answers.
        predictions: List of space-separated prediction strings.
        answer_col: Column name for ground truth answers.

    Returns:
        Dict with evaluation metrics.
    """
    true_labels = df[answer_col].tolist()
    map3 = map_at_k(true_labels, predictions, k=3)

    # Also compute accuracy (top-1)
    top1_correct = sum(
        1
        for true, pred_str in zip(true_labels, predictions)
        if pred_str.strip().split()[0].upper() == true.strip().upper()
    )
    accuracy = top1_correct / len(true_labels)

    results = {
        "map@3": map3,
        "accuracy_top1": accuracy,
        "total_questions": len(true_labels),
    }

    print(f"Evaluation Results:")
    print(f"  MAP@3:         {map3:.4f}")
    print(f"  Top-1 Accuracy: {accuracy:.4f}")
    print(f"  Total Questions: {len(true_labels)}")

    return results
