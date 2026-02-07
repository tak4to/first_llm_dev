"""Step 3: Fine-tune an OpenAI model on science exam data.

Usage:
    uv run python scripts/03_finetune.py [--wait] [--list-jobs]

Requires OPENAI_API_KEY environment variable.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kaggle_llm.config import load_config
from src.kaggle_llm.finetune import FineTuner


def main():
    parser = argparse.ArgumentParser(description="Fine-tune OpenAI model")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for fine-tuning to complete (can take hours)",
    )
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="List recent fine-tuning jobs and exit",
    )
    parser.add_argument(
        "--check-job",
        type=str,
        default=None,
        help="Check status of a specific job ID",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default=None,
        help="Override training JSONL file path",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default=None,
        help="Override validation JSONL file path",
    )
    args = parser.parse_args()

    config = load_config()
    finetuner = FineTuner(config)

    # List jobs mode
    if args.list_jobs:
        print("=== Recent Fine-Tuning Jobs ===")
        jobs = finetuner.list_jobs()
        for job in jobs:
            print(f"  {job['id']}: {job['status']} -> {job['fine_tuned_model']}")
        return

    # Check job mode
    if args.check_job:
        print(f"=== Job {args.check_job} Events ===")
        events = finetuner.list_events(args.check_job)
        for event in events:
            print(f"  [{event['level']}] {event['message']}")
        return

    # Fine-tuning mode
    data_config = config["data"]
    train_path = args.train_file or data_config["finetune_file"]
    val_path = args.val_file or data_config["validation_file"]

    # Validate data
    print("=== Validating training data ===")
    if not finetuner.validate_data(train_path):
        print("Training data validation failed!")
        sys.exit(1)
    if not finetuner.validate_data(val_path):
        print("Validation data validation failed!")
        sys.exit(1)

    # Upload files
    print("\n=== Uploading files to OpenAI ===")
    train_file_id = finetuner.upload_file(train_path)
    val_file_id = finetuner.upload_file(val_path)

    # Create fine-tuning job
    print("\n=== Creating fine-tuning job ===")
    job_id = finetuner.create_finetune_job(train_file_id, val_file_id)

    # Save job ID for later reference
    job_info = {
        "job_id": job_id,
        "train_file_id": train_file_id,
        "val_file_id": val_file_id,
        "base_model": config["openai"]["model"],
    }
    job_info_path = Path("data/processed/finetune_job.json")
    job_info_path.parent.mkdir(parents=True, exist_ok=True)
    with open(job_info_path, "w") as f:
        json.dump(job_info, f, indent=2)
    print(f"Job info saved to {job_info_path}")

    if args.wait:
        print("\n=== Waiting for completion ===")
        result = finetuner.wait_for_completion(job_id)
        print(f"\nFine-tuned model: {result['fine_tuned_model']}")
        print(
            "Update configs/config.yaml with the fine-tuned model name:"
        )
        print(
            f'  finetuned_model: "{result["fine_tuned_model"]}"'
        )
    else:
        print(f"\nJob submitted: {job_id}")
        print("Run with --wait to wait for completion, or check status with:")
        print(f"  uv run python scripts/03_finetune.py --check-job {job_id}")

    print(f"\nNext step: Run scripts/04_inference.py to generate predictions")


if __name__ == "__main__":
    main()
