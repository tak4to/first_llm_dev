"""OpenAI fine-tuning module for Kaggle LLM Science Exam.

Handles:
1. Uploading training data to OpenAI
2. Creating and monitoring fine-tuning jobs
3. Retrieving fine-tuned model information
"""

import json
import time

from openai import OpenAI

from .config import get_openai_api_key, load_config


class FineTuner:
    """Manages OpenAI fine-tuning for science question answering."""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()

        self.client = OpenAI(api_key=get_openai_api_key())
        self.base_model = config["openai"]["model"]
        self.ft_config = config["finetune"]

    def validate_data(self, filepath: str) -> bool:
        """Validate JSONL file format for OpenAI fine-tuning."""
        errors = []
        with open(filepath) as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    if "messages" not in data:
                        errors.append(f"Line {i}: missing 'messages' key")
                        continue
                    messages = data["messages"]
                    roles = [m["role"] for m in messages]
                    if roles != ["system", "user", "assistant"]:
                        errors.append(
                            f"Line {i}: expected roles "
                            "['system', 'user', 'assistant'], got {roles}"
                        )
                except json.JSONDecodeError:
                    errors.append(f"Line {i}: invalid JSON")

        if errors:
            print("Validation errors:")
            for e in errors:
                print(f"  {e}")
            return False
        print(f"Validation passed: {filepath}")
        return True

    def upload_file(self, filepath: str, purpose: str = "fine-tune") -> str:
        """Upload a file to OpenAI for fine-tuning.

        Returns:
            The file ID.
        """
        with open(filepath, "rb") as f:
            response = self.client.files.create(file=f, purpose=purpose)
        print(f"Uploaded {filepath} -> file ID: {response.id}")
        return response.id

    def create_finetune_job(
        self,
        train_file_id: str,
        val_file_id: str | None = None,
    ) -> str:
        """Create an OpenAI fine-tuning job.

        Args:
            train_file_id: OpenAI file ID for training data.
            val_file_id: Optional OpenAI file ID for validation data.

        Returns:
            The fine-tuning job ID.
        """
        params = {
            "training_file": train_file_id,
            "model": self.base_model,
            "suffix": self.ft_config["suffix"],
            "hyperparameters": {
                "n_epochs": self.ft_config["n_epochs"],
                "batch_size": self.ft_config["batch_size"],
                "learning_rate_multiplier": self.ft_config["learning_rate_multiplier"],
            },
        }
        if val_file_id:
            params["validation_file"] = val_file_id

        job = self.client.fine_tuning.jobs.create(**params)
        print(f"Created fine-tuning job: {job.id}")
        print(f"  Model: {self.base_model}")
        print(f"  Status: {job.status}")
        return job.id

    def wait_for_completion(
        self, job_id: str, poll_interval: int = 60
    ) -> dict:
        """Wait for a fine-tuning job to complete.

        Args:
            job_id: The fine-tuning job ID.
            poll_interval: Seconds between status checks.

        Returns:
            Dict with job details including fine_tuned_model name.
        """
        print(f"Waiting for fine-tuning job {job_id} to complete...")

        while True:
            job = self.client.fine_tuning.jobs.retrieve(job_id)
            status = job.status
            print(f"  Status: {status}")

            if status == "succeeded":
                model_name = job.fine_tuned_model
                print(f"Fine-tuning complete! Model: {model_name}")
                return {
                    "job_id": job_id,
                    "status": status,
                    "fine_tuned_model": model_name,
                    "trained_tokens": job.trained_tokens,
                }

            if status in ("failed", "cancelled"):
                error_msg = getattr(job, "error", "Unknown error")
                raise RuntimeError(
                    f"Fine-tuning job {status}: {error_msg}"
                )

            time.sleep(poll_interval)

    def list_events(self, job_id: str, limit: int = 20) -> list[dict]:
        """List events for a fine-tuning job."""
        events = self.client.fine_tuning.jobs.list_events(
            fine_tuning_job_id=job_id, limit=limit
        )
        result = []
        for event in events.data:
            result.append(
                {
                    "created_at": event.created_at,
                    "level": event.level,
                    "message": event.message,
                }
            )
        return result

    def list_jobs(self, limit: int = 10) -> list[dict]:
        """List recent fine-tuning jobs."""
        jobs = self.client.fine_tuning.jobs.list(limit=limit)
        result = []
        for job in jobs.data:
            result.append(
                {
                    "id": job.id,
                    "model": job.model,
                    "status": job.status,
                    "fine_tuned_model": job.fine_tuned_model,
                    "created_at": job.created_at,
                }
            )
        return result

    def run_full_pipeline(
        self, train_path: str, val_path: str
    ) -> str:
        """Run the complete fine-tuning pipeline.

        Args:
            train_path: Path to training JSONL file.
            val_path: Path to validation JSONL file.

        Returns:
            The fine-tuned model name.
        """
        # Validate
        print("=== Validating training data ===")
        if not self.validate_data(train_path):
            raise ValueError("Training data validation failed")
        if not self.validate_data(val_path):
            raise ValueError("Validation data validation failed")

        # Upload
        print("\n=== Uploading files ===")
        train_file_id = self.upload_file(train_path)
        val_file_id = self.upload_file(val_path)

        # Create job
        print("\n=== Creating fine-tuning job ===")
        job_id = self.create_finetune_job(train_file_id, val_file_id)

        # Wait
        print("\n=== Waiting for completion ===")
        result = self.wait_for_completion(job_id)

        return result["fine_tuned_model"]
