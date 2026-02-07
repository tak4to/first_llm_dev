"""Configuration loader for the Kaggle LLM Science Exam project."""

import os
from pathlib import Path

import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration file."""
    project_root = Path(__file__).parent.parent.parent
    full_path = project_root / config_path
    with open(full_path) as f:
        config = yaml.safe_load(f)
    return config


def get_openai_api_key() -> str:
    """Get OpenAI API key from environment variable."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it: export OPENAI_API_KEY=sk-..."
        )
    return api_key
