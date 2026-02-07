"""Kaggle LLM Science Exam - Pipeline CLI.

Usage:
    python scripts/main.py prepare   # Augment data with ChromaDB context
    python scripts/main.py train     # QLoRA fine-tuning
    python scripts/main.py predict   # Generate submission.csv
    python scripts/main.py evaluate  # Validate on training split
    python scripts/main.py all       # Run prepare -> train -> predict
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path so `src` package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Kaggle LLM Science Exam - Fine-tuning & Inference Pipeline"
    )
    parser.add_argument(
        "mode",
        choices=["prepare", "train", "predict", "evaluate", "all"],
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--model", type=str, default=None, help="HuggingFace model name/path"
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data", help="Data directory"
    )
    parser.add_argument(
        "--chroma-dir", type=str, default="./chroma_db", help="ChromaDB directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./outputs", help="Output directory"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models/finetuned",
        help="Model output directory",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path to LoRA adapter (for predict/evaluate)",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Per-device batch size"
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--max-seq-length", type=int, default=768, help="Max sequence length"
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of RAG documents to retrieve"
    )

    args = parser.parse_args()

    # Build config
    from src.config import Config

    config = Config(
        data_dir=args.data_dir,
        chroma_dir=args.chroma_dir,
        output_dir=args.output_dir,
        model_output_dir=args.model_dir,
        retrieval_top_k=args.top_k,
        num_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
    )
    if args.model:
        config.model_name = args.model

    # Execute requested stage(s)
    if args.mode in ("prepare", "all"):
        print("=" * 60)
        print("Stage: PREPARE (augment data with RAG context)")
        print("=" * 60)
        from src.data_prep import prepare_training_data

        prepare_training_data(config)

    if args.mode in ("train", "all"):
        print("=" * 60)
        print("Stage: TRAIN (QLoRA fine-tuning)")
        print("=" * 60)
        from src.finetune import train

        train(config)

    if args.mode in ("predict", "all"):
        print("=" * 60)
        print("Stage: PREDICT (generate submission)")
        print("=" * 60)
        from src.inference import predict

        predict(config, args.adapter_path)

    if args.mode == "evaluate":
        print("=" * 60)
        print("Stage: EVALUATE (validation metrics)")
        print("=" * 60)
        from src.inference import evaluate

        evaluate(config, args.adapter_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
