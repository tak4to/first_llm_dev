"""Step 2: Set up ChromaDB with Wikipedia science articles for RAG.

Usage:
    uv run python scripts/02_setup_chromadb.py [--source {wikipedia,local,both}] [--max-articles 1000]

Sources:
    wikipedia: Download and index from HuggingFace Wikipedia dataset
    local: Index text files from data/wikipedia/ directory
    both: Use both sources
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kaggle_llm.config import load_config
from src.kaggle_llm.rag import ScienceRAG


def main():
    parser = argparse.ArgumentParser(description="Set up ChromaDB for RAG")
    parser.add_argument(
        "--source",
        choices=["wikipedia", "local", "both"],
        default="both",
        help="Source of documents to index",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=1000,
        help="Maximum Wikipedia articles to index (for wikipedia source)",
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default=None,
        help="Directory containing text files for local indexing",
    )
    parser.add_argument(
        "--test-query",
        type=str,
        default=None,
        help="Run a test query after indexing",
    )
    args = parser.parse_args()

    config = load_config()
    rag = ScienceRAG(config)

    # Index from local text files
    if args.source in ("local", "both"):
        local_dir = args.local_dir or config["data"]["wikipedia_dir"]
        print(f"\n=== Indexing local text files from {local_dir} ===")
        rag.index_from_text_files(local_dir)

    # Index from Wikipedia dataset
    if args.source in ("wikipedia", "both"):
        print(f"\n=== Indexing Wikipedia articles (max: {args.max_articles}) ===")
        rag.index_wikipedia_from_dataset(max_articles=args.max_articles)

    # Check collection status
    collection = rag.get_or_create_collection()
    print(f"\n=== ChromaDB Status ===")
    print(f"Collection: {collection.name}")
    print(f"Total documents: {collection.count()}")

    # Test query
    if args.test_query:
        print(f"\n=== Test Query: '{args.test_query}' ===")
        results = rag.retrieve(args.test_query, n_results=3)
        for i, doc in enumerate(results, 1):
            print(f"\n[{i}] {doc[:200]}...")

    print(f"\nNext step: Run scripts/03_finetune.py to start fine-tuning")


if __name__ == "__main__":
    main()
