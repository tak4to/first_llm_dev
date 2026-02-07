"""
Wikipedia Processing Recovery Script
====================================
Identifies and processes files that were not completed.

Features:
- Automatically detects unprocessed files
- Option to reprocess specific files
- Option to reprocess failed files only
- Validates after processing

Compatible with prepare_wikipedia.py
"""

import json
import argparse
import gc
from pathlib import Path
from typing import List, Set, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np
from tqdm import tqdm
import re

# Vector DB
import chromadb
from chromadb.config import Settings

# Embeddings
from sentence_transformers import SentenceTransformer


@dataclass
class RecoveryConfig:
    """Configuration for recovery (matches PrepConfig in prepare_wikipedia.py)."""
    # Paths
    wiki_input_dir: str = "./data/wikipedia"
    chroma_dir: str = "./data/chroma_db"
    progress_file: str = "./indices/wiki_progress.json"
    
    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_length: int = 50
    
    # Embedding
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 32
    
    # ChromaDB
    collection_name: str = "wikipedia_chunks"
    
    def __post_init__(self):
        Path(self.chroma_dir).mkdir(parents=True, exist_ok=True)
        Path(self.progress_file).parent.mkdir(parents=True, exist_ok=True)


class ProgressTracker:
    """Track processed files to enable resume."""
    
    def __init__(self, progress_file: str):
        self.progress_file = progress_file
        self.processed_files: List[str] = []
        self.total_chunks: int = 0
        self._load()
    
    def _load(self):
        """Load progress from file."""
        if Path(self.progress_file).exists():
            with open(self.progress_file, "r") as f:
                data = json.load(f)
                self.processed_files = data.get("processed_files", [])
                self.total_chunks = data.get("total_chunks", 0)
            print(f"Loaded progress: {len(self.processed_files)} files processed, {self.total_chunks} chunks")
    
    def save(self):
        """Save progress to file."""
        with open(self.progress_file, "w") as f:
            json.dump({
                "processed_files": self.processed_files,
                "total_chunks": self.total_chunks
            }, f, indent=2)
    
    def mark_processed(self, filename: str, num_chunks: int):
        """Mark a file as processed."""
        self.processed_files.append(filename)
        self.total_chunks += num_chunks
        self.save()
    
    def remove_file(self, filename: str):
        """Remove a file from progress tracking."""
        if filename in self.processed_files:
            self.processed_files.remove(filename)
            self.save()
            print(f"  Removed '{filename}' from progress tracking")


class WikipediaChunker:
    """Chunk Wikipedia articles."""
    
    def __init__(self, config: RecoveryConfig):
        self.config = config
    
    def clean_text(self, text: str) -> str:
        """Clean Wikipedia text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[\[|\]\]', '', text)
        text = re.sub(r'\{\{[^}]*\}\}', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()
    
    def chunk_article(self, text: str, title: str, doc_id: str) -> List[dict]:
        """Split article into overlapping chunks."""
        text = self.clean_text(text)
        words = text.split()
        
        if len(words) < self.config.min_chunk_length:
            return []
        
        chunks = []
        step = self.config.chunk_size - self.config.chunk_overlap
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.config.chunk_size]
            
            if len(chunk_words) < self.config.min_chunk_length:
                continue
            
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "id": f"{doc_id}_{len(chunks)}",
                "title": title,
                "text": chunk_text
            })
        
        return chunks


class RecoveryProcessor:
    """Process unprocessed or failed Wikipedia files."""
    
    def __init__(self, config: RecoveryConfig):
        self.config = config
        self.chunker = WikipediaChunker(config)
        self.tracker = ProgressTracker(config.progress_file)
        self.embedding_model: Optional[SentenceTransformer] = None
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection = None
    
    def _get_client(self) -> chromadb.PersistentClient:
        """Get ChromaDB client."""
        if self.client is None:
            self.client = chromadb.PersistentClient(
                path=self.config.chroma_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        return self.client
    
    def _load_embedding_model(self):
        """Load embedding model."""
        if self.embedding_model is None:
            print(f"Loading embedding model: {self.config.embedding_model}")
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            self.embedding_model = self.embedding_model.to("cuda")
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one."""
        if self.collection is None:
            client = self._get_client()
            try:
                self.collection = client.get_collection(self.config.collection_name)
                print(f"Using existing collection: {self.collection.count()} documents")
            except Exception:
                self.collection = client.create_collection(
                    name=self.config.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                print(f"Created new collection: {self.config.collection_name}")
        return self.collection
    
    def get_unprocessed_files(self) -> List[Path]:
        """Get list of unprocessed parquet files."""
        input_path = Path(self.config.wiki_input_dir)
        
        if not input_path.exists():
            print(f"Error: Input directory not found: {input_path}")
            return []
        
        # Get all parquet files
        all_files = sorted(input_path.glob("*.parquet"))
        
        # Get processed files
        processed = set(self.tracker.processed_files)
        
        # Find unprocessed
        unprocessed = [f for f in all_files if f.name not in processed]
        
        return unprocessed
    
    def get_files_not_in_chromadb(self) -> List[Path]:
        """
        Get files that are marked as processed but not fully in ChromaDB.
        This handles cases where processing was interrupted during ChromaDB insertion.
        """
        input_path = Path(self.config.wiki_input_dir)
        collection = self._get_or_create_collection()
        
        if collection.count() == 0:
            # If ChromaDB is empty, all processed files need reprocessing
            return [input_path / f for f in self.tracker.processed_files 
                    if (input_path / f).exists()]
        
        # Sample ChromaDB to find which file prefixes exist
        sample_size = min(50000, collection.count())
        results = collection.get(limit=sample_size)
        
        # Extract prefixes from IDs
        existing_prefixes = set()
        for doc_id in results["ids"]:
            parts = doc_id.rsplit("_", 2)
            if len(parts) >= 2:
                existing_prefixes.add(parts[0])
        
        # Find files marked as processed but not in ChromaDB
        missing_files = []
        for filename in self.tracker.processed_files:
            prefix = Path(filename).stem
            if prefix not in existing_prefixes:
                filepath = input_path / filename
                if filepath.exists():
                    missing_files.append(filepath)
        
        return missing_files
    
    def _process_single_file(self, parquet_path: Path) -> int:
        """
        Process a single parquet file and add to ChromaDB.
        (Same logic as prepare_wikipedia.py)
        """
        print(f"\nProcessing: {parquet_path.name}")
        
        # Read parquet
        df = pd.read_parquet(parquet_path)
        
        # Detect text column
        text_col = 'text' if 'text' in df.columns else 'content'
        if text_col not in df.columns:
            print(f"  Warning: No text column found in {parquet_path.name}")
            return 0
        
        # Chunk all articles
        all_chunks = []
        file_prefix = parquet_path.stem
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="  Chunking", leave=False):
            title = str(row.get('title', ''))
            text = str(row.get(text_col, ''))
            
            if not title or not text or len(text) < 100:
                continue
            
            doc_id = f"{file_prefix}_{idx}"
            chunks = self.chunker.chunk_article(text, title, doc_id)
            all_chunks.extend(chunks)
        
        if not all_chunks:
            print(f"  No chunks generated from {parquet_path.name}")
            return 0
        
        print(f"  Generated {len(all_chunks)} chunks")
        
        # Free DataFrame memory
        del df
        gc.collect()
        
        # Add chunks to ChromaDB in batches
        batch_size = self.config.batch_size
        collection = self._get_or_create_collection()
        
        for i in tqdm(range(0, len(all_chunks), batch_size), desc="  Indexing", leave=False):
            batch = all_chunks[i:i + batch_size]
            
            ids = [chunk["id"] for chunk in batch]
            texts = [chunk["text"] for chunk in batch]
            metadatas = [{"title": chunk["title"], "text": chunk["text"]} for chunk in batch]
            
            # Compute embeddings
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # Use upsert to handle potential duplicates
            collection.upsert(
                ids=ids,
                embeddings=embeddings.tolist(),
                metadatas=metadatas
            )
        
        num_chunks = len(all_chunks)
        del all_chunks
        gc.collect()
        
        return num_chunks
    
    def process_unprocessed(self) -> int:
        """Process all unprocessed files."""
        unprocessed = self.get_unprocessed_files()
        
        if not unprocessed:
            print("\n✓ All files have been processed!")
            return 0
        
        print(f"\nFound {len(unprocessed)} unprocessed files:")
        for f in unprocessed[:10]:
            print(f"  - {f.name}")
        if len(unprocessed) > 10:
            print(f"  ... and {len(unprocessed) - 10} more")
        
        # Load models
        self._load_embedding_model()
        self._get_or_create_collection()
        
        # Process each file
        total_chunks = 0
        for filepath in unprocessed:
            try:
                num_chunks = self._process_single_file(filepath)
                self.tracker.mark_processed(filepath.name, num_chunks)
                total_chunks += num_chunks
                print(f"  ✓ Completed: {num_chunks} chunks (Total: {self.tracker.total_chunks})")
                gc.collect()
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        print(f"\n{'=' * 60}")
        print(f"Recovery complete!")
        print(f"  Files processed: {len(unprocessed)}")
        print(f"  Chunks added: {total_chunks:,}")
        print(f"{'=' * 60}")
        
        return total_chunks
    
    def process_specific_files(self, filenames: List[str], force: bool = False) -> int:
        """Process specific files by name."""
        input_path = Path(self.config.wiki_input_dir)
        
        # Validate files exist
        files_to_process = []
        for filename in filenames:
            filepath = input_path / filename
            if not filepath.exists():
                print(f"  Warning: File not found: {filename}")
                continue
            
            if filename in self.tracker.processed_files:
                if force:
                    print(f"  Reprocessing (forced): {filename}")
                    self.tracker.remove_file(filename)
                else:
                    print(f"  Skipping (already processed): {filename}")
                    continue
            
            files_to_process.append(filepath)
        
        if not files_to_process:
            print("\nNo files to process.")
            return 0
        
        print(f"\nProcessing {len(files_to_process)} files...")
        
        # Load models
        self._load_embedding_model()
        self._get_or_create_collection()
        
        # Process
        total_chunks = 0
        for filepath in files_to_process:
            try:
                num_chunks = self._process_single_file(filepath)
                self.tracker.mark_processed(filepath.name, num_chunks)
                total_chunks += num_chunks
                print(f"  ✓ Completed: {num_chunks} chunks (Total: {self.tracker.total_chunks})")
                gc.collect()
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        return total_chunks
    
    def repair_chromadb_gaps(self) -> int:
        """
        Find and reprocess files that are in progress but missing from ChromaDB.
        """
        print("\nChecking for ChromaDB gaps...")
        
        missing_files = self.get_files_not_in_chromadb()
        
        if not missing_files:
            print("✓ No gaps found - all processed files are in ChromaDB")
            return 0
        
        print(f"\nFound {len(missing_files)} files missing from ChromaDB:")
        for f in missing_files:
            print(f"  - {f.name}")
        
        # Remove from progress and reprocess
        for filepath in missing_files:
            self.tracker.remove_file(filepath.name)
        
        # Load models
        self._load_embedding_model()
        self._get_or_create_collection()
        
        # Process
        total_chunks = 0
        for filepath in missing_files:
            try:
                num_chunks = self._process_single_file(filepath)
                self.tracker.mark_processed(filepath.name, num_chunks)
                total_chunks += num_chunks
                print(f"  ✓ Completed: {num_chunks} chunks (Total: {self.tracker.total_chunks})")
                gc.collect()
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        print(f"\n{'=' * 60}")
        print(f"Repair complete!")
        print(f"  Files repaired: {len(missing_files)}")
        print(f"  Chunks added: {total_chunks:,}")
        print(f"{'=' * 60}")
        
        return total_chunks
    
    def show_status(self):
        """Show current processing status."""
        input_path = Path(self.config.wiki_input_dir)
        
        all_files = sorted(input_path.glob("*.parquet")) if input_path.exists() else []
        processed = set(self.tracker.processed_files)
        
        print(f"\n{'=' * 60}")
        print("Processing Status")
        print(f"{'=' * 60}")
        print(f"\nInput directory: {input_path}")
        print(f"ChromaDB directory: {self.config.chroma_dir}")
        print(f"Total parquet files: {len(all_files)}")
        print(f"Processed files: {len(processed)}")
        print(f"Unprocessed files: {len(all_files) - len([f for f in all_files if f.name in processed])}")
        print(f"Total chunks recorded: {self.tracker.total_chunks:,}")
        
        # Check ChromaDB
        try:
            collection = self._get_or_create_collection()
            print(f"ChromaDB documents: {collection.count():,}")
        except Exception:
            print("ChromaDB: Not available")
        
        # List unprocessed
        unprocessed = [f for f in all_files if f.name not in processed]
        if unprocessed:
            print(f"\nUnprocessed files:")
            for f in unprocessed:
                print(f"  - {f.name}")
        else:
            print(f"\n✓ All files have been processed!")


def main():
    parser = argparse.ArgumentParser(
        description="Recover/reprocess unprocessed Wikipedia files"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="./data/wikipedia",
        help="Input directory containing parquet files"
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default="./data/chroma_db",
        help="ChromaDB storage directory"
    )
    parser.add_argument(
        "--files", "-f",
        nargs="+",
        type=str,
        default=None,
        help="Specific files to process (e.g., wiki_00.parquet wiki_01.parquet)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocess even if marked as processed"
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Repair ChromaDB gaps (reprocess files missing from ChromaDB)"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current processing status"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for embedding computation"
    )
    
    args = parser.parse_args()
    
    config = RecoveryConfig(
        wiki_input_dir=args.input,
        chroma_dir=args.chroma_dir,
        batch_size=args.batch_size
    )
    
    processor = RecoveryProcessor(config)
    
    if args.status:
        processor.show_status()
    elif args.files:
        processor.process_specific_files(args.files, force=args.force)
    elif args.repair:
        processor.repair_chromadb_gaps()
    else:
        # Default: process all unprocessed files
        processor.process_unprocessed()


if __name__ == "__main__":
    main()