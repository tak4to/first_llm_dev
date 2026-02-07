"""
Wikipedia Data Preparation for Kaggle LLM Science Exam
======================================================
Memory-efficient version: Process one parquet file at a time
and incrementally add to ChromaDB.

Features:
- Process files one by one to avoid OOM
- Resume from where it stopped (tracks processed files)
- Direct insertion to ChromaDB (no intermediate JSON)
- Progress tracking and logging
"""

import json
import os
import re
import gc
from pathlib import Path
from typing import List, Dict, Generator, Optional
from tqdm import tqdm
from dataclasses import dataclass

import pandas as pd
import numpy as np

# Vector DB
import chromadb
from chromadb.config import Settings

# Embeddings
from sentence_transformers import SentenceTransformer


@dataclass
class PrepConfig:
    """Configuration for Wikipedia preparation."""
    # Paths
    wiki_input_dir: str = "./data/wikipedia"
    chroma_dir: str = "./chroma_db"
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
            print(f"Resumed: {len(self.processed_files)} files already processed, {self.total_chunks} chunks")
    
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
    
    def is_processed(self, filename: str) -> bool:
        """Check if file was already processed."""
        return filename in self.processed_files
    
    def reset(self):
        """Reset progress (start fresh)."""
        self.processed_files = []
        self.total_chunks = 0
        if Path(self.progress_file).exists():
            Path(self.progress_file).unlink()
        print("Progress reset")


class WikipediaChunker:
    """Chunk Wikipedia articles."""
    
    def __init__(self, config: PrepConfig):
        self.config = config
    
    def clean_text(self, text: str) -> str:
        """Clean Wikipedia text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[\[|\]\]', '', text)
        text = re.sub(r'\{\{[^}]*\}\}', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        return text.strip()
    
    def chunk_article(self, text: str, title: str, doc_id: str) -> List[Dict]:
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


class IncrementalIndexBuilder:
    """Build ChromaDB index incrementally, one file at a time."""
    
    def __init__(self, config: PrepConfig):
        self.config = config
        self.chunker = WikipediaChunker(config)
        self.tracker = ProgressTracker(config.progress_file)
        self.embedding_model: Optional[SentenceTransformer] = None
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection = None
    
    def _get_client(self) -> chromadb.PersistentClient:
        """Get or create ChromaDB client."""
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
    
    def _process_single_parquet(self, parquet_path: Path, file_prefix: str) -> int:
        """
        Process a single parquet file and add to ChromaDB.
        Returns number of chunks added.
        """
        print(f"\nProcessing: {parquet_path.name}")
        
        # Read parquet file
        df = pd.read_parquet(parquet_path)
        
        # Detect text column
        text_col = 'text' if 'text' in df.columns else 'content'
        if text_col not in df.columns:
            print(f"  Warning: No text column found in {parquet_path.name}")
            return 0
        
        # Chunk all articles in this file
        all_chunks = []
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
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                metadatas=metadatas
            )
        
        # Free chunks memory
        num_chunks = len(all_chunks)
        del all_chunks
        gc.collect()
        
        return num_chunks
    
    def build_from_directory(self, input_dir: Optional[str] = None, reset: bool = False):
        """
        Build index from all parquet files in directory.
        Processes one file at a time to save memory.
        """
        input_path = Path(input_dir or self.config.wiki_input_dir)
        
        if not input_path.exists():
            print(f"Error: Directory not found: {input_path}")
            return
        
        # Find all parquet files
        parquet_files = sorted(input_path.glob("*.parquet"))
        
        if not parquet_files:
            print(f"No parquet files found in {input_path}")
            return
        
        print(f"Found {len(parquet_files)} parquet files")
        
        # Reset if requested
        if reset:
            self.reset_index()
        
        # Load models
        self._load_embedding_model()
        self._get_or_create_collection()
        
        # Process each file
        for pq_file in parquet_files:
            filename = pq_file.name
            
            # Skip if already processed
            if self.tracker.is_processed(filename):
                print(f"Skipping (already processed): {filename}")
                continue
            
            # Process file
            try:
                file_prefix = pq_file.stem  # filename without extension
                num_chunks = self._process_single_parquet(pq_file, file_prefix)
                
                # Mark as processed
                self.tracker.mark_processed(filename, num_chunks)
                print(f"  ✓ Completed: {num_chunks} chunks (Total: {self.tracker.total_chunks})")
                
                # Force garbage collection
                gc.collect()
                
            except Exception as e:
                print(f"  ✗ Error processing {filename}: {e}")
                continue
        
        # Final summary
        print("\n" + "=" * 60)
        print(f"Indexing complete!")
        print(f"  Files processed: {len(self.tracker.processed_files)}")
        print(f"  Total chunks: {self.tracker.total_chunks}")
        print(f"  ChromaDB location: {self.config.chroma_dir}")
        print("=" * 60)
    
    def build_single_file(self, parquet_path: str, reset: bool = False):
        """
        Build index from a single parquet file.
        Useful for testing or processing specific files.
        """
        pq_file = Path(parquet_path)
        
        if not pq_file.exists():
            print(f"Error: File not found: {pq_file}")
            return
        
        if reset:
            self.reset_index()
        
        self._load_embedding_model()
        self._get_or_create_collection()
        
        file_prefix = pq_file.stem
        num_chunks = self._process_single_parquet(pq_file, file_prefix)
        self.tracker.mark_processed(pq_file.name, num_chunks)
        
        print(f"\n✓ Completed: {num_chunks} chunks added to ChromaDB")
    
    def reset_index(self):
        """Reset ChromaDB collection and progress tracker."""
        print("Resetting index...")
        
        client = self._get_client()
        try:
            client.delete_collection(self.config.collection_name)
            print(f"  Deleted collection: {self.config.collection_name}")
        except Exception:
            pass
        
        self.tracker.reset()
        self.collection = None
    
    def get_stats(self):
        """Get current index statistics."""
        self._get_or_create_collection()
        
        return {
            "collection_name": self.config.collection_name,
            "total_documents": self.collection.count() if self.collection else 0,
            "processed_files": len(self.tracker.processed_files),
            "chroma_dir": self.config.chroma_dir
        }


def main():
    """Main execution with CLI arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare Wikipedia data for RAG (memory-efficient version)"
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
        "--single-file", "-f",
        type=str,
        default=None,
        help="Process only a single parquet file"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset index and start fresh"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show current index statistics"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding computation"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Chunk size in words"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = PrepConfig(
        wiki_input_dir=args.input,
        chroma_dir=args.chroma_dir,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size
    )
    
    # Create builder
    builder = IncrementalIndexBuilder(config)
    
    # Execute command
    if args.stats:
        stats = builder.get_stats()
        print("\nIndex Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    elif args.single_file:
        builder.build_single_file(args.single_file, reset=args.reset)
    else:
        builder.build_from_directory(args.input, reset=args.reset)


if __name__ == "__main__":
    main()