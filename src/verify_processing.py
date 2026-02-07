"""
Wikipedia Processing Verification Script
=========================================
Verifies that all parquet files have been processed correctly.

Checks:
1. All parquet files in input directory are in progress.json
2. ChromaDB collection exists and has expected document count
3. No missing or duplicate document IDs
4. Sample verification of document content
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import Counter

import pandas as pd
from tqdm import tqdm

# Vector DB
import chromadb
from chromadb.config import Settings


@dataclass
class VerifyConfig:
    """Configuration for verification."""
    wiki_input_dir: str = "./data/wikipedia"
    chroma_dir: str = "./chroma_db"
    progress_file: str = "./indices/wiki_progress.json"
    collection_name: str = "wikipedia_chunks"


class ProcessingVerifier:
    """Verify Wikipedia processing completion and integrity."""
    
    def __init__(self, config: VerifyConfig):
        self.config = config
        self.issues: List[str] = []
        self.warnings: List[str] = []
    
    def _load_progress(self) -> Dict:
        """Load progress file."""
        progress_path = Path(self.config.progress_file)
        if not progress_path.exists():
            return {"processed_files": [], "total_chunks": 0}
        
        with open(progress_path, "r") as f:
            return json.load(f)
    
    def _get_parquet_files(self) -> List[str]:
        """Get list of parquet files in input directory."""
        input_path = Path(self.config.wiki_input_dir)
        if not input_path.exists():
            return []
        return sorted([f.name for f in input_path.glob("*.parquet")])
    
    def _get_chroma_collection(self):
        """Get ChromaDB collection."""
        try:
            client = chromadb.PersistentClient(
                path=self.config.chroma_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            return client.get_collection(self.config.collection_name)
        except Exception as e:
            return None
    
    def check_progress_file(self) -> Tuple[bool, Dict]:
        """Check if progress file exists and is valid."""
        print("\n" + "=" * 60)
        print("1. Checking Progress File")
        print("=" * 60)
        
        progress_path = Path(self.config.progress_file)
        
        if not progress_path.exists():
            self.issues.append("Progress file not found")
            print(f"  ✗ Progress file not found: {progress_path}")
            return False, {}
        
        try:
            progress = self._load_progress()
            print(f"  ✓ Progress file exists: {progress_path}")
            print(f"    - Processed files: {len(progress.get('processed_files', []))}")
            print(f"    - Total chunks recorded: {progress.get('total_chunks', 0):,}")
            return True, progress
        except json.JSONDecodeError:
            self.issues.append("Progress file is corrupted")
            print(f"  ✗ Progress file is corrupted")
            return False, {}
    
    def check_file_coverage(self, progress: Dict) -> Tuple[bool, List[str], List[str]]:
        """Check if all parquet files have been processed."""
        print("\n" + "=" * 60)
        print("2. Checking File Coverage")
        print("=" * 60)
        
        parquet_files = self._get_parquet_files()
        processed_files = set(progress.get("processed_files", []))
        
        if not parquet_files:
            self.warnings.append("No parquet files found in input directory")
            print(f"  ⚠ No parquet files found in: {self.config.wiki_input_dir}")
            return True, [], []
        
        print(f"  Found {len(parquet_files)} parquet files in input directory")
        print(f"  Found {len(processed_files)} files in progress record")
        
        # Find missing and extra files
        missing_files = [f for f in parquet_files if f not in processed_files]
        extra_files = [f for f in processed_files if f not in parquet_files]
        
        if missing_files:
            self.issues.append(f"{len(missing_files)} files not processed")
            print(f"\n  ✗ Missing (not processed): {len(missing_files)} files")
            for f in missing_files[:10]:  # Show first 10
                print(f"    - {f}")
            if len(missing_files) > 10:
                print(f"    ... and {len(missing_files) - 10} more")
        else:
            print(f"\n  ✓ All parquet files have been processed")
        
        if extra_files:
            self.warnings.append(f"{len(extra_files)} processed files not in input directory")
            print(f"\n  ⚠ Extra in progress (files removed?): {len(extra_files)} files")
            for f in extra_files[:5]:
                print(f"    - {f}")
        
        all_processed = len(missing_files) == 0
        return all_processed, missing_files, extra_files
    
    def check_chromadb(self, progress: Dict) -> Tuple[bool, int]:
        """Check ChromaDB collection."""
        print("\n" + "=" * 60)
        print("3. Checking ChromaDB Collection")
        print("=" * 60)
        
        chroma_path = Path(self.config.chroma_dir)
        
        if not chroma_path.exists():
            self.issues.append("ChromaDB directory not found")
            print(f"  ✗ ChromaDB directory not found: {chroma_path}")
            return False, 0
        
        collection = self._get_chroma_collection()
        
        if collection is None:
            self.issues.append("ChromaDB collection not found")
            print(f"  ✗ Collection '{self.config.collection_name}' not found")
            return False, 0
        
        actual_count = collection.count()
        expected_count = progress.get("total_chunks", 0)
        
        print(f"  ✓ ChromaDB collection exists: {self.config.collection_name}")
        print(f"    - Actual documents: {actual_count:,}")
        print(f"    - Expected (from progress): {expected_count:,}")
        
        if actual_count == expected_count:
            print(f"\n  ✓ Document count matches exactly")
            return True, actual_count
        elif actual_count > expected_count:
            diff = actual_count - expected_count
            self.warnings.append(f"ChromaDB has {diff:,} more documents than expected")
            print(f"\n  ⚠ ChromaDB has {diff:,} MORE documents than recorded")
            print(f"    (This may be okay if progress file was reset)")
            return True, actual_count
        else:
            diff = expected_count - actual_count
            self.issues.append(f"ChromaDB is missing {diff:,} documents")
            print(f"\n  ✗ ChromaDB is MISSING {diff:,} documents")
            return False, actual_count
    
    def check_document_integrity(self, sample_size: int = 100) -> bool:
        """Check integrity of sample documents."""
        print("\n" + "=" * 60)
        print("4. Checking Document Integrity (Sample)")
        print("=" * 60)
        
        collection = self._get_chroma_collection()
        if collection is None:
            print("  ✗ Cannot check integrity - collection not available")
            return False
        
        total_docs = collection.count()
        if total_docs == 0:
            self.issues.append("ChromaDB collection is empty")
            print("  ✗ Collection is empty")
            return False
        
        # Get sample documents
        sample_size = min(sample_size, total_docs)
        print(f"  Sampling {sample_size} documents...")
        
        results = collection.get(
            limit=sample_size,
            include=["metadatas"]
        )
        
        # Check for issues
        issues_found = 0
        
        # Check IDs
        ids = results["ids"]
        if len(ids) != len(set(ids)):
            duplicates = [id for id, count in Counter(ids).items() if count > 1]
            self.issues.append(f"Found {len(duplicates)} duplicate IDs")
            print(f"  ✗ Found duplicate IDs: {duplicates[:5]}")
            issues_found += 1
        
        # Check metadata
        empty_titles = 0
        empty_texts = 0
        
        for metadata in results["metadatas"]:
            if not metadata.get("title"):
                empty_titles += 1
            if not metadata.get("text"):
                empty_texts += 1
        
        if empty_titles > 0:
            self.warnings.append(f"{empty_titles} documents have empty titles")
            print(f"  ⚠ {empty_titles} documents have empty titles")
        
        if empty_texts > 0:
            self.issues.append(f"{empty_texts} documents have empty text")
            print(f"  ✗ {empty_texts} documents have empty text")
            issues_found += 1
        
        if issues_found == 0:
            print(f"  ✓ Sample integrity check passed")
            return True
        
        return False
    
    def check_file_prefix_coverage(self) -> bool:
        """Check that all file prefixes are represented in ChromaDB."""
        print("\n" + "=" * 60)
        print("5. Checking File Prefix Coverage in ChromaDB")
        print("=" * 60)
        
        collection = self._get_chroma_collection()
        if collection is None:
            print("  ✗ Cannot check - collection not available")
            return False
        
        progress = self._load_progress()
        processed_files = progress.get("processed_files", [])
        
        if not processed_files:
            print("  ⚠ No processed files recorded")
            return True
        
        # Get expected prefixes (filename without extension)
        expected_prefixes = set(Path(f).stem for f in processed_files)
        
        # Sample IDs to find actual prefixes
        print(f"  Sampling document IDs to verify file prefixes...")
        
        # Get a larger sample to check prefixes
        sample_size = min(10000, collection.count())
        results = collection.get(limit=sample_size)
        
        # Extract prefixes from IDs (format: prefix_rownum_chunknum)
        actual_prefixes = set()
        for doc_id in results["ids"]:
            parts = doc_id.rsplit("_", 2)
            if len(parts) >= 2:
                prefix = parts[0]
                actual_prefixes.add(prefix)
        
        print(f"  Expected file prefixes: {len(expected_prefixes)}")
        print(f"  Found prefixes in sample: {len(actual_prefixes)}")
        
        missing_prefixes = expected_prefixes - actual_prefixes
        
        if missing_prefixes and len(missing_prefixes) < len(expected_prefixes) * 0.5:
            # Some missing might be due to sampling
            self.warnings.append(f"Some file prefixes not found in sample (may be sampling issue)")
            print(f"  ⚠ {len(missing_prefixes)} prefixes not found in sample")
            print(f"    (This may be a sampling issue if files had few chunks)")
        elif missing_prefixes:
            self.issues.append(f"{len(missing_prefixes)} file prefixes missing from ChromaDB")
            print(f"  ✗ {len(missing_prefixes)} file prefixes missing:")
            for p in list(missing_prefixes)[:10]:
                print(f"    - {p}")
            return False
        else:
            print(f"  ✓ All file prefixes found in ChromaDB")
        
        return True
    
    def run_all_checks(self) -> bool:
        """Run all verification checks."""
        print("\n" + "=" * 60)
        print("Wikipedia Processing Verification")
        print("=" * 60)
        print(f"\nInput directory: {self.config.wiki_input_dir}")
        print(f"ChromaDB directory: {self.config.chroma_dir}")
        print(f"Progress file: {self.config.progress_file}")
        
        # Run checks
        progress_ok, progress = self.check_progress_file()
        
        if progress_ok:
            coverage_ok, missing, extra = self.check_file_coverage(progress)
            chroma_ok, doc_count = self.check_chromadb(progress)
            
            if chroma_ok and doc_count > 0:
                integrity_ok = self.check_document_integrity()
                prefix_ok = self.check_file_prefix_coverage()
            else:
                integrity_ok = False
                prefix_ok = False
        else:
            coverage_ok = False
            chroma_ok = False
            integrity_ok = False
            prefix_ok = False
        
        # Summary
        print("\n" + "=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)
        
        all_passed = True
        
        if self.issues:
            all_passed = False
            print(f"\n❌ ISSUES FOUND ({len(self.issues)}):")
            for issue in self.issues:
                print(f"   • {issue}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        if all_passed:
            print(f"\n✅ ALL CHECKS PASSED")
            print(f"\nProcessing completed successfully!")
        else:
            print(f"\n❌ VERIFICATION FAILED")
            print(f"\nSome issues need to be resolved.")
            
            if missing if 'missing' in dir() else []:
                print(f"\nTo process missing files, run:")
                print(f"  python prepare_wikipedia.py --input {self.config.wiki_input_dir}")
        
        print("\n" + "=" * 60)
        
        return all_passed
    
    def show_detailed_status(self):
        """Show detailed status of each file."""
        print("\n" + "=" * 60)
        print("Detailed File Status")
        print("=" * 60)
        
        parquet_files = self._get_parquet_files()
        progress = self._load_progress()
        processed_files = set(progress.get("processed_files", []))
        
        if not parquet_files:
            print("No parquet files found.")
            return
        
        print(f"\n{'File':<40} {'Status':<15}")
        print("-" * 55)
        
        for f in parquet_files:
            if f in processed_files:
                status = "✓ Processed"
            else:
                status = "✗ Not processed"
            print(f"{f:<40} {status:<15}")
        
        # Summary
        processed_count = len([f for f in parquet_files if f in processed_files])
        total_count = len(parquet_files)
        
        print("-" * 55)
        print(f"Total: {processed_count}/{total_count} files processed")
        
        if processed_count == total_count:
            print("\n✓ All files have been processed!")
        else:
            print(f"\n✗ {total_count - processed_count} files remaining")


def main():
    parser = argparse.ArgumentParser(
        description="Verify Wikipedia processing completion"
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
        default="./chroma_db",
        help="ChromaDB storage directory"
    )
    parser.add_argument(
        "--progress-file",
        type=str,
        default="./indices/wiki_progress.json",
        help="Progress tracking file"
    )
    parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="Show detailed status of each file"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick check (skip integrity checks)"
    )
    
    args = parser.parse_args()
    
    config = VerifyConfig(
        wiki_input_dir=args.input,
        chroma_dir=args.chroma_dir,
        progress_file=args.progress_file
    )
    
    verifier = ProcessingVerifier(config)
    
    if args.detailed:
        verifier.show_detailed_status()
    else:
        success = verifier.run_all_checks()
        exit(0 if success else 1)


if __name__ == "__main__":
    main()