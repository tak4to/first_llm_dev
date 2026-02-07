"""ChromaDB-based RAG (Retrieval Augmented Generation) module.

Handles:
1. Indexing Wikipedia articles into ChromaDB
2. Retrieving relevant context for science questions
3. OpenAI embedding integration
"""

import os
from pathlib import Path

import chromadb
from openai import OpenAI

from .config import get_openai_api_key, load_config


class OpenAIEmbeddingFunction:
    """ChromaDB-compatible embedding function using OpenAI API."""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI(api_key=get_openai_api_key())
        self.model = model

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        # OpenAI API has a limit per request; batch if necessary
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(input), batch_size):
            batch = input[i : i + batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model)
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)
        return all_embeddings


class ScienceRAG:
    """RAG system using ChromaDB for science question answering."""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()

        chroma_config = config["chromadb"]
        self.persist_dir = chroma_config["persist_directory"]
        self.collection_name = chroma_config["collection_name"]
        self.embedding_model = chroma_config["embedding_model"]
        self.n_results = chroma_config["n_results"]

        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embedding_fn = OpenAIEmbeddingFunction(model=self.embedding_model)

    def get_or_create_collection(self) -> chromadb.Collection:
        """Get or create the Wikipedia science collection."""
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def index_documents(
        self,
        documents: list[str],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
        batch_size: int = 100,
    ) -> None:
        """Index documents into ChromaDB.

        Args:
            documents: List of text documents to index.
            metadatas: Optional metadata for each document.
            ids: Optional unique IDs for each document.
            batch_size: Number of documents to index per batch.
        """
        collection = self.get_or_create_collection()

        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        if metadatas is None:
            metadatas = [{"source": "wikipedia"} for _ in documents]

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]

            collection.upsert(
                documents=batch_docs,
                metadatas=batch_meta,
                ids=batch_ids,
            )
            print(
                f"Indexed batch {i // batch_size + 1}"
                f"/{(len(documents) - 1) // batch_size + 1}"
            )

        print(f"Total documents in collection: {collection.count()}")

    def index_wikipedia_from_dataset(self, max_articles: int | None = None) -> None:
        """Index Wikipedia science articles from HuggingFace datasets.

        Uses the 'wikipedia' dataset filtered for science-related content.
        """
        from datasets import load_dataset

        print("Loading Wikipedia dataset from HuggingFace...")
        dataset = load_dataset(
            "wikipedia", "20220301.en", split="train", streaming=True
        )

        science_keywords = [
            "physics",
            "chemistry",
            "biology",
            "mathematics",
            "astronomy",
            "geology",
            "quantum",
            "molecule",
            "atom",
            "cell",
            "evolution",
            "thermodynamics",
            "electromagnetism",
            "relativity",
            "genetics",
            "ecology",
            "neuroscience",
            "biochemistry",
            "organic chemistry",
            "inorganic chemistry",
            "nuclear",
            "particle",
            "cosmology",
            "astrophysics",
            "optics",
            "mechanics",
            "calculus",
            "algebra",
            "statistics",
            "probability",
        ]

        documents = []
        metadatas = []
        ids = []
        count = 0

        for article in dataset:
            title_lower = article["title"].lower()
            text_lower = article["text"][:500].lower()

            is_science = any(
                kw in title_lower or kw in text_lower for kw in science_keywords
            )

            if is_science:
                # Chunk long articles into smaller pieces
                text = article["text"]
                chunks = self._chunk_text(text, chunk_size=1000, overlap=200)

                for j, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append(
                        {"source": "wikipedia", "title": article["title"]}
                    )
                    ids.append(f"wiki_{count}_{j}")

                count += 1
                if count % 100 == 0:
                    print(f"Processed {count} science articles...")

                if max_articles and count >= max_articles:
                    break

        print(f"Found {count} science articles, {len(documents)} chunks total")
        self.index_documents(documents, metadatas, ids)

    def index_from_text_files(self, directory: str) -> None:
        """Index text files from a local directory into ChromaDB."""
        dir_path = Path(directory)
        if not dir_path.exists():
            print(f"Directory {directory} does not exist, skipping.")
            return

        documents = []
        metadatas = []
        ids = []

        for filepath in sorted(dir_path.glob("*.txt")):
            text = filepath.read_text(encoding="utf-8")
            chunks = self._chunk_text(text, chunk_size=1000, overlap=200)

            for j, chunk in enumerate(chunks):
                documents.append(chunk)
                metadatas.append(
                    {"source": "local", "filename": filepath.name}
                )
                ids.append(f"local_{filepath.stem}_{j}")

        if documents:
            print(f"Indexing {len(documents)} chunks from {directory}")
            self.index_documents(documents, metadatas, ids)
        else:
            print(f"No text files found in {directory}")

    def retrieve(self, query: str, n_results: int | None = None) -> list[str]:
        """Retrieve relevant documents for a query.

        Args:
            query: The search query (typically the question text).
            n_results: Number of results to return.

        Returns:
            List of relevant document texts.
        """
        collection = self.get_or_create_collection()

        if collection.count() == 0:
            return []

        n = n_results or self.n_results
        results = collection.query(query_texts=[query], n_results=n)

        return results["documents"][0] if results["documents"] else []

    def retrieve_for_question(self, question_row: dict) -> str:
        """Retrieve and format context for a science question.

        Args:
            question_row: Dict with 'prompt', 'A', 'B', 'C', 'D', 'E' keys.

        Returns:
            Formatted context string from retrieved documents.
        """
        # Use the question and options as the query for better retrieval
        query_parts = [question_row["prompt"]]
        for opt in ["A", "B", "C", "D", "E"]:
            if opt in question_row:
                query_parts.append(str(question_row[opt]))
        query = " ".join(query_parts)

        docs = self.retrieve(query)
        if not docs:
            return ""

        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[{i}] {doc}")
        return "\n\n".join(context_parts)

    @staticmethod
    def _chunk_text(
        text: str, chunk_size: int = 1000, overlap: int = 200
    ) -> list[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                last_period = text[start:end].rfind(". ")
                if last_period > chunk_size // 2:
                    end = start + last_period + 1

            chunks.append(text[start:end].strip())
            start = end - overlap

        return [c for c in chunks if c]
