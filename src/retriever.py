"""ChromaDB Retriever for RAG-based science question answering."""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional


class ChromaDBRetriever:
    """Retrieve relevant Wikipedia context from ChromaDB."""

    def __init__(
        self,
        chroma_dir: str,
        collection_name: str,
        embedding_model_name: str,
        top_k: int = 5,
    ):
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.top_k = top_k
        self.embedding_model_name = embedding_model_name

        self.client = None
        self.collection = None
        self.embedding_model = None

    def _init(self):
        """Initialize ChromaDB client and embedding model lazily."""
        if self.client is None:
            self.client = chromadb.PersistentClient(
                path=self.chroma_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            self.collection = self.client.get_collection(self.collection_name)
            print(f"Connected to ChromaDB: {self.collection.count()} documents")

        if self.embedding_model is None:
            print(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Retrieve relevant documents for a query string."""
        self._init()
        k = top_k or self.top_k

        query_embedding = self.embedding_model.encode(
            [query], normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["metadatas", "distances"],
        )

        documents = []
        if results and results["metadatas"]:
            for i, metadata in enumerate(results["metadatas"][0]):
                documents.append(
                    {
                        "title": metadata.get("title", ""),
                        "text": metadata.get("text", ""),
                        "distance": (
                            results["distances"][0][i] if results["distances"] else 0
                        ),
                    }
                )

        return documents

    def retrieve_for_question(
        self, question: str, options: List[str], top_k: Optional[int] = None
    ) -> str:
        """Retrieve and format context for a multiple-choice question.

        Combines the question with answer options to create a richer
        search query, then returns formatted context passages.
        """
        query = f"{question} {' '.join(options)}"
        docs = self.retrieve(query, top_k)

        context_parts = []
        for doc in docs:
            title = doc["title"]
            text = doc["text"]
            context_parts.append(f"[{title}] {text}")

        return "\n\n".join(context_parts)
