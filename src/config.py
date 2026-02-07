from pathlib import Path

@dataclass
class Config:
    """Configuration for the RAG pipeline."""
    # Paths
    data_dir: str = "../data"
    wiki_dir: str = "../data/wikipedia"
    index_dir: str = "../indices"
    output_dir: str = "../outputs"
    
    # Retrieval
    bm25_top_k: int = 50
    dense_top_k: int = 50
    rerank_top_k: int = 10
    
    # Models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # LLM (Ollama)
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "mistral:7b-instruct-v0.2-q4_K_M"
    
    # Processing
    chunk_size: int = 512
    chunk_overlap: int = 64
    batch_size: int = 32
    
    def __post_init__(self):
        for path in [self.data_dir, self.wiki_dir, self.index_dir, self.output_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)