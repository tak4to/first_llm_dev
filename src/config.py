from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Configuration for the Kaggle LLM Science Exam pipeline."""

    # Paths
    data_dir: str = "./data"
    wiki_dir: str = "./data/wikipedia"
    index_dir: str = "./indices"
    output_dir: str = "./outputs"
    chroma_dir: str = "./chroma_db"
    model_output_dir: str = "./models/finetuned"

    # ChromaDB
    collection_name: str = "wikipedia_chunks"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Retrieval
    retrieval_top_k: int = 5
    max_context_length: int = 1024

    # LLM Model (base model for fine-tuning)
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"

    # QLoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Training
    learning_rate: float = 2e-4
    num_epochs: int = 3
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 768
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    val_split_ratio: float = 0.1

    # Inference
    max_new_tokens: int = 5

    def __post_init__(self):
        for path_attr in [self.data_dir, self.output_dir, self.model_output_dir]:
            Path(path_attr).mkdir(parents=True, exist_ok=True)
