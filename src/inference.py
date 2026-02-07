"""Inference for Kaggle LLM Science Exam.

Loads the base model with the fine-tuned LoRA adapter,
generates ranked answer predictions for each question,
and writes a MAP@3 submission CSV.
"""

import json
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from src.config import Config

ANSWER_LETTERS = ["A", "B", "C", "D", "E"]


def load_inference_model(config: Config, adapter_path: Optional[str] = None):
    """Load the base model in 4-bit, optionally merging a LoRA adapter."""
    print(f"Loading base model: {config.model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Resolve adapter path
    if adapter_path is None:
        adapter_path = str(Path(config.model_output_dir) / "final_adapter")

    if Path(adapter_path).exists():
        print(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("LoRA adapter merged.")
    else:
        print(f"No adapter found at {adapter_path}. Using base model only.")

    model.eval()
    return model, tokenizer


def _get_answer_token_ids(tokenizer) -> Dict[str, List[int]]:
    """Return a mapping of answer letter -> candidate token IDs.

    Includes both with and without a leading space, since
    tokenizers may represent ` A` and `A` differently.
    """
    token_map = {}
    for letter in ANSWER_LETTERS:
        ids_plain = tokenizer.encode(letter, add_special_tokens=False)
        ids_space = tokenizer.encode(f" {letter}", add_special_tokens=False)
        token_map[letter] = list(set(ids_plain + ids_space))
    return token_map


def predict_single(
    model,
    tokenizer,
    prompt: str,
    answer_token_map: Dict[str, List[int]],
    device: str = "cuda",
) -> List[str]:
    """Predict answer ranking for one question.

    Returns answer letters sorted by descending logit score.
    """
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=768
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Logits at the last input position
    last_logits = outputs.logits[0, -1, :]

    scores = {}
    for letter in ANSWER_LETTERS:
        candidate_ids = answer_token_map[letter]
        # Take the maximum logit across candidate token IDs
        scores[letter] = max(last_logits[tid].item() for tid in candidate_ids)

    ranked = sorted(scores, key=scores.get, reverse=True)
    return ranked


def predict(config: Config, adapter_path: Optional[str] = None) -> Path:
    """Generate predictions for the test set and write submission.csv.

    Returns the path to the generated submission file.
    """
    test_data_path = Path(config.output_dir) / "test_prepared.json"
    if not test_data_path.exists():
        raise FileNotFoundError(
            f"Prepared test data not found at {test_data_path}.\n"
            f"Run 'python scripts/main.py prepare' first."
        )

    with open(test_data_path, encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"Loaded {len(test_data)} test examples")

    model, tokenizer = load_inference_model(config, adapter_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    answer_token_map = _get_answer_token_ids(tokenizer)

    predictions = []
    for item in tqdm(test_data, desc="Predicting"):
        ranked = predict_single(model, tokenizer, item["prompt"], answer_token_map, device)
        predictions.append(
            {"id": item["id"], "prediction": " ".join(ranked[:3])}
        )

    submission_df = pd.DataFrame(predictions)
    submission_path = Path(config.output_dir) / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved -> {submission_path}")

    return submission_path


def evaluate(config: Config, adapter_path: Optional[str] = None) -> Dict:
    """Evaluate the model on the validation split of training data.

    Prints top-1 accuracy and MAP@3 score.
    """
    train_data_path = Path(config.output_dir) / "train_prepared.json"
    if not train_data_path.exists():
        raise FileNotFoundError(
            f"Prepared training data not found at {train_data_path}.\n"
            f"Run 'python scripts/main.py prepare' first."
        )

    with open(train_data_path, encoding="utf-8") as f:
        all_data = json.load(f)

    # Use the same split as training
    split_idx = int(len(all_data) * (1 - config.val_split_ratio))
    val_data = all_data[split_idx:]
    print(f"Evaluating on {len(val_data)} validation examples")

    model, tokenizer = load_inference_model(config, adapter_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    answer_token_map = _get_answer_token_ids(tokenizer)

    correct = 0
    map3_scores = []

    for item in tqdm(val_data, desc="Evaluating"):
        ranked = predict_single(model, tokenizer, item["prompt"], answer_token_map, device)
        true_answer = item["answer"].strip()

        if ranked[0] == true_answer:
            correct += 1

        score = 0.0
        for k, pred in enumerate(ranked[:3]):
            if pred == true_answer:
                score = 1.0 / (k + 1)
                break
        map3_scores.append(score)

    accuracy = correct / len(val_data) if val_data else 0
    map3 = sum(map3_scores) / len(map3_scores) if map3_scores else 0

    print(f"\nValidation Results:")
    print(f"  Top-1 Accuracy : {accuracy:.4f}")
    print(f"  MAP@3          : {map3:.4f}")

    return {"accuracy": accuracy, "map3": map3}


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import Config

    config = Config()
    evaluate(config)
