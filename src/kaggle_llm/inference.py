"""Inference module for Kaggle LLM Science Exam.

Handles:
1. Running inference with fine-tuned OpenAI model
2. RAG-augmented inference
3. Multi-answer prediction for MAP@3
4. Generating submission files
"""

import re
import time

import pandas as pd
from openai import OpenAI

from .config import get_openai_api_key, load_config
from .data_preparation import SYSTEM_PROMPT, SYSTEM_PROMPT_WITH_CONTEXT, format_question
from .rag import ScienceRAG


RANKING_SYSTEM_PROMPT = (
    "You are a science exam expert. Given a multiple-choice science question "
    "with 5 options (A, B, C, D, E), rank the top 3 most likely correct "
    "answers in order of confidence. Respond with exactly 3 letters separated "
    "by spaces (e.g., 'C A B'). The first letter should be your most confident answer."
)

RANKING_SYSTEM_PROMPT_WITH_CONTEXT = (
    "You are a science exam expert. You are given relevant context from "
    "Wikipedia articles, followed by a multiple-choice science question with "
    "5 options (A, B, C, D, E). Use the context to help determine the correct "
    "answer. Rank the top 3 most likely correct answers in order of confidence. "
    "Respond with exactly 3 letters separated by spaces (e.g., 'C A B')."
)


class ScienceExamInference:
    """Inference engine for science exam questions."""

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()

        self.config = config
        self.client = OpenAI(api_key=get_openai_api_key())

        # Use fine-tuned model if available, otherwise base model
        self.model = (
            config["openai"]["finetuned_model"] or config["openai"]["model"]
        )
        self.temperature = config["openai"]["temperature"]
        self.max_tokens = config["openai"]["max_tokens"]
        self.use_rag = config["inference"]["use_rag"]
        self.top_k = config["inference"]["top_k_predictions"]

        self.rag = None
        if self.use_rag:
            self.rag = ScienceRAG(config)

    def predict_single(
        self, question_row: dict, use_ranking: bool = True
    ) -> str:
        """Predict answer(s) for a single question.

        Args:
            question_row: Dict with 'prompt', 'A', 'B', 'C', 'D', 'E' keys.
            use_ranking: If True, ask for top 3 ranked answers.

        Returns:
            Space-separated prediction string (e.g., 'C A B').
        """
        # Get RAG context if available
        context = None
        if self.rag:
            context = self.rag.retrieve_for_question(question_row)

        question_text = format_question(
            pd.Series(question_row), context=context
        )

        if use_ranking:
            system_prompt = (
                RANKING_SYSTEM_PROMPT_WITH_CONTEXT
                if context
                else RANKING_SYSTEM_PROMPT
            )
        else:
            system_prompt = (
                SYSTEM_PROMPT_WITH_CONTEXT if context else SYSTEM_PROMPT
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_text},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        raw_answer = response.choices[0].message.content.strip()
        return self._parse_prediction(raw_answer, use_ranking)

    def predict_batch(
        self, df: pd.DataFrame, use_ranking: bool = True
    ) -> list[str]:
        """Predict answers for a batch of questions.

        Args:
            df: DataFrame with question columns.
            use_ranking: If True, ask for top 3 ranked answers.

        Returns:
            List of prediction strings.
        """
        predictions = []
        total = len(df)

        for i, (_, row) in enumerate(df.iterrows()):
            question_row = row.to_dict()
            try:
                pred = self.predict_single(question_row, use_ranking)
                predictions.append(pred)
            except Exception as e:
                print(f"Error on question {i}: {e}")
                predictions.append("A B C")  # Fallback

            if (i + 1) % 10 == 0:
                print(f"Progress: {i + 1}/{total}")

            # Rate limiting: brief pause between requests
            time.sleep(0.5)

        return predictions

    def generate_submission(
        self,
        test_df: pd.DataFrame,
        output_path: str,
    ) -> pd.DataFrame:
        """Generate a Kaggle submission file.

        Args:
            test_df: Test DataFrame with question columns.
            output_path: Path to save the submission CSV.

        Returns:
            Submission DataFrame.
        """
        print(f"Running inference on {len(test_df)} questions...")
        predictions = self.predict_batch(test_df, use_ranking=True)

        submission = pd.DataFrame(
            {"id": test_df["id"], "prediction": predictions}
        )
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        return submission

    @staticmethod
    def _parse_prediction(raw: str, use_ranking: bool) -> str:
        """Parse model output into valid prediction format.

        Args:
            raw: Raw model output string.
            use_ranking: Whether we asked for ranked top 3.

        Returns:
            Space-separated prediction (e.g., 'C A B').
        """
        valid_answers = {"A", "B", "C", "D", "E"}

        # Extract letters from the response
        letters = re.findall(r"[A-E]", raw.upper())
        # Remove duplicates while preserving order
        seen = set()
        unique_letters = []
        for letter in letters:
            if letter not in seen and letter in valid_answers:
                seen.add(letter)
                unique_letters.append(letter)

        if not unique_letters:
            return "A B C"

        if use_ranking:
            # Ensure we have exactly 3 predictions
            while len(unique_letters) < 3:
                for fallback in ["A", "B", "C", "D", "E"]:
                    if fallback not in seen:
                        unique_letters.append(fallback)
                        seen.add(fallback)
                        break
            return " ".join(unique_letters[:3])
        else:
            return unique_letters[0]
