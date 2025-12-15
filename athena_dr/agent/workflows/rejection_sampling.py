import os
import re
import string
from collections import Counter
from dataclasses import dataclass

import weave
from litellm import completion
from omegaconf import OmegaConf

from athena_dr.agent.shared_prompts import GRADER_SYSTEM_PROMPT


@dataclass
class GraderConfig:
    grader_model_name: str = "azure/gpt-4.1"
    grader_api_key: str = "dummy-key"
    grader_base_url: str = "https://soumik-dr.cognitiveservices.azure.com/"
    grader_max_tokens: int = 32000
    grader_temperature: float = 0.3
    grader_max_tool_calls: int = 5


class TraceChecker:
    def __init__(
        self, grader_config_path: os.PathLike, f1_overlap_score_threshold: float = 0.9
    ) -> None:
        self.grader_config = self.get_grader_config(grader_config_path)
        self.f1_overlap_score_threshold = f1_overlap_score_threshold

    def get_grader_config(self, grader_config_path: os.PathLike) -> GraderConfig:
        yaml_conf = OmegaConf.load(grader_config_path)
        resolved_dict = OmegaConf.to_container(yaml_conf, resolve=True)
        grader_config = GraderConfig(**resolved_dict)
        return grader_config

    @weave.op
    def get_f1_overlap_score(self, prediction: str, gold: str) -> float:
        """
        SQuAD-style token-overlap F1 between two answer strings (range: 0.0 to 1.0).
        """

        def normalize_answer(s: str) -> str:
            # Lowercase, remove punctuation, remove articles, fix whitespace
            def lower(text: str) -> str:
                return text.lower()

            def remove_punc(text: str) -> str:
                exclude = set(string.punctuation)
                return "".join(ch for ch in text if ch not in exclude)

            def remove_articles(text: str) -> str:
                return re.sub(r"\b(a|an|the)\b", " ", text)

            def white_space_fix(text: str) -> str:
                return " ".join(text.split())

            return white_space_fix(remove_articles(remove_punc(lower(s))))

        pred_tokens = normalize_answer(prediction).split()
        gold_tokens = normalize_answer(gold).split()

        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens) if pred_tokens else 0.0
        recall = num_same / len(gold_tokens) if gold_tokens else 0.0
        return (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall)
            else 0.0
        )

    @weave.op
    def check_answer_correctness(
        self, question: str, target: str, predicted_answer: str
    ) -> bool:
        f1_overlap_score = self.get_f1_overlap_score(
            prediction=predicted_answer, gold=target
        )
        if f1_overlap_score < self.f1_overlap_score_threshold:
            result = completion(
                model=self.grader_config.grader_model_name,
                messages=[
                    {
                        "role": "user",
                        "content": GRADER_SYSTEM_PROMPT.format(
                            question=question,
                            target=target,
                            predicted_answer=predicted_answer,
                        ),
                    },
                ],
                api_key=self.grader_config.grader_api_key,
                base_url=self.grader_config.grader_base_url,
                max_tokens=self.grader_config.grader_max_tokens,
                temperature=self.grader_config.grader_temperature,
            )
            if result.choices[0].message.content.lower() == "a":
                return True
            else:
                return False
        return True
