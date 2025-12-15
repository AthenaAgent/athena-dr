import os
from uuid import uuid4

import weave
from datasets import Dataset
from tqdm.auto import tqdm

from athena_dr.agent.shared_prompts import (
    SHORT_FORM_ANSWER_EVALUATION_USER_PROMPT_FORMAT,
    UNIFIED_TOOL_CALLING_STRUCTURED_PROMPTS,
)
from athena_dr.agent.workflows import AutoReasonSearchWorkflow


class TraceGenerator:
    def __init__(
        self,
        auto_search_config_path: os.PathLike,
        dataset: Dataset,
        dataset_name: str,
        max_examples: int | None = None,
        f1_overlap_score_threshold: float = 0.9,
    ) -> None:
        self.workflow = AutoReasonSearchWorkflow(configuration=auto_search_config_path)
        if max_examples is not None:
            dataset = dataset.select(range(min(max_examples, len(dataset))))
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.f1_overlap_score_threshold = f1_overlap_score_threshold
        self.traces = []

    def __getstate__(self):
        # Return only lightweight data for serialization
        return {"f1_overlap_score_threshold": self.f1_overlap_score_threshold}

    @weave.op
    async def process_example(
        self,
        example_idx: int,
        example: dict,
        dataset_name: str,
        prompt_column: str,
        gt_answer_column: str,
    ) -> dict | None:
        problem = SHORT_FORM_ANSWER_EVALUATION_USER_PROMPT_FORMAT.format(
            prompt=example[prompt_column]
        )
        trace = await self.workflow(problem=problem)
        return {
            "id": str(uuid4()),
            "source_id": f"{dataset_name}_{example_idx}",
            "question": example[prompt_column],
            "source": dataset_name,
            "type": "short_form",
            "num_tool_calls": trace["total_tool_calls"],
            "conversations": [
                {
                    "content": UNIFIED_TOOL_CALLING_STRUCTURED_PROMPTS,
                    "role": "system",
                },
                {
                    "content": problem,
                    "role": "user",
                },
                {
                    "content": trace["full_traces"].generated_text,
                    "role": "assistant",
                },
            ],
        }

    @weave.op
    async def generate_trace(
        self,
        prompt_column: str,
        gt_answer_column: str,
        export_dataset: str | None = None,
        max_attempts_per_example: int = 5,
    ) -> dict:
        for example_idx, example in tqdm(
            enumerate(self.dataset), total=len(self.dataset), desc="Generating traces"
        ):
            trace = await self.process_example(
                example_idx, example, self.dataset_name, prompt_column, gt_answer_column
            )
            self.traces.append(trace)

        if export_dataset is not None:
            dataset = Dataset.from_list(self.traces)
            dataset.push_to_hub(export_dataset)
        return self.traces
