import os
from uuid import uuid4

import rich
import weave
from datasets import Dataset
from tqdm.auto import tqdm

from athena_dr.agent.shared_prompts import (
    SHORT_FORM_ANSWER_EVALUATION_USER_PROMPT_FORMAT,
    UNIFIED_TOOL_CALLING_STRUCTURED_PROMPTS,
)
from athena_dr.agent.workflows import AutoReasonSearchWorkflow
from athena_dr.agent.workflows.rejection_sampling import TraceChecker


class TraceGenerator:
    def __init__(
        self,
        auto_search_config_path: os.PathLike,
        rejection_sampling_config_path: os.PathLike,
        dataset: Dataset,
        dataset_name: str,
        max_examples: int | None = None,
        f1_overlap_score_threshold: float = 0.9,
    ) -> None:
        self.workflow = AutoReasonSearchWorkflow(configuration=auto_search_config_path)
        self.trace_checker = TraceChecker(rejection_sampling_config_path)
        if max_examples is not None:
            dataset = dataset.select(range(min(max_examples, len(dataset))))
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.f1_overlap_score_threshold = f1_overlap_score_threshold
        self.sft_traces = []

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
    ) -> tuple[dict, dict] | None:
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
                    "content": UNIFIED_TOOL_CALLING_STRUCTURED_PROMPTS[
                        self.workflow.configuration.prompt_version
                    ]["system_prompt"],
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
        }, trace

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
            for attempt_idx in tqdm(range(max_attempts_per_example), leave=False):
                sft_data_point, trace = await self.process_example(
                    example_idx,
                    example,
                    self.dataset_name,
                    prompt_column,
                    gt_answer_column,
                )
                if self.trace_checker.check_answer_correctness(
                    question=example[prompt_column],
                    target=example[gt_answer_column],
                    predicted_answer=trace["final_response"],
                ):
                    self.sft_traces.append(sft_data_point)
                    break
                else:
                    rich.print(
                        f"Retrying example {example_idx} because the answer is incorrect"
                    )
                    continue

        if export_dataset is not None:
            dataset = Dataset.from_list(self.sft_traces)
            dataset.push_to_hub(export_dataset)
        return self.sft_traces
