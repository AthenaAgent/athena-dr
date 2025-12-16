import asyncio
import warnings

import weave
from datasets import load_dataset

from athena_dr.agent.workflows import TraceGenerator

warnings.filterwarnings("ignore")
weave.init(project_name="athena_dr")
hotpotqa_dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
trace_generator = TraceGenerator(
    auto_search_config_path="configs/auto_search_configs.yml",
    rejection_sampling_config_path="configs/rejection_sampling_configs.yml",
    dataset=hotpotqa_dataset,
    dataset_name="hotpotqa",
    max_examples=5,
)
traces = asyncio.run(
    trace_generator.generate_trace(
        prompt_column="question",
        gt_answer_column="answer",
        max_attempts_per_example=3,
        export_dataset="geekyrakshit/hotpotqa_sft_traces",
    )
)
