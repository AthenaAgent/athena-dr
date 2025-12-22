import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any, Tuple

import weave
from datasets import Dataset
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from smolagents import (
    ActionStep,
    AgentLogger,
    Model,
    MultiStepAgent,
    PythonInterpreterTool,
)

from athena_dr.agent.model import OpenAIModelWithThinkingTraces
from athena_dr.agent.prompts import (
    EXACT_ANSWER_PROMPT_TEMPLATE,
    LONG_ANSWER_PROMPT_TEMPLATE,
    SHORT_ANSWER_PROMPT_TEMPLATE,
    TOOL_CALLING_AGENT_DESCRIPTION,
)
from athena_dr.agent.token_limited_agent import TokenLimitedToolCallingAgent
from athena_dr.agent.tools import (
    Crawl4AIFetchTool,
    JinaFetchTool,
    SerperSearchTool,
    TheSportsDBSearchTool,
)
from athena_dr.utils import WorkflowConfig


class AnswerType(Enum):
    SHORT = "short"
    LONG = "long"
    EXACT = "exact"


@weave.op
def increment_web_agent_token_counts(
    final_answer: str, memory_step: int, agent: MultiStepAgent
):
    token_counts_web = agent.monitor.get_total_token_counts()
    token_counts_web.input_tokens += token_counts_web.input_tokens
    token_counts_web.output_tokens += token_counts_web.output_tokens
    return True


class DeepResearchAgent(weave.Model):
    config: WorkflowConfig
    verbosity_level: int = 2
    planning_interval: int = 1
    _tools: list
    _model: Model
    _tool_calling_agent: MultiStepAgent
    _manager_agent: MultiStepAgent

    def model_post_init(self, context: Any, /) -> None:
        self._tools = [
            SerperSearchTool(),
            Crawl4AIFetchTool(),
            JinaFetchTool(),
            TheSportsDBSearchTool(),
            PythonInterpreterTool(),
        ]
        self._model = OpenAIModelWithThinkingTraces(
            model_id=self.config.model_name,
            api_base=self.config.base_url,
            api_key=self.config.api_key,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            extra_body={"reasoning": {"enabled": True}},
        )
        self._tool_calling_agent = TokenLimitedToolCallingAgent(
            model=self._model,
            tools=self._tools,
            max_steps=self.config.agent_max_steps,
            max_output_tokens=self.config.max_output_tokens,
            verbosity_level=self.verbosity_level,
            planning_interval=self.planning_interval,
            name=self.config.agent_name,
            description=TOOL_CALLING_AGENT_DESCRIPTION,
            provide_run_summary=True,
            final_answer_checks=[increment_web_agent_token_counts],
            max_tool_threads=self.config.max_tool_threads,
        )

    @weave.op
    def postprocess_final_result(
        self, final_result: str, answer_type: AnswerType
    ) -> str:
        final_result = re.sub(
            r"<thinking>.*?</thinking>", "", final_result, flags=re.DOTALL
        )
        final_result = (
            final_result
            if not "<answer>" in final_result
            else final_result.split("<answer>")[1].split("</answer>")[0].strip()
        )
        return final_result

    @weave.op
    def predict(self, query: str, answer_type: AnswerType) -> Tuple[str, list]:
        if answer_type == AnswerType.EXACT:
            query = EXACT_ANSWER_PROMPT_TEMPLATE.format(query=query)
        elif answer_type == AnswerType.SHORT:
            query = SHORT_ANSWER_PROMPT_TEMPLATE.format(query=query)
        elif answer_type == AnswerType.LONG:
            query = LONG_ANSWER_PROMPT_TEMPLATE.format(query=query)
        final_result = self._tool_calling_agent.run(query)
        agent_memory = self._tool_calling_agent.write_memory_to_messages()
        trace = []
        for step in agent_memory:
            trace.append(
                {
                    "role": step.role,
                    "content": step.content[0]["text"],
                }
            )
        tool_calls = []
        token_usage_per_step = []
        tool_calling_errors = 0
        for step in self._tool_calling_agent.memory.steps:
            if isinstance(step, ActionStep):
                if step.tool_calls:
                    # Exclude final_answer tool calls if you only want actual tool usage
                    for tool_call in step.tool_calls:
                        if tool_call.name != "final_answer":
                            tool_calls.append(tool_call.name)

                # Count tool calling errors
                if step.error is not None:
                    tool_calling_errors += 1

                # Collect token usage for this step
                if step.token_usage:
                    token_usage_per_step.append(
                        {
                            "step_number": step.step_number,
                            "input_tokens": step.token_usage.input_tokens,
                            "output_tokens": step.token_usage.output_tokens,
                            "total_tokens": step.token_usage.total_tokens,
                        }
                    )

        return {
            "final_result": self.postprocess_final_result(final_result, answer_type),
            "agent_memory": agent_memory,
            "trace": trace,
            "total_tool_calls": len(tool_calls),
            "tool_calls": tool_calls,
            "tool_calling_errors": tool_calling_errors,
            "token_usage_per_step": token_usage_per_step,
        }

    @weave.op
    def generate_sft_traces(
        self,
        dataset: Dataset,
        answer_type: AnswerType,
        prompt_column: str,
        answer_column: str,
        dataset_name: str | None = None,
        min_index: int | None = None,
        max_index: int | None = None,
    ) -> list[dict]:
        if min_index is not None or max_index is not None:
            start = min_index if min_index is not None else 0
            end = max_index if max_index is not None else len(dataset)
            dataset = dataset.select(range(start, end))
        else:
            dataset = dataset
        silent_logger = AgentLogger(level=0)
        self._tool_calling_agent.logger = silent_logger
        self._tool_calling_agent.monitor.logger = silent_logger
        data_points = []

        def process_data_point(data_point):
            try:
                result = self.predict(
                    data_point[prompt_column], answer_type=answer_type
                )
                # Calculate total input and output tokens across all steps
                total_input_tokens = sum(
                    step["input_tokens"] for step in result["token_usage_per_step"]
                )
                total_output_tokens = sum(
                    step["output_tokens"] for step in result["token_usage_per_step"]
                )

                return {
                    "prompt": data_point[prompt_column],
                    "original_answer": data_point[answer_column],
                    "answer": result["final_result"],
                    "conversations": result["trace"],
                    "tool_calls": result["tool_calls"],
                    "total_tool_calls": result["total_tool_calls"],
                    "total_input_tokens": total_input_tokens,
                    "total_output_tokens": total_output_tokens,
                    "tool_calling_errors": result["tool_calling_errors"],
                }
            except Exception as e:
                return None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Generating SFT Traces", total=len(dataset))

            with ThreadPoolExecutor(
                max_workers=self.config.max_agent_workers
            ) as executor:
                futures = {
                    executor.submit(process_data_point, data_point): data_point
                    for data_point in dataset
                }

                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        data_points.append(result)
                    progress.update(task, advance=1)

        if dataset_name:
            dataset = Dataset.from_list(data_points)
            dataset.push_to_hub(dataset_name)

        return data_points
