from enum import Enum
from typing import Any, Tuple

import weave
from datasets import Dataset
from smolagents import (
    ActionStep,
    AgentLogger,
    Model,
    MultiStepAgent,
    PythonInterpreterTool,
    ToolCallingAgent,
)
from tqdm.auto import tqdm

from athena_dr.agent.model import OpenAIModelWithThinkingTraces
from athena_dr.agent.prompts import (
    EXACT_ANSWER_PROMPT_TEMPLATE,
    LONG_ANSWER_PROMPT_TEMPLATE,
    SHORT_ANSWER_PROMPT_TEMPLATE,
    TOOL_CALLING_AGENT_DESCRIPTION,
)
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
        self._tool_calling_agent = ToolCallingAgent(
            model=self._model,
            tools=self._tools,
            max_steps=self.config.agent_max_steps,
            verbosity_level=self.verbosity_level,
            planning_interval=self.planning_interval,
            name=self.config.agent_name,
            description=TOOL_CALLING_AGENT_DESCRIPTION,
            provide_run_summary=True,
            final_answer_checks=[increment_web_agent_token_counts],
        )

    @weave.op
    def postprocess_final_result(
        self, final_result: str, answer_type: AnswerType
    ) -> str:
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
        for step in self._tool_calling_agent.memory.steps:
            if isinstance(step, ActionStep) and step.tool_calls:
                # Exclude final_answer tool calls if you only want actual tool usage
                for tool_call in step.tool_calls:
                    if tool_call.name != "final_answer":
                        tool_calls.append(tool_call.name)
        return {
            "final_result": self.postprocess_final_result(final_result, answer_type),
            "agent_memory": agent_memory,
            "trace": trace,
            "total_tool_calls": len(tool_calls),
            "tool_calls": tool_calls,
        }

    @weave.op
    def generate_sft_traces(
        self,
        dataset: Dataset,
        answer_type: AnswerType,
        prompt_column: str,
        answer_column: str,
        dataset_name: str | None = None,
        max_examples: int | None = None,
    ) -> list[dict]:
        dataset = dataset.select(range(max_examples)) if max_examples else dataset
        self._tool_calling_agent.logger = AgentLogger(verbosity_level=0)
        data_points = []
        for data_point in tqdm(
            dataset, desc="Generating SFT Traces", total=len(dataset)
        ):
            try:
                result = self.predict(data_point[prompt_column], answer_type=answer_type)
                data_points.append(
                    {
                        "prompt": data_point[prompt_column],
                        "original_answer": data_point[answer_column],
                        "answer": result["final_result"],
                        "trace": result["trace"],
                        "tool_calls": result["tool_calls"],
                        "total_tool_calls": result["total_tool_calls"],
                    }
                )
            except Exception as e:
                pass

        if dataset_name:
            dataset = Dataset.from_list(data_points)
            dataset.push_to_hub(dataset_name)

        return data_points
