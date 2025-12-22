from enum import Enum
from typing import Any, Tuple

import weave
from smolagents import (
    Model,
    MultiStepAgent,
    PythonInterpreterTool,
    ToolCallingAgent,
)

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
            verbosity_level=2,
            planning_interval=1,
            name=self.config.agent_name,
            description=TOOL_CALLING_AGENT_DESCRIPTION,
            provide_run_summary=True,
            final_answer_checks=[increment_web_agent_token_counts],
        )

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
        return {
            "final_result": final_result,
            "agent_memory": agent_memory,
            "trace": trace,
        }
