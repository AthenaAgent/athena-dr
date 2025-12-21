from typing import Any, Tuple

import weave
from smolagents import (
    Model,
    MultiStepAgent,
    OpenAIModel,
    ToolCallingAgent,
)

from athena_dr.agent.prompts import (
    DEEP_RESEARCH_PROMPT_TEMPLATE,
    TOOL_CALLING_AGENT_DESCRIPTION,
)
from athena_dr.agent.tools import Crawl4AIFetchTool, JinaFetchTool, SerperSearchTool
from athena_dr.utils import WorkflowConfig


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
        self._tools = [SerperSearchTool(), Crawl4AIFetchTool(), JinaFetchTool()]
        self._model = OpenAIModel(
            model_id=self.config.model_name,
            api_base=self.config.base_url,
            api_key=self.config.api_key,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
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
    def predict(self, query: str) -> Tuple[str, list]:
        query = DEEP_RESEARCH_PROMPT_TEMPLATE.format(task=query)
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
