from dataclasses import dataclass
from typing import Any, Dict, Optional

from athena_dr.agent.agent_interface import BaseAgent
from athena_dr.agent.client import DocumentToolOutput, ToolOutput
from athena_dr.agent.shared_prompts import UNIFIED_TOOL_CALLING_STRUCTURED_PROMPTS
from athena_dr.agent.tool_interface.mcp_tools import (
    BaseTool,
)


@dataclass
class SearchAgent(BaseAgent):
    prompt_version: str = "v20250907"

    def prompt(
        self,
        question: str,
        dataset_name: Optional[str] = None,
    ) -> str:
        PROMPT = UNIFIED_TOOL_CALLING_STRUCTURED_PROMPTS[self.prompt_version]
        system_prompt = PROMPT["system_prompt"]

        if dataset_name in [
            "2wiki",
            "simpleqa",
            "browsecomp",
            "bc_synthetic_depth_one_v2_verified",
            "bc_synthetic_varied_depth_o3_verified",
            "webwalker",
            "hle",
        ]:
            instruction_field_name = "exact_answer"
        elif dataset_name in ["sqav2", "genetic_diseases_qa"]:
            instruction_field_name = "long_form"
        elif dataset_name in ["healthbench", "deep_research_bench", "researchqa"]:
            instruction_field_name = "short_form"
        elif dataset_name and "sft-mix" in dataset_name:
            if "short_form" in dataset_name:
                instruction_field_name = "exact_answer"
            elif "long_form" in dataset_name:
                instruction_field_name = "long_form"  # or "short_form"?
            else:
                raise ValueError(
                    f"Unclear which instruction field name to use for the sft mix dataset: {dataset_name}"
                )
        else:
            if "short_form" in str(dataset_name):
                instruction_field_name = "exact_answer"
            elif "long_form" in str(dataset_name):
                instruction_field_name = "long_form"
            else:
                print("set additional instructions none")
                instruction_field_name = None

        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    question
                    + "\n\n"
                    + PROMPT["additional_instructions"][instruction_field_name]
                    if instruction_field_name is not None
                    else question
                ),
            },
        ]

    def postprocess_output(self, result: Dict[str, Any]) -> str:
        output_string = result.generated_text
        if "</think>" in output_string:
            output_string = "".join(output_string.split("</think>")[1:]).strip()

        if "<answer>" in output_string:
            output_string = (
                output_string.split("<answer>")[1].split("</answer>")[0].strip()
            )

        # Replace the "\boxed{" with "\\boxed{"
        output_string = output_string.replace("\boxed{", "\\boxed{")

        if "\\boxed{" in output_string:
            output_string = output_string.split("\\boxed{")[1].split("}")[0].strip()

        return output_string


@dataclass
class AnswerAgent(BaseAgent):
    prompt_version: str = "v20250907"

    def prompt(self, question: str, history: str, dataset_name: str) -> str:
        PROMPT = UNIFIED_TOOL_CALLING_STRUCTURED_PROMPTS[self.prompt_version]
        if dataset_name in [
            "2wiki",
            "simpleqa",
            "browsecomp",
            "bc_synthetic_depth_one_v2_verified",
            "bc_synthetic_varied_depth_o3_verified",
            "webwalker",
        ]:
            instruction_field_name = "exact_answer"
        elif dataset_name in ["sqav2", "genetic_diseases_qa"]:
            instruction_field_name = "long_form"
        elif dataset_name in ["healthbench", "deep_research_bench", "researchqa"]:
            instruction_field_name = "short_form"
        else:
            if "short_form" in str(dataset_name):
                instruction_field_name = "short_form"
            elif "long_form" in str(dataset_name):
                instruction_field_name = "long_form"
            elif "exact_answer" in str(dataset_name):
                instruction_field_name = "exact_answer"
            else:
                raise ValueError(f"Unknown dataset name: {dataset_name}")

        return [
            {
                "role": "system",
                "content": PROMPT["system_prompt"],
            },
            {
                "role": "user",
                "content": (
                    question
                    + "\n\n"
                    + PROMPT["additional_instructions"][instruction_field_name]
                    if instruction_field_name is not None
                    else question
                ),
            },
            {
                "role": "assistant",
                "content": history,
            },
            {
                "role": "user",
                "content": "Now please generate an answer based on the search results by far.",
            },
        ]

    def postprocess_output(self, result: Dict[str, Any]) -> str:
        output_string = result.generated_text
        if "</think>" in output_string:
            output_string = "".join(output_string.split("</think>")[1:]).strip()

        if "<answer>" in output_string:
            output_string = (
                output_string.split("<answer>")[1].split("</answer>")[0].strip()
            )

        # Replace the "\boxed{" with "\\boxed{"
        output_string = output_string.replace("\boxed{", "\\boxed{")

        if "\\boxed{" in output_string:
            output_string = output_string.split("\\boxed{")[1].split("}")[0].strip()

        return output_string


class NoBrowseTool(BaseTool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def __call__(self, *args, **kwargs):
        return DocumentToolOutput(
            output="Browse tool is not available at this time. Please try other tools.",
            called=True,
            timeout=False,
            runtime=0.0,
            error=None,
            call_id=self._generate_call_id(),
            raw_output=None,
            documents=[],
            tool_name="no_browse",
        )

    def _format_output(self, output: ToolOutput) -> str:
        return output.output

    def _generate_tool_schema(self):
        return {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "URL to browse"}},
            "required": ["url"],
        }
