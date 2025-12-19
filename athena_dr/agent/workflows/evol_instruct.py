import os
from dataclasses import dataclass
from random import randint

import weave
from litellm import completion
from omegaconf import OmegaConf

from athena_dr.agent.shared_prompts import DEEPENING_PROMPT, CONCRETIZATION_PROMPT

USER_PROMPT_FORMAT = """
#Given Prompt#:
{prompt}
#Rewritten Prompt#:
"""


@dataclass
class EvolInstructConfig:
    evol_instruct_model_name: str = "azure/gpt-4.1"
    evol_instruct_api_key: str = "dummy-key"
    evol_instruct_base_url: str = "https://soumik-dr.cognitiveservices.azure.com/"
    evol_instruct_max_tokens: int = 32000
    evol_instruct_temperature: float = 0.3
    evol_instruct_max_tool_calls: int = 5


class EvolInstructWorkflow:
    prompts: list[str] = [DEEPENING_PROMPT, CONCRETIZATION_PROMPT]

    def __init__(self, evol_instruct_config_path: os.PathLike, max_iterations: int = 3):
        self.evol_instruct_config = self.get_evol_instruct_config(
            evol_instruct_config_path
        )
        self.max_iterations = max_iterations

    def get_evol_instruct_config(
        self, evol_instruct_config_path: os.PathLike
    ) -> EvolInstructConfig:
        yaml_conf = OmegaConf.load(evol_instruct_config_path)
        resolved_dict = OmegaConf.to_container(yaml_conf, resolve=True)
        evol_instruct_config = EvolInstructConfig(**resolved_dict)
        return evol_instruct_config

    @weave.op
    def rewrite_prompt(self, prompt: str) -> str:
        for idx in range(self.max_iterations):
            prompt_for_rewriting = self.prompts[randint(0, len(self.prompts) - 1)]
            prompt = (
                completion(
                    model=self.evol_instruct_config.evol_instruct_model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt_for_rewriting
                            + "\n\n"
                            + USER_PROMPT_FORMAT.format(prompt=prompt),
                        },
                    ],
                    api_key=self.evol_instruct_config.evol_instruct_api_key,
                    base_url=self.evol_instruct_config.evol_instruct_base_url,
                    max_tokens=self.evol_instruct_config.evol_instruct_max_tokens,
                    temperature=self.evol_instruct_config.evol_instruct_temperature,
                )
                .choices[0]
                .message.content
            )
            if prompt.startswith("#Rewritten Prompt#:"):
                return prompt.replace("#Rewritten Prompt#:", "").strip()
        return prompt
