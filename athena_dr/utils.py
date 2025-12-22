import os
from dataclasses import dataclass

from omegaconf import OmegaConf


@dataclass
class WorkflowConfig:
    model_name: str
    api_key: str
    base_url: str
    max_tokens: int
    temperature: float
    agent_max_steps: int
    agent_name: str
    max_output_tokens: int = 30000


def get_config(config_path: os.PathLike) -> WorkflowConfig:
    yaml_conf = OmegaConf.load(config_path)
    resolved_dict = OmegaConf.to_container(yaml_conf, resolve=True)
    config = WorkflowConfig(**resolved_dict)
    return config
