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
    max_tool_calls: int


def get_config(config_path: os.PathLike) -> WorkflowConfig:
    yaml_conf = OmegaConf.load(config_path)
    resolved_dict = OmegaConf.to_container(yaml_conf, resolve=True)
    config = WorkflowConfig(**resolved_dict)
    return config
