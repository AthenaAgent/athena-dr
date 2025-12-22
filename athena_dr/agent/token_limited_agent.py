from typing import Any

from smolagents import ToolCallingAgent
from smolagents.utils import AgentMaxStepsError


class TokenLimitedToolCallingAgent(ToolCallingAgent):
    """
    A ToolCallingAgent that stops execution when max_output_tokens is reached
    instead of using max_steps.

    This agent modifies the max_steps check to use token limits instead.
    """

    def __init__(
        self,
        max_output_tokens: int = 10000,
        *args,
        **kwargs,
    ):
        # Set a high max_steps as fallback safety to prevent infinite loops
        if "max_steps" not in kwargs:
            kwargs["max_steps"] = 1000

        super().__init__(*args, **kwargs)
        self.max_output_tokens = max_output_tokens

    def step(self, *args, **kwargs) -> Any:
        """
        Override step to check token limits before executing each step.
        Raises AgentMaxStepsError if token limit is exceeded.
        """
        # Check if we've exceeded the max output tokens limit
        current_output_tokens = self.monitor.total_output_token_count

        if current_output_tokens >= self.max_output_tokens:
            error_msg = (
                f"Reached maximum output tokens: {current_output_tokens}/{self.max_output_tokens}. "
                f"Stopping agent execution."
            )
            self.logger.log(error_msg, level=1)
            raise AgentMaxStepsError(error_msg, self.logger)

        # Call the parent's step method
        return super().step(*args, **kwargs)
