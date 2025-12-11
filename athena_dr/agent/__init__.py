__version__ = "0.0.1"

# Main library components
from athena_dr.agent.agent_interface import BaseAgent
from athena_dr.agent.client import (
    GenerateWithToolsOutput,
    GenerationConfig,
    LLMToolClient,
)

# Shared prompts
from athena_dr.agent.shared_prompts import UNIFIED_TOOL_CALLING_PROMPTS

# Tool interface components
from athena_dr.agent.tool_interface import (
    AgentAsTool,
    BaseTool,
    ChainedTool,
    Document,
    DocumentToolOutput,
    MassiveServeSearchTool,
    MCPMixin,
    SemanticScholarSnippetSearchTool,
    SerperBrowseTool,
    SerperSearchTool,
    ToolCallInfo,
    ToolCallParser,
    ToolInput,
    ToolOutput,
    VllmHostedRerankerTool,
)
from athena_dr.agent.workflow import BaseWorkflow, BaseWorkflowConfiguration

__all__ = [
    # Core components
    "BaseAgent",
    "LLMToolClient",
    "GenerateWithToolsOutput",
    "GenerationConfig",
    "BaseWorkflow",
    "BaseWorkflowConfiguration",
    # Tool interface - Core
    "BaseTool",
    "ToolInput",
    "ToolOutput",
    "AgentAsTool",
    "ChainedTool",
    # Tool interface - Data types
    "Document",
    "DocumentToolOutput",
    # Tool interface - MCP Tools
    "MCPMixin",
    "SemanticScholarSnippetSearchTool",
    "SerperSearchTool",
    "MassiveServeSearchTool",
    "SerperBrowseTool",
    "VllmHostedRerankerTool",
    # Tool interface - Parsing
    "ToolCallInfo",
    "ToolCallParser",
    # Prompts
    "UNIFIED_TOOL_CALLING_PROMPTS",
]
