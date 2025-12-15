import asyncio
import os
from typing import Any, Dict, Optional

from athena_dr.agent.client import LLMToolClient
from athena_dr.agent.tool_interface.chained_tool import ChainedTool
from athena_dr.agent.tool_interface.mcp_tools import (
    Crawl4AIBrowseTool,
    JinaBrowseTool,
    SemanticScholarSnippetSearchTool,
    SerperBrowseTool,
    SerperSearchTool,
)
from athena_dr.agent.workflows.base import BaseWorkflow, BaseWorkflowConfiguration
from athena_dr.agent.workflows.sub_agents import AnswerAgent, NoBrowseTool, SearchAgent
from athena_dr.agent.workflows.web_page_reader import WebPageReaderAgentV2


class AutoReasonSearchWorkflow(BaseWorkflow):
    _default_configuration_path = os.path.join(
        os.path.dirname(__file__), "auto_search.yaml"
    )

    class Configuration(BaseWorkflowConfiguration):
        tool_parser: str

        search_tool_name: str = "serper"

        # Separate generation client (SFT model)
        search_agent_base_url: Optional[str] = None
        search_agent_model_name: str = "dr-tulu/DR-Tulu-8B"
        search_agent_tokenizer_name: str = "Qwen/Qwen3-8B"
        search_agent_api_key: str = "dummy-key"
        search_agent_max_tokens: int = 32000
        search_agent_temperature: float = 0.7
        search_agent_max_tool_calls: int = 10

        use_browse_agent: bool = False
        browse_agent_base_url: Optional[str] = None
        browse_agent_model_name: str = "Qwen/Qwen3-8B"
        browse_agent_tokenizer_name: str = "Qwen/Qwen3-8B"
        browse_agent_api_key: str = "dummy-key"
        browse_agent_max_tokens: int = 32000
        browse_agent_temperature: float = 0.3

        grader_base_url: Optional[str] = None
        grader_model_name: str = "Qwen/Qwen3-8B"
        grader_tokenizer_name: str = "Qwen/Qwen3-8B"
        grader_api_key: str = "dummy-key"
        grader_max_tokens: int = 32000
        grader_temperature: float = 0.3

        # MCP transport configuration
        mcp_transport_type: str = "StreamableHttpTransport"
        mcp_executable: Optional[str] = None
        mcp_port: int = 8000

        # Search configuration
        number_documents_to_search: int = 10
        search_timeout: int = 60

        # Browse configuration
        browse_tool_name: Optional[str] = "crawl4ai"
        browse_timeout: int = 60
        browse_max_pages_to_fetch: int = 10
        browse_context_char_length: int = 6000
        crawl4ai_use_docker_version: bool = False
        crawl4ai_use_ai2_config: bool = False

        prompt_version: str = "v20250907"

    def setup_components(
        self,
        mcp_transport_type: Optional[str] = "StreamableHttpTransport",
        mcp_executable: Optional[str] = None,
        mcp_port: Optional[int] = 8000,
    ) -> None:
        cfg = self.configuration
        assert cfg is not None
        # print(cfg)

        # Allow configuration overrides for MCP settings
        if getattr(cfg, "mcp_transport_type", None):
            mcp_transport_type = cfg.mcp_transport_type
        if getattr(cfg, "mcp_executable", None):
            mcp_executable = cfg.mcp_executable
        if getattr(cfg, "mcp_port", None) is not None:
            mcp_port = cfg.mcp_port

        # Search and browse tools (MCP-backed) with unified tool parser
        if cfg.search_tool_name == "serper":
            self.search_tool = SerperSearchTool(
                tool_parser=cfg.tool_parser,
                number_documents_to_search=cfg.number_documents_to_search,
                timeout=cfg.search_timeout,
                name="snippet_search",  # <- to test this v20250824 model, we need to set the tool name in a hacky way.
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )

            self.search_tool2 = SerperSearchTool(
                tool_parser=cfg.tool_parser,
                number_documents_to_search=cfg.number_documents_to_search,
                timeout=cfg.search_timeout,
                name="google_search",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        elif cfg.search_tool_name == "s2":
            self.search_tool = SemanticScholarSnippetSearchTool(
                tool_parser=cfg.tool_parser,
                number_documents_to_search=cfg.number_documents_to_search,
                timeout=cfg.search_timeout,
                name="snippet_search",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )

            self.search_tool2 = SerperSearchTool(
                tool_parser=cfg.tool_parser,
                number_documents_to_search=cfg.number_documents_to_search,
                timeout=cfg.search_timeout,
                name="google_search",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        elif cfg.search_tool_name == "s2-only":
            self.search_tool = SemanticScholarSnippetSearchTool(
                tool_parser=cfg.tool_parser,
                number_documents_to_search=cfg.number_documents_to_search,
                timeout=cfg.search_timeout,
                name="snippet_search",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )

            self.search_tool2 = SemanticScholarSnippetSearchTool(
                tool_parser=cfg.tool_parser,
                number_documents_to_search=cfg.number_documents_to_search,
                timeout=cfg.search_timeout,
                name="google_search",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        else:
            raise ValueError(f"Invalid search tool name: {cfg.search_tool_name}")

        if cfg.browse_tool_name == "serper":
            self.browse_tool = SerperBrowseTool(
                tool_parser=cfg.tool_parser,
                max_pages_to_fetch=cfg.browse_max_pages_to_fetch,
                timeout=cfg.browse_timeout,
                name="browse_webpage",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        elif cfg.browse_tool_name == "crawl4ai":
            self.browse_tool = Crawl4AIBrowseTool(
                tool_parser=cfg.tool_parser,
                max_pages_to_fetch=cfg.browse_max_pages_to_fetch,
                timeout=cfg.browse_timeout,
                name="browse_webpage",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
                context_chars=cfg.browse_context_char_length,
                use_docker_version=cfg.crawl4ai_use_docker_version,
                use_ai2_config=cfg.crawl4ai_use_ai2_config,
            )
        elif cfg.browse_tool_name == "jina":
            self.browse_tool = JinaBrowseTool(
                tool_parser=cfg.tool_parser,
                timeout=cfg.browse_timeout,
                name="browse_webpage",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        elif cfg.browse_tool_name is None:
            self.browse_tool = NoBrowseTool(
                tool_parser=cfg.tool_parser,
                name="browse_webpage",
            )
        else:
            raise ValueError(f"Invalid browse tool name: {cfg.browse_tool_name}")
        print("Using browse tool: ", self.browse_tool)

        if cfg.use_browse_agent:
            with LLMToolClient(
                model_name=cfg.browse_agent_model_name,
                tokenizer_name=cfg.browse_agent_tokenizer_name,
                base_url=cfg.browse_agent_base_url,
                api_key=cfg.browse_agent_api_key,
            ) as client:
                self.browse_agent = WebPageReaderAgentV2(client=client).as_tool(
                    max_tokens=cfg.browse_agent_max_tokens,
                    temperature=cfg.browse_agent_temperature,
                )
                self.composed_browse_tool = ChainedTool(
                    [self.browse_tool, self.browse_agent],
                    name="browse_webpage",
                    tool_parser=cfg.tool_parser,
                    output_formatting="last",
                )
        else:
            self.composed_browse_tool = self.browse_tool

        with LLMToolClient(
            model_name=cfg.search_agent_model_name,
            tokenizer_name=cfg.search_agent_tokenizer_name,
            base_url=cfg.search_agent_base_url,
            api_key=cfg.search_agent_api_key,
        ) as client:
            self.search_agent = SearchAgent(
                client=client,
                tools=[self.search_tool, self.search_tool2, self.composed_browse_tool],
                prompt_version=cfg.prompt_version,
            )
            self.answer_agent = AnswerAgent(
                client=client,
                prompt_version=cfg.prompt_version,
            )

    async def __call__(
        self,
        problem: str,
        dataset_name: Optional[str] = None,
        verbose: bool = True,
        search_callback: Optional[Any] = None,
        step_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        cfg = self.configuration
        assert cfg is not None

        # import litellm

        # litellm._turn_on_debug()

        # Set the question for the browse agent
        # TODO: This is a bit hectic and hacky, but it works for now
        # The problem: it uses a bad way to enable the runtime dynamics
        if isinstance(self.composed_browse_tool, ChainedTool):
            browse_tool = self.composed_browse_tool.tools[0]
            browse_tool.bm25_query = problem
            browse_agent = self.composed_browse_tool.tools[-1]
            browse_agent.agent.question = problem
        else:
            browse_tool = self.composed_browse_tool
            browse_tool.bm25_query = problem

        results = await self.search_agent(
            question=problem,
            dataset_name=dataset_name,
            max_tokens=cfg.search_agent_max_tokens,
            temperature=cfg.search_agent_temperature,
            max_tool_calls=cfg.search_agent_max_tool_calls,
            verbose=verbose,
            on_step_callback=step_callback,
        )

        if search_callback:
            if asyncio.iscoroutinefunction(search_callback):
                await search_callback(results)
            else:
                search_callback(results)

        browsed_links = []
        searched_links = []
        total_tool_calls = 0
        failed_tool_calls = 0
        failed_tool_call_errors = []
        for tool_output in results.tool_calls:
            total_tool_calls += 1
            if tool_output.error != "":
                failed_tool_calls += 1
                failed_tool_call_errors.append(tool_output.error)

            if tool_output.tool_name in ["snippet_search", "google_search"]:
                searched_links.extend(
                    [document.url for document in tool_output.documents]
                )

            if tool_output.tool_name == "browse_webpage":
                if isinstance(self.composed_browse_tool, ChainedTool):
                    if tool_output.raw_output is None:
                        continue
                    if chained_tool_outputs := tool_output.raw_output.get(
                        "tool_outputs"
                    ):
                        for document in chained_tool_outputs[0].documents:
                            if document.url:
                                browsed_links.append(document.url)
                else:
                    if hasattr(tool_output, "documents"):
                        for document in tool_output.documents:
                            if document.url:
                                browsed_links.append(document.url)
                    else:
                        print(
                            f"Warning: browse_webpage tool output has no documents: {tool_output}"
                        )

        browsed_links = list(set(browsed_links))
        searched_links = list(set(searched_links))

        if "<answer>" in results.generated_text:
            return {
                "final_response": self.search_agent.postprocess_output(results),
                "full_traces": results,
                "browsed_links": browsed_links,
                "searched_links": searched_links,
                "total_tool_calls": total_tool_calls,
                "total_failed_tool_calls": failed_tool_calls,
                "failed_tool_call_errors": failed_tool_call_errors,
            }

        answer = await self.answer_agent(
            question=problem,
            history=results.generated_text,
            dataset_name=dataset_name,
            additional_instructions="Now please generate an based on the search results by far:",
            generation_prefix="<answer>",
            max_tokens=cfg.search_agent_max_tokens,
            temperature=cfg.search_agent_temperature,
            verbose=verbose,
            on_step_callback=step_callback,
        )

        if verbose:
            print(results)  # noqa: T201

        answer.tool_calls = [results.model_dump()]

        return {
            "final_response": self.answer_agent.postprocess_output(answer),
            "full_traces": answer,
            "browsed_links": browsed_links,
            "searched_links": searched_links,
        }


if __name__ == "__main__":
    AutoReasonSearchWorkflow.app()
