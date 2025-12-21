import asyncio
from threading import Thread
from typing import Dict, Optional

import weave
from pydantic import BaseModel, Field
from smolagents import Tool


class Crawl4AiResult(BaseModel):
    url: str
    success: bool
    markdown: str
    fit_markdown: Optional[str] = Field(
        None, description="Only present when content filtering is used"
    )
    html: Optional[str] = Field(
        None, description="Only present when include_html=True or markdown is empty"
    )
    error: Optional[str] = Field(None, description="Only present when success=False")


async def _fetch_markdown_async(
    url: str,
    query: Optional[str] = None,
    ignore_links: bool = True,
    use_pruning: bool = False,
    bypass_cache: bool = True,
    headless: bool = True,
    timeout_ms: int = 60000,
    include_html: bool = False,
) -> Crawl4AiResult:
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig
        from crawl4ai.content_filter_strategy import (
            BM25ContentFilter,
            PruningContentFilter,
        )
        from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

        content_filter = None
        if query:
            try:
                content_filter = BM25ContentFilter(
                    user_query=query, bm25_threshold=1.2, language="english"
                )
            except Exception:
                content_filter = None
        elif use_pruning:
            try:
                content_filter = PruningContentFilter(
                    threshold=0.5, threshold_type="fixed", min_word_threshold=50
                )
            except Exception:
                content_filter = None

        md_generator = (
            DefaultMarkdownGenerator(options={"ignore_links": ignore_links})
            if content_filter is None
            else DefaultMarkdownGenerator(
                content_filter=content_filter,
                options={"ignore_links": ignore_links},
            )
        )

        browser_conf = BrowserConfig(headless=headless)
        run_conf = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS if bypass_cache else CacheMode.ENABLED,
            page_timeout=timeout_ms,
            markdown_generator=md_generator,
            exclude_social_media_links=True,
            excluded_tags=["form", "header", "footer", "nav"],
            exclude_domains=["ads.com", "spammytrackers.net"],
            word_count_threshold=10,
        )

        async with AsyncWebCrawler(config=browser_conf) as crawler:
            result = await crawler.arun(url=url, config=run_conf)

        if not getattr(result, "success", True):
            return Crawl4AiResult(
                url=getattr(result, "url", url),
                success=False,
                markdown="",
                html=getattr(result, "html", ""),
                error=getattr(result, "error_message", "Unknown error"),
            )

        md_value = ""
        fit_markdown_value = None
        md_obj = getattr(result, "markdown", None)
        if isinstance(md_obj, str):
            md_value = md_obj
        else:
            fit_markdown_value = getattr(md_obj, "fit_markdown", None)
            raw_markdown = getattr(md_obj, "raw_markdown", None)
            if fit_markdown_value:
                md_value = fit_markdown_value
            elif raw_markdown:
                md_value = raw_markdown
            else:
                md_value = str(md_obj) if md_obj is not None else ""

        response_data = {
            "url": getattr(result, "url", url),
            "success": True,
            "markdown": md_value,
        }
        if fit_markdown_value:
            response_data["fit_markdown"] = fit_markdown_value
        if include_html or not md_value:
            response_data["html"] = getattr(result, "html", "")

        return Crawl4AiResult(**response_data)
    except Exception as e:
        return Crawl4AiResult(
            url=url,
            success=False,
            markdown="",
            html="",
            error=str(e),
        )


def _fetch_markdown_sync(
    url: str,
    query: Optional[str] = None,
    ignore_links: bool = True,
    use_pruning: bool = False,
    bypass_cache: bool = True,
    headless: bool = True,
    timeout_ms: int = 80000,
    include_html: bool = False,
) -> Crawl4AiResult:
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            result_holder: Dict[str, Crawl4AiResult] = {}

            def _runner():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    result_holder["result"] = new_loop.run_until_complete(
                        _fetch_markdown_async(
                            url=url,
                            query=query,
                            ignore_links=ignore_links,
                            use_pruning=use_pruning,
                            bypass_cache=bypass_cache,
                            headless=headless,
                            timeout_ms=timeout_ms,
                            include_html=include_html,
                        )
                    )
                finally:
                    new_loop.close()

            th = Thread(target=_runner)
            th.start()
            th.join()
            return result_holder.get(
                "result",
                Crawl4AiResult(
                    url=url,
                    success=False,
                    markdown="",
                    html="",
                    error="Unknown error",
                ),
            )
        else:
            return asyncio.run(
                _fetch_markdown_async(
                    url=url,
                    query=query,
                    ignore_links=ignore_links,
                    use_pruning=use_pruning,
                    bypass_cache=bypass_cache,
                    headless=headless,
                    timeout_ms=timeout_ms,
                    include_html=include_html,
                )
            )
    except Exception as e:
        return Crawl4AiResult(
            url=url,
            success=False,
            markdown="",
            html="",
            error=str(e),
        )


class Crawl4AIFetchTool(Tool):
    name = "crawl4ai_fetch_webpage_content"
    description = """
    Open a specific URL and extract readable page text as snippets using Crawl4AI.
    
    Purpose: Fetch and parse webpage content (typically URLs returned from google_search) to extract clean, readable text.
    This tool is useful for opening articles, documentation, and webpages to read their full content.
    """
    inputs = {
        "url": {
            "type": "string",
            "description": "URL to fetch and extract content from",
        },
        "ignore_links": {
            "type": "boolean",
            "description": "If True, remove hyperlinks in markdown",
            "default": True,
            "nullable": True,
        },
        "use_pruning": {
            "type": "boolean",
            "description": "Apply pruning content filter to extract main content (used when bm25_query is not provided)",
            "default": False,
            "nullable": True,
        },
        "bm25_query": {
            "type": "string",
            "description": "Optional query to enable BM25-based content filtering for focused extraction",
            "nullable": True,
        },
        "bypass_cache": {
            "type": "boolean",
            "description": "If True, bypass Crawl4AI cache",
            "default": True,
            "nullable": True,
        },
        "timeout_ms": {
            "type": "integer",
            "description": "Per-page timeout in milliseconds",
            "default": 80000,
            "nullable": True,
        },
        "include_html": {
            "type": "boolean",
            "description": "Whether to include raw HTML in the response",
            "default": False,
            "nullable": True,
        },
    }
    output_type = "object"

    @weave.op
    def forward(
        self,
        url: str,
        ignore_links: bool = True,
        use_pruning: bool = False,
        bm25_query: Optional[str] = None,
        bypass_cache: bool = True,
        timeout_ms: int = 80000,
        include_html: bool = False,
    ) -> dict:
        result = _fetch_markdown_sync(
            url=url,
            query=bm25_query,
            ignore_links=ignore_links,
            use_pruning=use_pruning,
            bypass_cache=bypass_cache,
            headless=True,
            timeout_ms=timeout_ms,
            include_html=include_html,
        )
        return result.model_dump()
