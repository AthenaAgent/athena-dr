import os
from typing import Optional

import requests
import weave
from pydantic import BaseModel, Field
from smolagents import Tool


class JinaMetadata(BaseModel):
    lang: Optional[str] = None
    viewport: Optional[str] = None


class JinaWebpageResponse(BaseModel):
    url: str
    title: str = ""
    content: str = ""
    description: str = ""
    publishedTime: str = ""
    metadata: Optional[JinaMetadata] = Field(default_factory=lambda: JinaMetadata())
    success: bool = True
    error: Optional[str] = None


def _fetch_webpage_content_jina(
    url: str,
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> JinaWebpageResponse:
    """
    Fetch webpage content using Jina Reader API with JSON format.

    Args:
        url: The URL of the webpage to fetch
        api_key: Jina API key (if not provided, will use JINA_API_KEY env var)
        timeout: Request timeout in seconds (default: 30)

    Returns:
        JinaWebpageResponse containing webpage content and metadata
    """
    if not api_key:
        api_key = os.getenv("JINA_API_KEY")
        if not api_key:
            return JinaWebpageResponse(
                url=url,
                success=False,
                error="JINA_API_KEY environment variable is not set or api_key parameter not provided",
            )

    jina_url = f"https://r.jina.ai/{url}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    try:
        response = requests.get(jina_url, headers=headers, timeout=timeout)

        if response.status_code != 200:
            return JinaWebpageResponse(
                url=url,
                success=False,
                error=f"API request failed with status {response.status_code}: {response.text}",
            )

        json_response = response.json()
        data = json_response.get("data", {})

        metadata_dict = data.get("metadata", {})
        metadata = JinaMetadata(
            lang=metadata_dict.get("lang"),
            viewport=metadata_dict.get("viewport"),
        )

        return JinaWebpageResponse(
            url=data.get("url", url),
            title=data.get("title", ""),
            content=data.get("content", ""),
            description=data.get("description", ""),
            publishedTime=data.get("publishedTime", ""),
            metadata=metadata,
            success=True,
        )
    except Exception as e:
        return JinaWebpageResponse(
            url=url,
            success=False,
            error=str(e),
        )


class JinaFetchTool(Tool):
    name = "jina_fetch_webpage_content"
    description = """
    Fetch the content of a webpage using Jina Reader API with timeout support.
    
    Purpose: Extract clean webpage content as text/markdown using Jina's cloud-based API.
    This tool is useful for fetching articles, documentation, and webpages without requiring browser automation.
    
    Returns content with a snippet ID (e.g., [jina_<url_hash>]) for citation.
    Use this ID to cite sources with <cite id="jina_...">claim</cite> format.
    """
    inputs = {
        "webpage_url": {
            "type": "string",
            "description": "The URL of the webpage to fetch",
        },
        "timeout": {
            "type": "integer",
            "description": "Request timeout in seconds",
            "default": 30,
            "nullable": True,
        },
    }
    output_type = "string"

    @weave.op
    def forward(
        self,
        webpage_url: str,
        timeout: int = 30,
    ) -> str:
        import hashlib

        result = _fetch_webpage_content_jina(
            url=webpage_url,
            api_key=None,
            timeout=timeout,
        )

        # Generate a short hash for the URL to create a unique snippet ID
        url_hash = hashlib.md5(webpage_url.encode()).hexdigest()[:8]
        snippet_id = f"jina_{url_hash}"

        if not result.success:
            return f"[{snippet_id}] Error fetching URL: {webpage_url}\nError: {result.error}"

        # Format output with snippet ID for citation
        formatted_output = [
            f"[{snippet_id}] {result.title or 'Untitled'}",
            f"URL: {result.url}",
        ]

        if result.description:
            formatted_output.append(f"Description: {result.description}")

        if result.publishedTime:
            formatted_output.append(f"Published: {result.publishedTime}")

        formatted_output.append("")
        formatted_output.append(
            result.content if result.content else "No content extracted."
        )

        return "\n".join(formatted_output)
