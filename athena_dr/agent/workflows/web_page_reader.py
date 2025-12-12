import asyncio
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
from urllib.parse import quote, unquote, urlparse

import weave
from bs4 import BeautifulSoup

from athena_dr.agent.agent_interface import BaseAgent
from athena_dr.agent.client import DocumentToolOutput, LLMToolClient

# Maximum characters to send to the LLM (~8k tokens assuming ~4 chars/token)
MAX_TEXT_CHARS = 32000


@dataclass
class WebPageReaderAgentV2(BaseAgent):
    question: Optional[str] = None
    prompt = """
We are searching on the internet for the following question:
{question}
Here is some webpage scraped from the internet:
{document}
Can you clean the raw webpage text and convert it into a more readable format? You should remove all the unnecessary information and keep the main content of the page. Please produce the output in the format of "Cleaned webpage text:\n[you text here]".
""".strip()

    def preprocess_input(self, documents: Union[str, Any]) -> Dict[str, str]:
        # Accept either a raw string or a ToolOutput-like object with an `output` attribute
        assert self.question is not None, "Question is not set"

        if isinstance(documents, DocumentToolOutput):
            # print("using DocumentToolOutput")
            doc_str = "\n".join(
                [
                    document.simple_stringify()[: 32000 * 4 // len(documents.documents)]
                    for document in documents.documents
                ]
            )
        elif hasattr(documents, "output"):
            doc_str = documents.output
        else:
            doc_str = documents if isinstance(documents, str) else str(documents)
        input_params = {"question": self.question, "document": doc_str}
        # print(input_params)
        return input_params

    def postprocess_output(self, result: Dict[str, Any]) -> str:
        output_string = result.generated_text
        if "</think>" in output_string:
            output_string = "".join(output_string.split("</think>")[1:]).strip()

        if "Cleaned webpage text:" in output_string:
            output_string = output_string.split("Cleaned webpage text:")[1].strip()

        return output_string


def _extract_text_from_html(html: str) -> str:
    """
    Parse HTML and extract readable text content, removing scripts, styles, etc.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove non-content elements
    for tag in soup(
        ["script", "style", "nav", "footer", "header", "aside", "noscript"]
    ):
        tag.decompose()

    # Get text with reasonable whitespace handling
    text = soup.get_text(separator=" ", strip=True)

    # Collapse multiple whitespace into single spaces
    text = " ".join(text.split())

    return text


@weave.op()
def run_web_page_reader(question: str, urls: list[str]) -> list[Any]:
    def get_text_from_url(url: str) -> str:
        """
        Fetch the URL and return extracted text content (not raw HTML).
        """
        headers = {
            # Many sites (including Wikipedia) reject requests without a UA.
            "User-Agent": "athena-dr/0.1 (+https://github.com; contact: dev@localhost) python-urllib",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        def fetch(target_url: str) -> str:
            req = urllib.request.Request(target_url, headers=headers, method="GET")
            with urllib.request.urlopen(req, timeout=30) as response:
                html = response.read()
                return html.decode("utf-8", errors="replace")

        try:
            html = fetch(url)
        except urllib.error.HTTPError as e:
            # Wikipedia frequently 403s "generic" scraping requests. Use the official
            # REST HTML endpoint as a fallback when the input URL is a /wiki/<title> page.
            if e.code == 403:
                parsed = urlparse(url)
                if parsed.netloc.endswith("wikipedia.org") and parsed.path.startswith(
                    "/wiki/"
                ):
                    title = unquote(parsed.path[len("/wiki/") :])
                    rest_url = (
                        f"{parsed.scheme}://{parsed.netloc}/api/rest_v1/page/html/"
                        f"{quote(title, safe='')}"
                    )
                    html = fetch(rest_url)
                else:
                    raise
            else:
                raise

        # Extract text and truncate to fit model context
        text = _extract_text_from_html(html)
        return text[:MAX_TEXT_CHARS]

    text_contents = [get_text_from_url(url) for url in urls]
    results = []
    with LLMToolClient(model_name="openrouter/openai/gpt-4o") as client:
        for text_content in text_contents:
            agent = WebPageReaderAgentV2(client=client, question=question)
            reader_tool = agent.as_tool()
            tool_output = asyncio.run(reader_tool(text_content))
            results.append(tool_output)
    return results
