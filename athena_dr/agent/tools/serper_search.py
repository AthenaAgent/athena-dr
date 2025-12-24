import json
import os

import requests
import weave
from smolagents import Tool


class SerperSearchTool(Tool):
    name = "serper_search_tool"
    description = """
    This is a tool that searches the web using serper.
    Returns search results with snippet IDs for citation (e.g., [serper_1], [serper_2]).
    Use these IDs to cite sources in your answer with <cite id="serper_1">claim</cite> format."""
    inputs = {
        "query": {
            "type": "string",
            "description": "the query to search",
        }
    }
    output_type = "string"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_url = "https://google.serper.dev/search"

    @weave.op
    def forward(self, query: str) -> str:
        payload = json.dumps({"q": query})
        headers = {
            "X-API-KEY": os.getenv("SERPER_API_KEY"),
            "Content-Type": "application/json",
        }
        response = requests.request(
            "POST", self.base_url, headers=headers, data=payload
        )
        data = response.json()

        # Format output with snippet IDs for citation
        formatted_results = []

        # Add answer box if present (highest priority)
        if "answerBox" in data:
            ab = data["answerBox"]
            snippet_id = "serper_answer"
            answer_text = ab.get("answer", "") or ab.get("snippet", "")
            formatted_results.append(
                f"[{snippet_id}] Answer Box: {ab.get('title', 'Direct Answer')}\n"
                f"Answer: {answer_text}\n"
                f"URL: {ab.get('link', 'N/A')}\n"
            )

        # Add knowledge graph if present
        if "knowledgeGraph" in data:
            kg = data["knowledgeGraph"]
            snippet_id = "serper_kg"
            kg_info = [f"[{snippet_id}] Knowledge Graph: {kg.get('title', 'N/A')}"]
            if kg.get("type"):
                kg_info.append(f"Type: {kg.get('type')}")
            if kg.get("description"):
                kg_info.append(f"Description: {kg.get('description')}")
            if kg.get("website"):
                kg_info.append(f"URL: {kg.get('website')}")
            # Add attributes if present
            if kg.get("attributes"):
                for attr_key, attr_val in kg["attributes"].items():
                    kg_info.append(f"{attr_key}: {attr_val}")
            formatted_results.append("\n".join(kg_info) + "\n")

        # Add organic results with IDs
        if "organic" in data:
            for idx, result in enumerate(data["organic"], start=1):
                snippet_id = f"serper_{idx}"
                formatted_results.append(
                    f"[{snippet_id}] {result.get('title', 'N/A')}\n"
                    f"URL: {result.get('link', 'N/A')}\n"
                    f"Snippet: {result.get('snippet', 'N/A')}\n"
                )

        return (
            "\n".join(formatted_results) if formatted_results else "No results found."
        )
