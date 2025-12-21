import json
import os

import requests
import weave
from smolagents import Tool


class SerperSearchTool(Tool):
    name = "serper_search_tool"
    description = """
    This is a tool that searches the web using serper"""
    inputs = {
        "query": {
            "type": "string",
            "description": "the query to search",
        }
    }
    output_type = "object"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_url = "https://google.serper.dev/search"

    @weave.op
    def forward(self, query: str) -> dict:
        payload = json.dumps({"q": query})
        headers = {
            "X-API-KEY": os.getenv("SERPER_API_KEY"),
            "Content-Type": "application/json",
        }
        response = requests.request(
            "POST", self.base_url, headers=headers, data=payload
        )
        return response.json()
