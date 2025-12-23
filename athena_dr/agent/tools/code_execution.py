import requests
from smolagents import Tool


class CodeExecutionTool(Tool):
    name = "code_execution_tool"
    description = """
    This is a tool that executes code and returns the result"""
    inputs = {
        "code": {
            "type": "string",
            "description": "the code to execute",
        }
    }
    output_type = "object"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executor_url = "http://localhost:8080/run_code"

    def run(self, code: str) -> str:
        return requests.post(
            self.executor_url, json={"code": code, "language": "python"}
        ).json()
