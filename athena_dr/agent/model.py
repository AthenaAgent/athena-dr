import json
import re
import xml.etree.ElementTree as ET
from typing import Any

import weave
from smolagents import ChatMessage, OpenAIModel, TokenUsage
from smolagents.tools import Tool


class OpenAIModelWithThinkingTraces(OpenAIModel):
    def _prepare_completion_kwargs(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Override to remove tools parameter for XML-based tool calling."""
        completion_kwargs = super()._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=None,  # Don't send tools to the API
            **kwargs,
        )
        # Store tools for reference but don't send them in the request
        self._current_tools = tools_to_call_from
        return completion_kwargs

    def _parse_xml_tool_calls(
        self, content: str
    ) -> tuple[str, list[dict[str, Any]] | None]:
        """Parse XML-formatted tool calls and convert to OpenAI format.

        Returns:
            Tuple of (cleaned_content, tool_calls_list or None)
        """
        if not content or "<tool_call>" not in content:
            return content, None

        tool_calls = []
        cleaned_content = content

        # Extract all tool_call blocks
        tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.finditer(tool_call_pattern, content, re.DOTALL)

        for idx, match in enumerate(matches):
            tool_call_xml = match.group(1).strip()

            try:
                # Parse the XML content
                root = ET.fromstring(f"<root>{tool_call_xml}</root>")

                # Extract tool name
                tool_name_elem = root.find("tool_name")
                if tool_name_elem is None or not tool_name_elem.text:
                    continue

                tool_name = tool_name_elem.text.strip()

                # Extract arguments (all other elements are arguments)
                arguments = {}
                for child in root:
                    if child.tag != "tool_name" and child.text:
                        arguments[child.tag] = child.text.strip()

                # Create OpenAI-style tool call
                tool_call = {
                    "id": f"call_{idx}",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": json.dumps(arguments)},
                }
                tool_calls.append(tool_call)

            except ET.ParseError:
                # If XML parsing fails, try simple regex extraction
                tool_name_match = re.search(
                    r"<tool_name>(.*?)</tool_name>", tool_call_xml
                )
                if tool_name_match:
                    tool_name = tool_name_match.group(1).strip()
                    arguments = {}

                    # Extract all argument tags
                    arg_pattern = r"<([^/>]+)>(.*?)</\1>"
                    for arg_match in re.finditer(arg_pattern, tool_call_xml):
                        arg_name = arg_match.group(1)
                        if arg_name != "tool_name":
                            arguments[arg_name] = arg_match.group(2).strip()

                    tool_call = {
                        "id": f"call_{idx}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(arguments),
                        },
                    }
                    tool_calls.append(tool_call)

        # Remove tool_call blocks from content
        if tool_calls:
            cleaned_content = re.sub(
                tool_call_pattern, "", content, flags=re.DOTALL
            ).strip()

        return cleaned_content, tool_calls if tool_calls else None

    @weave.op
    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs,
    ) -> ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        self._apply_rate_limit()
        response = self.retryer(
            self.client.chat.completions.create, **completion_kwargs
        )
        content = response.choices[0].message.content
        reasoning_trace = getattr(response.choices[0].message, "reasoning", None)
        content = (
            f"<thinking>{reasoning_trace}</thinking>\n\n{content}"
            if reasoning_trace is not None
            else content
        )
        content = content.strip()
        if stop_sequences is not None and not self.supports_stop_parameter:
            content = remove_content_after_stop_sequences(content, stop_sequences)

        # Parse XML tool calls if present and no JSON tool calls exist
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls is None and content:
            content, parsed_tool_calls = self._parse_xml_tool_calls(content)
            if parsed_tool_calls:
                # Convert to the format expected by smolagents
                from openai.types.chat import ChatCompletionMessageToolCall
                from openai.types.chat.chat_completion_message_tool_call import Function

                tool_calls = [
                    ChatCompletionMessageToolCall(
                        id=tc["id"],
                        type=tc["type"],
                        function=Function(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                    )
                    for tc in parsed_tool_calls
                ]

        return ChatMessage(
            role=response.choices[0].message.role,
            content=content,
            tool_calls=tool_calls,
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )
