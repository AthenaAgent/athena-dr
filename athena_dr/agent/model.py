import json
import re
import xml.etree.ElementTree as ET
from typing import Any

import weave
from smolagents import ChatMessage, OpenAIModel, TokenUsage
from smolagents.models import remove_content_after_stop_sequences
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

    def _parse_bracket_tool_calls(
        self, content: str
    ) -> tuple[str, list[dict[str, Any]] | None]:
        """Parse [TOOL_CALL]...[/TOOL_CALL] formatted tool calls.

        Supports multiple formats:
        1. Arrow notation with CLI args:
           [TOOL_CALL]
           {tool => "tool_name", args => {
             --param1 value1
             --param2 "value with spaces"
           }}
           [/TOOL_CALL]

        2. JSON-like with single quotes:
           [TOOL_CALL]
           { 'name': 'tool_name', 'args': {
             'param1': 'value1',
             'param2': 10
           }}
           [/TOOL_CALL]

        Returns:
            Tuple of (cleaned_content, tool_calls_list or None)
        """
        if not content or "[TOOL_CALL]" not in content:
            return content, None

        tool_calls = []
        cleaned_content = content

        # Pattern to match [TOOL_CALL]...[/TOOL_CALL] blocks
        pattern = r"\[TOOL_CALL\](.*?)\[/TOOL_CALL\]"
        matches = list(re.finditer(pattern, content, re.DOTALL))

        for idx, match in enumerate(matches):
            block = match.group(1).strip()
            tool_name = None
            arguments = {}

            try:
                # Try Format 1: Arrow notation {tool => "tool_name", args => {...}}
                tool_name_match = re.search(r'\{\s*tool\s*=>\s*"([^"]+)"', block)
                if tool_name_match:
                    tool_name = tool_name_match.group(1)

                    # Extract arguments from the args => {...} block
                    args_match = re.search(
                        r"args\s*=>\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}",
                        block,
                        re.DOTALL,
                    )

                    if args_match:
                        args_content = args_match.group(1)
                        # Parse CLI-style arguments: --param value or --param "value"
                        arg_pattern = r'--(\w+)\s+(?:"([^"]*)"|([\S]+))'
                        for arg_match in re.finditer(arg_pattern, args_content):
                            param_name = arg_match.group(1)
                            param_value = (
                                arg_match.group(2)
                                if arg_match.group(2) is not None
                                else arg_match.group(3)
                            )
                            if param_value.isdigit():
                                arguments[param_name] = int(param_value)
                            else:
                                arguments[param_name] = param_value

                # Try Format 2: JSON-like with single quotes {'name': 'tool_name', 'args': {...}}
                if tool_name is None:
                    # Convert single quotes to double quotes for JSON parsing
                    json_block = block.replace("'", '"')
                    # Try to parse as JSON
                    try:
                        data = json.loads(json_block)
                        if isinstance(data, dict):
                            tool_name = data.get("name")
                            args_data = data.get("args", {})
                            if isinstance(args_data, dict):
                                arguments = args_data
                    except json.JSONDecodeError:
                        # If JSON parsing fails, try regex extraction
                        name_match = re.search(
                            r"['\"]name['\"]\s*:\s*['\"]([^'\"]+)['\"]", block
                        )
                        if name_match:
                            tool_name = name_match.group(1)

                            # Extract args using regex
                            args_section = re.search(
                                r"['\"]args['\"]\s*:\s*\{([^}]*)\}", block, re.DOTALL
                            )
                            if args_section:
                                args_content = args_section.group(1)
                                # Parse key-value pairs: 'key': 'value' or 'key': 123
                                kv_pattern = r"['\"](\w+)['\"]\s*:\s*(?:['\"]([^'\"]*)['\"]|(\d+))"
                                for kv_match in re.finditer(kv_pattern, args_content):
                                    key = kv_match.group(1)
                                    if kv_match.group(2) is not None:
                                        arguments[key] = kv_match.group(2)
                                    elif kv_match.group(3) is not None:
                                        arguments[key] = int(kv_match.group(3))

                if tool_name:
                    tool_call = {
                        "id": f"call_{idx}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(arguments),
                        },
                    }
                    tool_calls.append(tool_call)

            except Exception:
                continue

        if tool_calls:
            cleaned_content = re.sub(pattern, "", content, flags=re.DOTALL).strip()

        return cleaned_content, tool_calls if tool_calls else None

    def _parse_action_tool_calls(
        self, content: str
    ) -> tuple[str, list[dict[str, Any]] | None]:
        """Parse Action: {...} formatted tool calls.

        Format example:
        Action:
        {
          "name": "semantic_scholar_paper_search",
          "arguments": {"query": "search terms", "limit": 10}
        }

        Returns:
            Tuple of (cleaned_content, tool_calls_list or None)
        """
        if not content or "Action:" not in content:
            return content, None

        tool_calls = []
        cleaned_content = content

        # Pattern to match Action: followed by JSON
        # Match Action: then whitespace then { ... }
        action_pattern = r"Action:\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}"
        matches = list(re.finditer(action_pattern, content, re.DOTALL))

        for idx, match in enumerate(matches):
            try:
                json_content = "{" + match.group(1) + "}"

                # Try to parse as JSON
                try:
                    data = json.loads(json_content)
                except json.JSONDecodeError:
                    # If fails, continue to regex extraction
                    data = None

                if data and isinstance(data, dict):
                    tool_name = data.get("name")
                    arguments = data.get("arguments", {})

                    if tool_name:
                        tool_call = {
                            "id": f"call_{idx}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(arguments)
                                if isinstance(arguments, dict)
                                else arguments,
                            },
                        }
                        tool_calls.append(tool_call)
                else:
                    # Try regex extraction for malformed JSON
                    block = match.group(0)
                    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', block)
                    if name_match:
                        tool_name = name_match.group(1)

                        # Try to extract arguments
                        arguments = {}
                        args_match = re.search(
                            r'"arguments"\s*:\s*\{([^}]*)\}', block, re.DOTALL
                        )
                        if args_match:
                            args_content = args_match.group(1)
                            # Extract simple key-value pairs
                            kv_pattern = r'"(\w+)"\s*:\s*(?:"([^"]*)"|([\d.]+)|(\w+))'
                            for kv in re.finditer(kv_pattern, args_content):
                                key = kv.group(1)
                                if kv.group(2) is not None:
                                    arguments[key] = kv.group(2)
                                elif kv.group(3) is not None:
                                    val = kv.group(3)
                                    arguments[key] = (
                                        int(val) if val.isdigit() else float(val)
                                    )
                                elif kv.group(4) is not None:
                                    arguments[key] = kv.group(4)

                        tool_call = {
                            "id": f"call_{idx}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                        tool_calls.append(tool_call)

            except Exception:
                continue

        if tool_calls:
            cleaned_content = re.sub(
                action_pattern, "", content, flags=re.DOTALL
            ).strip()
            # Also remove trailing <cite> tags that follow actions
            cleaned_content = re.sub(
                r"<cite\s+id=['\"][^'\"]*['\"]\s*>\s*</cite>\s*---?",
                "",
                cleaned_content,
            ).strip()

        return cleaned_content, tool_calls if tool_calls else None

    def _parse_json_tool_calls(
        self, content: str
    ) -> tuple[str, list[dict[str, Any]] | None]:
        """Parse JSON-formatted tool calls in various formats.

        Handles formats like:
        - [Event: {"tool_calls": [...]}]
        - {"tool": "name", "args": {...}}

        Returns:
            Tuple of (cleaned_content, tool_calls_list or None)
        """
        if not content:
            return content, None

        tool_calls = []
        cleaned_content = content

        # Pattern 1: [Event: {"tool_calls": [...]}] format
        event_pattern = r"\[Event:\s*(\{[^]]*\})\s*\]"
        event_matches = list(re.finditer(event_pattern, content, re.DOTALL))

        for idx, match in enumerate(event_matches):
            try:
                json_str = match.group(1)
                data = json.loads(json_str)

                if "tool_calls" in data:
                    calls = data["tool_calls"]
                    if isinstance(calls, list):
                        for call in calls:
                            if isinstance(call, dict):
                                # Handle {"query": "...", "search_type": "..."} format
                                tool_name = call.pop("search_type", None) or call.pop(
                                    "tool", None
                                )
                                if tool_name:
                                    # Map search_type to actual tool names
                                    tool_name_map = {
                                        "scholarly_search": "semantic_scholar_paper_search",
                                        "web_search": "serper_search_tool",
                                    }
                                    tool_name = tool_name_map.get(tool_name, tool_name)

                                    tool_call = {
                                        "id": f"call_{len(tool_calls)}",
                                        "type": "function",
                                        "function": {
                                            "name": tool_name,
                                            "arguments": json.dumps(call),
                                        },
                                    }
                                    tool_calls.append(tool_call)
                    elif isinstance(calls, dict):
                        # Single tool call as dict (e.g., webpage_url)
                        if "webpage_url" in calls:
                            tool_call = {
                                "id": f"call_{len(tool_calls)}",
                                "type": "function",
                                "function": {
                                    "name": "crawl4ai_fetch_webpage_content",
                                    "arguments": json.dumps(
                                        {"url": calls["webpage_url"]}
                                    ),
                                },
                            }
                            tool_calls.append(tool_call)
                elif "webpage_url" in data:
                    tool_call = {
                        "id": f"call_{len(tool_calls)}",
                        "type": "function",
                        "function": {
                            "name": "crawl4ai_fetch_webpage_content",
                            "arguments": json.dumps({"url": data["webpage_url"]}),
                        },
                    }
                    tool_calls.append(tool_call)

            except json.JSONDecodeError:
                continue

        if tool_calls:
            cleaned_content = re.sub(
                event_pattern, "", content, flags=re.DOTALL
            ).strip()

        return cleaned_content, tool_calls if tool_calls else None

    def _parse_invoke_tool_calls(
        self, content: str
    ) -> tuple[str, list[dict[str, Any]] | None]:
        """Parse <invoke><tool_name>...</tool_name></invoke> formatted tool calls.

        Format example:
        <invoke><semantic_scholar_paper_search>
        <query>search terms</query>
        <year>2020-2025</year>
        <limit>15</limit>
        </semantic_scholar_paper_search></invoke>

        Returns:
            Tuple of (cleaned_content, tool_calls_list or None)
        """
        if not content or "<invoke>" not in content:
            return content, None

        tool_calls = []
        cleaned_content = content

        # Pattern to match <invoke>...</invoke> blocks
        invoke_pattern = r"<invoke>(.*?)</invoke>"
        matches = list(re.finditer(invoke_pattern, content, re.DOTALL))

        for idx, match in enumerate(matches):
            invoke_content = match.group(1).strip()

            try:
                # Clean up escaped newlines that might be in the content
                invoke_content = invoke_content.replace("\\n", "\n")

                # Find the tool name - it's the outermost tag inside <invoke>
                # Pattern: <tool_name>...</tool_name>
                tool_tag_match = re.match(
                    r"<([a-zA-Z_][a-zA-Z0-9_]*)>(.*?)</\1>",
                    invoke_content,
                    re.DOTALL,
                )

                if not tool_tag_match:
                    continue

                tool_name = tool_tag_match.group(1)
                inner_content = tool_tag_match.group(2)

                # Extract arguments from inner XML tags
                arguments = {}
                arg_pattern = r"<([a-zA-Z_][a-zA-Z0-9_]*)>(.*?)</\1>"
                for arg_match in re.finditer(arg_pattern, inner_content, re.DOTALL):
                    arg_name = arg_match.group(1)
                    arg_value = arg_match.group(2).strip()

                    # Skip empty values
                    if not arg_value:
                        continue

                    # Convert numeric strings to appropriate types
                    if arg_value.isdigit():
                        arguments[arg_name] = int(arg_value)
                    else:
                        arguments[arg_name] = arg_value

                tool_call = {
                    "id": f"call_{idx}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments),
                    },
                }
                tool_calls.append(tool_call)

            except Exception:
                continue

        if tool_calls:
            cleaned_content = re.sub(
                invoke_pattern, "", content, flags=re.DOTALL
            ).strip()

        return cleaned_content, tool_calls if tool_calls else None

    def _parse_all_tool_calls(
        self, content: str
    ) -> tuple[str, list[dict[str, Any]] | None]:
        """Parse tool calls from content using all supported formats.

        Tries parsing in order:
        1. XML format: <tool_call>...</tool_call>
        2. Invoke format: <invoke><tool_name>...</tool_name></invoke>
        3. Bracket format: [TOOL_CALL]...[/TOOL_CALL]
        4. Action format: Action: {...}
        5. JSON/Event format: [Event: {...}]

        Returns:
            Tuple of (cleaned_content, combined_tool_calls_list or None)
        """
        if not content:
            return content, None

        all_tool_calls = []

        # Try XML format first
        content, xml_tool_calls = self._parse_xml_tool_calls(content)
        if xml_tool_calls:
            all_tool_calls.extend(xml_tool_calls)

        # Try invoke format
        content, invoke_tool_calls = self._parse_invoke_tool_calls(content)
        if invoke_tool_calls:
            # Re-number IDs to avoid conflicts
            for i, tc in enumerate(invoke_tool_calls):
                tc["id"] = f"call_{len(all_tool_calls) + i}"
            all_tool_calls.extend(invoke_tool_calls)

        # Try bracket/arrow format
        content, bracket_tool_calls = self._parse_bracket_tool_calls(content)
        if bracket_tool_calls:
            # Re-number IDs to avoid conflicts
            for i, tc in enumerate(bracket_tool_calls):
                tc["id"] = f"call_{len(all_tool_calls) + i}"
            all_tool_calls.extend(bracket_tool_calls)

        # Try Action format
        content, action_tool_calls = self._parse_action_tool_calls(content)
        if action_tool_calls:
            # Re-number IDs to avoid conflicts
            for i, tc in enumerate(action_tool_calls):
                tc["id"] = f"call_{len(all_tool_calls) + i}"
            all_tool_calls.extend(action_tool_calls)

        # Try JSON/Event format
        content, json_tool_calls = self._parse_json_tool_calls(content)
        if json_tool_calls:
            # Re-number IDs to avoid conflicts
            for i, tc in enumerate(json_tool_calls):
                tc["id"] = f"call_{len(all_tool_calls) + i}"
            all_tool_calls.extend(json_tool_calls)

        return content, all_tool_calls if all_tool_calls else None

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

        # Parse tool calls from content if no native tool calls exist
        # Supports XML, bracket/arrow notation, and JSON/Event formats
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls is None and content:
            content, parsed_tool_calls = self._parse_all_tool_calls(content)
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
