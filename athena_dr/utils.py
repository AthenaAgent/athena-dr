import asyncio
from typing import Any

from fastmcp import Client
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from athena_dr.agent.mcp_backend.main import mcp


def _get_type_string(prop_schema: dict[str, Any]) -> str:
    """Extract a readable type string from a JSON schema property."""
    if "type" in prop_schema:
        return prop_schema["type"]
    elif "anyOf" in prop_schema:
        types = []
        for option in prop_schema["anyOf"]:
            if "type" in option:
                types.append(option["type"])
        return " | ".join(types)
    elif "allOf" in prop_schema:
        return "object"
    return "any"


def _create_parameters_table(input_schema: dict[str, Any]) -> Table:
    """Create a Rich table from the input schema properties."""
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Parameter", style="cyan", no_wrap=True)
    table.add_column("Type", style="yellow")
    table.add_column("Required", justify="center")
    table.add_column("Default", style="dim")
    table.add_column("Description", style="white")

    properties = input_schema.get("properties", {})
    required_params = set(input_schema.get("required", []))

    for param_name, param_schema in properties.items():
        # Type
        param_type = _get_type_string(param_schema)

        # Required
        is_required = param_name in required_params
        required_str = "[bold red]✓[/bold red]" if is_required else "[dim]—[/dim]"

        # Default value
        default_value = param_schema.get("default", None)
        if default_value is None:
            default_str = "[dim]—[/dim]"
        elif isinstance(default_value, bool):
            default_str = str(default_value).lower()
        elif isinstance(default_value, str):
            default_str = f'"{default_value}"' if default_value else '""'
        else:
            default_str = str(default_value)

        # Description
        description = param_schema.get("description", "")
        # Truncate long descriptions
        if len(description) > 60:
            description = description[:57] + "..."

        table.add_row(param_name, param_type, required_str, default_str, description)

    return table


def pretty_print_tools(tools: list, console: Console) -> None:
    """Print the list of tools in a formatted way using Rich."""
    if not tools:
        console.print("[yellow]No tools found on the server.[/yellow]")
        return

    # Header
    console.print()
    console.print(
        Panel(
            f"[bold green]Found {len(tools)} tool(s)[/bold green]",
            title="🔧 MCP Server Tools",
            border_style="blue",
        )
    )
    console.print()

    for tool in tools:
        # Tool name as a styled header
        tool_header = Text()
        tool_header.append("📦 ", style="bold")
        tool_header.append(tool.name, style="bold cyan")

        # Description panel
        description = tool.description or "No description available."
        first_line = description.split("\n")[0].strip()

        # Create panel for tool with description
        console.print(
            Panel(
                f"[white]{first_line}[/white]",
                title=tool_header,
                border_style="dim",
                expand=False,
            )
        )

        # Show input schema as a table
        if tool.inputSchema and tool.inputSchema.get("properties"):
            params_table = _create_parameters_table(tool.inputSchema)
            console.print(
                Panel(
                    params_table,
                    title="[dim]Parameters[/dim]",
                    border_style="dim",
                    expand=False,
                )
            )

        console.print()


async def _list_tools_from_server(server) -> None:
    """Connect to a FastMCP server object directly and list tools.
    
    Args:
        server: A FastMCP server instance
    """
    console = Console()
    async with Client(server) as client:
        tools = await client.list_tools()
        pretty_print_tools(tools, console)


def list_tools_from_server():
    asyncio.run(_list_tools_from_server(mcp))
