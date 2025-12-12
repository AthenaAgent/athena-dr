from typing import Annotated, Optional

import rich
import typer
import weave
from dotenv import load_dotenv

from athena_dr.agent.workflows import run_web_page_reader


def web_page_reader(
    urls: Annotated[list[str], typer.Argument(help="URLs of web pages to read.")],
    question: Annotated[
        str,
        typer.Option(
            "--question", "-q", help="The question to answer from the web pages."
        ),
    ],
    project_name: Annotated[
        Optional[str],
        typer.Option("--project", "-p", help="Weave project name for tracing."),
    ] = "athena_dr",
) -> None:
    """
    Read web pages and answer a question based on their content.
    """
    load_dotenv()
    if project_name is not None:
        weave.init(project_name=project_name)

    result = run_web_page_reader(question=question, urls=urls)
    rich.print(result)


def main() -> None:
    typer.run(web_page_reader)


if __name__ == "__main__":
    main()
