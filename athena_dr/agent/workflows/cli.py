import asyncio
import warnings
from typing import Annotated, Optional

import rich
import typer
import weave
from datasets import load_dataset
from dotenv import load_dotenv

from athena_dr.agent.workflows import (
    AutoReasonSearchWorkflow,
    TraceGenerator,
    run_web_page_reader,
)

app = typer.Typer(help="Athena DR CLI for trace generation and workflows.")


@app.command("generate-sft-trace")
def generate_sft_trace(
    dataset_path: Annotated[
        str,
        typer.Argument(help="HuggingFace dataset path (e.g., 'hotpotqa/hotpot_qa')."),
    ],
    dataset_name: Annotated[
        str,
        typer.Argument(
            help="Name of the dataset configuration (e.g., 'distractor' for hotpotqa)."
        ),
    ],
    prompt_column: Annotated[
        str,
        typer.Option("--prompt-column", "-p", help="Column name for the prompt/question."),
    ] = "question",
    gt_answer_column: Annotated[
        str,
        typer.Option("--gt-answer-column", "-g", help="Column name for the ground truth answer."),
    ] = "answer",
    split: Annotated[
        str,
        typer.Option("--split", "-s", help="Dataset split to use."),
    ] = "train",
    max_examples: Annotated[
        Optional[int],
        typer.Option("--max-examples", "-n", help="Maximum number of examples to process."),
    ] = None,
    max_attempts_per_example: Annotated[
        int,
        typer.Option("--max-attempts", "-a", help="Maximum attempts per example for rejection sampling."),
    ] = 3,
    auto_search_config_path: Annotated[
        str,
        typer.Option("--auto-search-config", help="Path to auto search configuration file."),
    ] = "configs/auto_search_configs.yml",
    rejection_sampling_config_path: Annotated[
        str,
        typer.Option("--rejection-sampling-config", help="Path to rejection sampling configuration file."),
    ] = "configs/rejection_sampling_configs.yml",
    export_dataset: Annotated[
        Optional[str],
        typer.Option("--export-dataset", "-e", help="HuggingFace Hub path to export the generated traces."),
    ] = None,
    project_name: Annotated[
        str,
        typer.Option("--project", help="Weave project name for tracing."),
    ] = "athena_dr",
) -> None:
    """
    Generate SFT traces from a HuggingFace dataset using the TraceGenerator.

    This command loads a dataset, generates traces using AutoReasonSearchWorkflow,
    and optionally exports the traces to HuggingFace Hub.
    """
    load_dotenv()
    warnings.filterwarnings("ignore")
    weave.init(project_name=project_name)

    dataset = load_dataset(dataset_path, dataset_name, split=split)

    trace_generator = TraceGenerator(
        auto_search_config_path=auto_search_config_path,
        rejection_sampling_config_path=rejection_sampling_config_path,
        dataset=dataset,
        dataset_name=dataset_name,
        max_examples=max_examples,
    )

    traces = asyncio.run(
        trace_generator.generate_trace(
            prompt_column=prompt_column,
            gt_answer_column=gt_answer_column,
            max_attempts_per_example=max_attempts_per_example,
            export_dataset=export_dataset,
        )
    )

    rich.print(f"[green]Generated {len(traces)} SFT traces.[/green]")
    if export_dataset:
        rich.print(f"[green]Traces exported to: {export_dataset}[/green]")


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


def launch_web_page_reader_cli() -> None:
    typer.run(web_page_reader)


def launch_auto_search_workflow_cli() -> None:
    load_dotenv()
    AutoReasonSearchWorkflow.app()


def launch_trace_generator_cli() -> None:
    load_dotenv()
    app()
