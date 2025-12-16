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


def _register_generate_sft_trace_command(app: typer.Typer) -> None:
    """Register the generate-sft-trace command to the given Typer app."""

    @app.command(name="generate-sft-trace")
    def generate_sft_trace(
        dataset_name: Annotated[
            str,
            typer.Argument(
                help="Name of the dataset to load (e.g., 'hotpotqa/hotpot_qa')."
            ),
        ],
        prompt_column: Annotated[
            str,
            typer.Option(
                "--prompt-column",
                "-p",
                help="Column name containing the prompts/questions.",
            ),
        ],
        gt_answer_column: Annotated[
            str,
            typer.Option(
                "--gt-answer-column",
                "-g",
                help="Column name containing ground truth answers.",
            ),
        ],
        auto_search_config: Annotated[
            str,
            typer.Option(
                "--auto-search-config",
                "-a",
                help="Path to auto search configuration file.",
            ),
        ] = "configs/auto_search_configs.yml",
        rejection_sampling_config: Annotated[
            str,
            typer.Option(
                "--rejection-sampling-config",
                "-r",
                help="Path to rejection sampling configuration file.",
            ),
        ] = "configs/rejection_sampling_configs.yml",
        dataset_subset: Annotated[
            Optional[str],
            typer.Option(
                "--dataset-subset", "-s", help="Dataset subset/configuration to load."
            ),
        ] = None,
        dataset_split: Annotated[
            str,
            typer.Option("--dataset-split", help="Dataset split to use."),
        ] = "train",
        max_examples: Annotated[
            Optional[int],
            typer.Option(
                "--max-examples", "-n", help="Maximum number of examples to process."
            ),
        ] = None,
        max_attempts_per_example: Annotated[
            int,
            typer.Option(
                "--max-attempts",
                "-m",
                help="Maximum attempts per example for rejection sampling.",
            ),
        ] = 3,
        export_dataset: Annotated[
            Optional[str],
            typer.Option(
                "--export-dataset",
                "-e",
                help="Hugging Face Hub dataset name to export traces to.",
            ),
        ] = None,
        project_name: Annotated[
            str,
            typer.Option("--project", help="Weave project name for tracing."),
        ] = "athena_dr",
    ) -> None:
        """
        Generate SFT traces from a dataset using the TraceGenerator.

        This command loads a dataset, generates traces using the AutoReasonSearchWorkflow,
        applies rejection sampling to filter out incorrect answers, and optionally exports
        the resulting traces to Hugging Face Hub.
        """
        load_dotenv()
        warnings.filterwarnings("ignore")
        weave.init(project_name=project_name)

        # Load the dataset
        if dataset_subset is not None:
            dataset = load_dataset(dataset_name, dataset_subset, split=dataset_split)
        else:
            dataset = load_dataset(dataset_name, split=dataset_split)

        # Extract the simple dataset name for identification
        simple_dataset_name = dataset_name.split("/")[-1]

        trace_generator = TraceGenerator(
            auto_search_config_path=auto_search_config,
            rejection_sampling_config_path=rejection_sampling_config,
            dataset=dataset,
            dataset_name=simple_dataset_name,
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

        rich.print(f"[green]Generated {len(traces)} SFT traces successfully![/green]")
        if export_dataset:
            rich.print(f"[blue]Traces exported to: {export_dataset}[/blue]")


def launch_auto_search_workflow_cli() -> None:
    """Launch the athena-dr CLI with all commands."""
    load_dotenv()
    # Get the base Typer app from AutoReasonSearchWorkflow
    app = AutoReasonSearchWorkflow.get_typer_app()
    # Register the generate-sft-trace command
    _register_generate_sft_trace_command(app)
    # Run the app
    app()
