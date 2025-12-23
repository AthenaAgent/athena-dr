"""
CLI for running athena-dr evaluations.
"""

import argparse
import asyncio
import os
from typing import Optional

from dotenv import load_dotenv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run athena-dr benchmark evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run GAIA benchmark with custom endpoint
    athena-dr-evals gaia --run-name my-test --model-id openrouter/anthropic/claude-3.5-sonnet
    
    # Run with workflow config file
    athena-dr-evals gaia --run-name my-test --config configs/gaia/custom.yml
    
    # Run with specific split and limited examples
    athena-dr-evals gaia --run-name test --split validation --max-examples 10
        """,
    )
    
    subparsers = parser.add_subparsers(dest="benchmark", help="Benchmark to run")
    
    # GAIA benchmark subparser
    gaia_parser = subparsers.add_parser("gaia", help="Run GAIA benchmark")
    gaia_parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Name for this evaluation run",
    )
    gaia_parser.add_argument(
        "--model-id",
        type=str,
        default="openrouter/anthropic/claude-3.5-sonnet",
        help="Model ID in litellm format (e.g., openrouter/model-name)",
    )
    gaia_parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of parallel evaluations",
    )
    gaia_parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Dataset split to use",
    )
    gaia_parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate",
    )
    gaia_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to workflow configuration YAML file",
    )
    gaia_parser.add_argument(
        "--output-dir",
        type=str,
        default="output/gaia",
        help="Directory for output files",
    )
    gaia_parser.add_argument(
        "--use-raw-dataset",
        action="store_true",
        help="Use raw GAIA dataset instead of annotated version",
    )
    gaia_parser.add_argument(
        "--max-tokens",
        type=int,
        default=32000,
        help="Maximum tokens for generation",
    )
    gaia_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation",
    )
    gaia_parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=10,
        help="Maximum tool calls per question",
    )
    
    # Metrics subparser
    metrics_parser = subparsers.add_parser("metrics", help="Compute metrics from results")
    metrics_parser.add_argument(
        "--results-file",
        type=str,
        required=True,
        help="Path to results JSONL file",
    )
    
    return parser.parse_args()


async def run_gaia(args) -> None:
    """Run GAIA benchmark."""
    from athena_dr.evals.benchmarks.gaia import GAIABenchmark, GAIABenchmarkConfig
    
    config = GAIABenchmarkConfig(
        run_name=args.run_name,
        model_id=args.model_id,
        concurrency=args.concurrency,
        split=args.split,
        output_dir=args.output_dir,
        workflow_config_path=args.config,
        use_raw_dataset=args.use_raw_dataset,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_tool_calls=args.max_tool_calls,
    )
    
    benchmark = GAIABenchmark(config)
    
    print(f"Starting GAIA benchmark: {args.run_name}")
    print(f"Model: {args.model_id}")
    print(f"Split: {args.split}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Output: {benchmark.output_file}")
    print("-" * 50)
    
    results = await benchmark.run(max_examples=args.max_examples)
    
    print("-" * 50)
    print(f"Completed {len(results)} evaluations")
    
    # Compute and display metrics
    metrics = benchmark.compute_metrics()
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def compute_metrics(args) -> None:
    """Compute metrics from results file."""
    import json
    import pandas as pd
    
    try:
        df = pd.read_json(args.results_file, lines=True)
    except Exception as e:
        print(f"Error loading results: {e}")
        return
    
    total = len(df)
    if total == 0:
        print("No results found")
        return
    
    print(f"\nResults from: {args.results_file}")
    print(f"Total examples: {total}")
    print(f"Errors: {df['agent_error'].notna().sum()}")
    print(f"Error rate: {df['agent_error'].notna().mean():.2%}")
    
    if "total_tool_calls" in df.columns:
        print(f"Avg tool calls: {df['total_tool_calls'].mean():.2f}")
    
    if "task" in df.columns:
        print("\nBy task level:")
        for level in sorted(df["task"].unique()):
            level_df = df[df["task"] == level]
            print(f"  Level {level}: {len(level_df)} examples, "
                  f"{level_df['agent_error'].notna().mean():.2%} error rate")


def main():
    """Main entry point."""
    load_dotenv(override=True)
    
    args = parse_args()
    
    if args.benchmark == "gaia":
        asyncio.run(run_gaia(args))
    elif args.benchmark == "metrics":
        compute_metrics(args)
    else:
        print("Please specify a benchmark. Use --help for usage.")


if __name__ == "__main__":
    main()
