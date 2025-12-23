#!/usr/bin/env python
"""
Run GAIA Benchmark with athena-dr

This script runs the GAIA benchmark using the athena-dr agent system.
It supports custom endpoints and OpenRouter for model inference.

Example usage:
    # Run with OpenRouter (default)
    python run_gaia.py --run-name my-test-run --model-id openrouter/anthropic/claude-3.5-sonnet

    # Run with custom endpoint
    python run_gaia.py --run-name my-test-run --model-id custom/my-model --base-url https://my-endpoint.com/v1

    # Run with workflow config file
    python run_gaia.py --run-name my-test-run --config configs/gaia/custom_endpoint.yml
    
    # Run with high concurrency
    python run_gaia.py --run-name my-test-run --concurrency 32 --model-id openrouter/deepseek/deepseek-chat
"""

import argparse
import asyncio
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from athena_dr.agent.workflows import AutoReasonSearchWorkflow

load_dotenv(override=True)

# Login to HuggingFace if token available
if os.getenv("HF_TOKEN"):
    login(os.getenv("HF_TOKEN"))

append_answer_lock = threading.Lock()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GAIA benchmark with athena-dr",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--model-id", type=str, default="openrouter/anthropic/claude-3.5-sonnet")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "test"])
    parser.add_argument("--use-raw-dataset", action="store_true")
    parser.add_argument("--config", type=str, default=None, help="Path to workflow config YAML")
    parser.add_argument("--base-url", type=str, default=None, help="Custom endpoint base URL")
    parser.add_argument("--api-key", type=str, default=None, help="API key (uses env var if not set)")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tool-calls", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="output/gaia")
    parser.add_argument("--data-dir", type=str, default="data/gaia")
    parser.add_argument("--include-all-questions", action="store_true", 
                       help="Include questions without files (default: only file questions)")
    return parser.parse_args()


print("Make sure you deactivated any VPN, else some URLs may be blocked!")


def create_workflow(args) -> AutoReasonSearchWorkflow:
    """Create the athena-dr workflow with the given configuration."""
    if args.config:
        return AutoReasonSearchWorkflow(configuration=args.config)
    
    # Build configuration dynamically
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY", "")
    
    config = {
        "tool_parser": "v20250824",
        "search_agent_model_name": args.model_id,
        "search_agent_api_key": api_key,
        "search_agent_max_tokens": args.max_tokens,
        "search_agent_temperature": args.temperature,
        "search_agent_max_tool_calls": args.max_tool_calls,
        "number_documents_to_search": 10,
        "search_timeout": 60,
        "browse_tool_name": "jina",
        "browse_timeout": 60,
        "browse_max_pages_to_fetch": 10,
        "browse_context_char_length": 6000,
        "use_browse_agent": False,
        "prompt_version": "v20250907",
        # Use StreamableHttpTransport to connect to external MCP server
        # Can be overridden via MCP_TRANSPORT and MCP_TRANSPORT_PORT env vars
        "mcp_transport_type": os.getenv("MCP_TRANSPORT", "StreamableHttpTransport"),
        "mcp_port": int(os.getenv("MCP_TRANSPORT_PORT", "8000")),
    }
    
    if args.base_url:
        config["search_agent_base_url"] = args.base_url
    
    # Add OpenRouter provider order if using OpenRouter
    if "openrouter/" in args.model_id.lower():
        config["openrouter_provider_order"] = ["Anthropic", "DeepSeek", "Together"]
        config["openrouter_allow_fallbacks"] = True
    
    return AutoReasonSearchWorkflow(configuration=config)


def load_gaia_dataset(args) -> datasets.Dataset:
    """Load the GAIA dataset from JSONL metadata (datasets lib no longer supports .py scripts)."""
    data_path = Path(args.data_dir)
    
    if not data_path.exists():
        print(f"Downloading GAIA dataset to {data_path}...")
        if args.use_raw_dataset:
            snapshot_download(
                repo_id="gaia-benchmark/GAIA",
                repo_type="dataset",
                local_dir=str(data_path),
                ignore_patterns=[".gitattributes", "README.md"],
            )
        else:
            snapshot_download(
                repo_id="smolagents/GAIA-annotated",
                repo_type="dataset",
                local_dir=str(data_path),
                ignore_patterns=[".gitattributes", "README.md"],
            )
    
    # Load from JSONL metadata directly (datasets lib deprecated .py scripts)
    metadata_file = data_path / "2023" / args.split / "metadata.jsonl"
    if not metadata_file.exists():
        # Try alternate path structure
        metadata_file = data_path / args.split / "metadata.jsonl"
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"Could not find metadata.jsonl in {data_path}")
    
    print(f"Loading dataset from: {metadata_file}")
    eval_ds = datasets.load_dataset("json", data_files=str(metadata_file), split="train")
    
    # Rename columns to standard format (handle both old and new column names)
    rename_map = {}
    if "Question" in eval_ds.column_names:
        rename_map["Question"] = "question"
    if "Final answer" in eval_ds.column_names:
        rename_map["Final answer"] = "true_answer"
    if "Level" in eval_ds.column_names:
        rename_map["Level"] = "task"
    
    if rename_map:
        eval_ds = eval_ds.rename_columns(rename_map)
    
    # Preprocess file paths - look in 2023/split/ or split/ directory
    def preprocess_file_paths(row):
        file_name = row.get("file_name", "")
        if file_name and len(file_name) > 0:
            # Try 2023/split/ path first
            file_path = data_path / "2023" / args.split / file_name
            if not file_path.exists():
                # Fall back to split/ path
                file_path = data_path / args.split / file_name
            row["file_name"] = str(file_path)
        return row
    
    eval_ds = eval_ds.map(preprocess_file_paths)
    return eval_ds


def append_answer(entry: dict, jsonl_file: str) -> None:
    """Thread-safe append of result to JSONL file."""
    jsonl_path = Path(jsonl_file)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with append_answer_lock, open(jsonl_file, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry) + "\n")
    print(f"Result saved to: {jsonl_path.resolve()}")


def format_question(example: Dict[str, Any], data_dir: str) -> str:
    """Format the GAIA question with file context if available."""
    augmented_question = """You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist).
Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
Run verification steps if that's needed, you must make sure you find the correct answer! Here is the task:

""" + example["question"]
    
    if example.get("file_name") and len(example["file_name"]) > 0:
        file_path = Path(example["file_name"])
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            augmented_question += f"\n\nTo solve the task above, you will have to use this attached file:\n"
            augmented_question += f"File: {file_path.name} ({size_mb:.2f} MB, type: {file_path.suffix})\n"
            
            if file_path.suffix.lower() == ".zip":
                import zipfile
                try:
                    with zipfile.ZipFile(file_path, 'r') as zf:
                        files = zf.namelist()
                        augmented_question += f"Archive contents ({len(files)} files):\n"
                        for f in files[:10]:
                            augmented_question += f"  - {f}\n"
                        if len(files) > 10:
                            augmented_question += f"  ... and {len(files) - 10} more\n"
                except Exception as e:
                    augmented_question += f"(Error reading archive: {e})\n"
            elif file_path.suffix.lower() in [".txt", ".md", ".csv", ".json"]:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read(3000)
                    augmented_question += f"Preview:\n{content}\n"
                except Exception as e:
                    augmented_question += f"(Error reading file: {e})\n"
    
    return augmented_question


def answer_single_question(
    example: Dict[str, Any],
    workflow: AutoReasonSearchWorkflow,
    answers_file: str,
    args,
) -> None:
    """Evaluate a single GAIA question."""
    augmented_question = format_question(example, args.data_dir)
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        # Run the agent workflow
        trace = asyncio.run(workflow(
            problem=augmented_question,
            dataset_name="gaia",
            verbose=True,
        ))
        
        output = trace.get("final_response", "")
        intermediate_steps = []
        if "full_traces" in trace:
            full_traces = trace["full_traces"]
            if hasattr(full_traces, "generated_text"):
                intermediate_steps = [full_traces.generated_text]
        
        parsing_error = False
        iteration_limit_exceeded = "Agent stopped due to iteration limit" in str(output)
        raised_exception = False
        exception = None
        
    except Exception as e:
        print(f"Error on {example.get('task_id', 'unknown')}: {e}")
        output = None
        intermediate_steps = []
        parsing_error = False
        iteration_limit_exceeded = False
        exception = e
        raised_exception = True
    
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    annotated_example = {
        "agent_name": args.model_id,
        "question": example["question"],
        "augmented_question": augmented_question,
        "prediction": output,
        "intermediate_steps": intermediate_steps,
        "parsing_error": parsing_error,
        "iteration_limit_exceeded": iteration_limit_exceeded,
        "agent_error": str(exception) if raised_exception else None,
        "task": example.get("task", ""),
        "task_id": example.get("task_id", ""),
        "true_answer": example.get("true_answer", ""),
        "start_time": start_time,
        "end_time": end_time,
        "browsed_links": trace.get("browsed_links", []) if not raised_exception else [],
        "searched_links": trace.get("searched_links", []) if not raised_exception else [],
        "total_tool_calls": trace.get("total_tool_calls", 0) if not raised_exception else 0,
        "failed_tool_calls": trace.get("total_failed_tool_calls", 0) if not raised_exception else 0,
    }
    
    append_answer(annotated_example, answers_file)


def get_examples_to_answer(answers_file: str, eval_ds: datasets.Dataset, include_all: bool = False) -> List[Dict]:
    """Get examples that haven't been evaluated yet."""
    answers_path = Path(answers_file)
    print(f"Loading answers from {answers_file}...")
    done_questions = []
    
    if answers_path.exists() and answers_path.stat().st_size > 0:
        try:
            done_df = pd.read_json(answers_file, lines=True)
            if not done_df.empty and "question" in done_df.columns:
                done_questions = done_df["question"].tolist()
            print(f"Found {len(done_questions)} previous results!")
        except Exception as e:
            print(f"Error reading previous results ({e}), starting fresh.")
    else:
        print("No previous results found or file is empty, starting fresh.")
    
    examples = []
    for line in eval_ds.to_list():
        if line["question"] in done_questions:
            continue
        if not include_all and not line.get("file_name"):
            continue  # Skip questions without files by default
        examples.append(line)
    
    return examples


def main():
    args = parse_args()
    print(f"Starting GAIA benchmark run: {args.run_name}")
    print(f"Model: {args.model_id}")
    print(f"Split: {args.split}")
    print(f"Concurrency: {args.concurrency}")
    
    # Load dataset
    eval_ds = load_gaia_dataset(args)
    print("Loaded evaluation dataset:")
    print(pd.DataFrame(eval_ds)["task"].value_counts())
    
    # Create workflow
    workflow = create_workflow(args)
    
    # Get output file path
    answers_file = os.path.join(args.output_dir, args.split, f"{args.run_name}.jsonl")
    print(f"Output file: {answers_file}")
    
    # Get pending examples
    tasks_to_run = get_examples_to_answer(
        answers_file, 
        eval_ds, 
        include_all=args.include_all_questions
    )
    
    if args.max_examples:
        tasks_to_run = tasks_to_run[:args.max_examples]
    
    print(f"Running {len(tasks_to_run)} examples...")
    
    # Run evaluations with thread pool
    with ThreadPoolExecutor(max_workers=args.concurrency) as exe:
        futures = [
            exe.submit(answer_single_question, example, workflow, answers_file, args)
            for example in tasks_to_run
        ]
        for f in tqdm(as_completed(futures), total=len(tasks_to_run), desc="Evaluating"):
            try:
                f.result()
            except Exception as e:
                print(f"Task error: {e}")
    
    print("All tasks completed!")
    
    # Print summary
    try:
        df = pd.read_json(answers_file, lines=True)
        print(f"\nSummary:")
        print(f"  Total: {len(df)}")
        print(f"  Errors: {df['agent_error'].notna().sum()}")
        print(f"  Avg tool calls: {df['total_tool_calls'].mean():.2f}")
    except Exception as e:
        print(f"Could not load summary: {e}")


if __name__ == "__main__":
    main()
