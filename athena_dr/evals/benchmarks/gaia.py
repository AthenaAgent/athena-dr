"""
GAIA Benchmark implementation for athena-dr.

GAIA (General AI Assistants) is a benchmark that evaluates AI assistants on
complex tasks requiring browsing, reasoning, and tool use.

Reference: https://huggingface.co/datasets/gaia-benchmark/GAIA
"""

import asyncio
import os
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import datasets
from huggingface_hub import login, snapshot_download

from athena_dr.evals.base import BaseBenchmark, BenchmarkConfig, BenchmarkResult


@dataclass
class GAIABenchmarkConfig(BenchmarkConfig):
    """Configuration for GAIA benchmark."""
    
    # Dataset configuration
    split: str = "validation"  # "validation" or "test"
    use_raw_dataset: bool = False  # Use raw GAIA or annotated version
    data_dir: str = "data/gaia"
    
    # File handling
    include_file_questions: bool = True  # Only include questions with files
    
    # Workflow configuration
    workflow_config_path: Optional[str] = None
    max_tool_calls: int = 10
    
    # OpenRouter/Custom endpoint configuration
    openrouter_provider_order: Optional[List[str]] = None
    openrouter_allow_fallbacks: bool = True


class GAIABenchmark(BaseBenchmark):
    """
    GAIA Benchmark evaluation using athena-dr agent.
    
    GAIA tests AI assistants on real-world tasks that require:
    - Web browsing and navigation
    - File analysis (PDFs, images, audio, etc.)
    - Multi-step reasoning
    - Tool use
    
    Example usage:
        config = GAIABenchmarkConfig(
            run_name="gaia-test-run",
            model_id="openrouter/anthropic/claude-3.5-sonnet",
            concurrency=8,
        )
        benchmark = GAIABenchmark(config)
        results = await benchmark.run()
    """
    
    def __init__(
        self,
        config: GAIABenchmarkConfig,
        hf_token: Optional[str] = None,
    ):
        super().__init__(config)
        self.config: GAIABenchmarkConfig = config
        
        # Login to HuggingFace if token provided
        if hf_token:
            login(hf_token)
        elif os.getenv("HF_TOKEN"):
            login(os.getenv("HF_TOKEN"))
        
        # Initialize workflow
        self._setup_workflow()
    
    def _setup_workflow(self) -> None:
        """Initialize the athena-dr workflow."""
        from athena_dr.agent.workflows import AutoReasonSearchWorkflow
        
        # Create workflow configuration
        if self.config.workflow_config_path:
            self.workflow = AutoReasonSearchWorkflow(
                configuration=self.config.workflow_config_path
            )
        else:
            # Create dynamic configuration for custom endpoint
            workflow_config = {
                "tool_parser": "v20250824",
                "search_agent_model_name": self.config.model_id,
                "search_agent_api_key": self.config.api_key or os.getenv("OPENROUTER_API_KEY", ""),
                "search_agent_max_tokens": self.config.max_tokens,
                "search_agent_temperature": self.config.temperature,
                "search_agent_max_tool_calls": self.config.max_tool_calls,
                "number_documents_to_search": 10,
                "search_timeout": 60,
                "browse_tool_name": "jina",
                "browse_timeout": 60,
                "browse_max_pages_to_fetch": 10,
                "browse_context_char_length": 6000,
                "use_browse_agent": False,
                "mcp_transport_type": "FastMCPTransport",
            }
            
            # Add base_url if using custom endpoint
            if self.config.base_url:
                workflow_config["search_agent_base_url"] = self.config.base_url
            
            # Add OpenRouter provider preferences
            if self.config.openrouter_provider_order:
                workflow_config["openrouter_provider_order"] = self.config.openrouter_provider_order
                workflow_config["openrouter_allow_fallbacks"] = self.config.openrouter_allow_fallbacks
            
            self.workflow = AutoReasonSearchWorkflow(configuration=workflow_config)
    
    def load_dataset(self) -> datasets.Dataset:
        """
        Load the GAIA dataset from JSONL metadata.
        
        Downloads from HuggingFace Hub if not already cached.
        Returns the specified split (validation or test).
        """
        data_path = Path(self.config.data_dir)
        
        if not data_path.exists():
            print(f"Downloading GAIA dataset to {data_path}...")
            
            if self.config.use_raw_dataset:
                # Download raw GAIA dataset
                snapshot_download(
                    repo_id="gaia-benchmark/GAIA",
                    repo_type="dataset",
                    local_dir=str(data_path),
                    ignore_patterns=[".gitattributes", "README.md"],
                )
            else:
                # Download annotated version (requires access approval)
                snapshot_download(
                    repo_id="smolagents/GAIA-annotated",
                    repo_type="dataset",
                    local_dir=str(data_path),
                    ignore_patterns=[".gitattributes", "README.md"],
                )
        
        # Load from JSONL metadata directly (datasets lib deprecated .py scripts)
        metadata_file = data_path / "2023" / self.config.split / "metadata.jsonl"
        if not metadata_file.exists():
            # Try alternate path structure
            metadata_file = data_path / self.config.split / "metadata.jsonl"
        
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
                file_path = data_path / "2023" / self.config.split / file_name
                if not file_path.exists():
                    # Fall back to split/ path
                    file_path = data_path / self.config.split / file_name
                row["file_name"] = str(file_path)
            return row
        
        eval_ds = eval_ds.map(preprocess_file_paths)
        
        # Filter to only include questions with files if configured
        if self.config.include_file_questions:
            eval_ds = eval_ds.filter(lambda x: len(x.get("file_name", "")) > 0)
        
        return eval_ds
    
    def format_question(self, example: Dict[str, Any]) -> str:
        """
        Format the GAIA question with file context if available.
        
        Args:
            example: Dataset example with question and optional file_name
            
        Returns:
            Augmented question string
        """
        augmented_question = """You have one question to answer. It is paramount that you provide a correct answer.
Give it all you can: I know for a fact that you have access to all the relevant tools to solve it and find the correct answer (the answer does exist).
Failure or 'I cannot answer' or 'None found' will not be tolerated, success will be rewarded.
Run verification steps if that's needed, you must make sure you find the correct answer! Here is the task:

""" + example["question"]
        
        # Add file context if available
        if example.get("file_name") and len(example["file_name"]) > 0:
            file_path = Path(example["file_name"])
            
            if file_path.suffix == ".zip":
                prompt_use_files = "\n\nTo solve the task above, you will have to use these attached files:\n"
                prompt_use_files += self._get_zip_description(file_path, example["question"])
            else:
                prompt_use_files = "\n\nTo solve the task above, you will have to use this attached file:\n"
                prompt_use_files += self._get_file_description(file_path, example["question"])
            
            augmented_question += prompt_use_files
        
        return augmented_question
    
    def _get_file_description(self, file_path: Path, question: str) -> str:
        """Get description of a single file."""
        if not file_path.exists():
            return f"File: {file_path.name} (not found)"
        
        suffix = file_path.suffix.lower()
        size_mb = file_path.stat().st_size / (1024 * 1024)
        
        desc = f"File: {file_path.name} ({size_mb:.2f} MB)\n"
        desc += f"Type: {suffix}\n"
        
        if suffix in [".txt", ".md", ".csv", ".json"]:
            # Read text content preview
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read(5000)  # First 5000 chars
                desc += f"Preview:\n{content}\n"
            except Exception as e:
                desc += f"Error reading file: {e}\n"
        elif suffix in [".pdf"]:
            desc += "PDF document - use appropriate tools to read\n"
        elif suffix in [".png", ".jpg", ".jpeg", ".gif"]:
            desc += "Image file - use visual inspection tools\n"
        elif suffix in [".mp3", ".wav", ".ogg"]:
            desc += "Audio file - use audio transcription tools\n"
        elif suffix in [".xlsx", ".xls"]:
            desc += "Excel spreadsheet - use appropriate tools to read\n"
        
        return desc
    
    def _get_zip_description(self, zip_path: Path, question: str) -> str:
        """Get description of a zip file's contents."""
        if not zip_path.exists():
            return f"Archive: {zip_path.name} (not found)"
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                files = zf.namelist()
                desc = f"Archive: {zip_path.name}\n"
                desc += f"Contains {len(files)} files:\n"
                for f in files[:20]:  # First 20 files
                    desc += f"  - {f}\n"
                if len(files) > 20:
                    desc += f"  ... and {len(files) - 20} more files\n"
                return desc
        except Exception as e:
            return f"Archive: {zip_path.name} (error reading: {e})"
    
    async def evaluate_single(
        self,
        example: Dict[str, Any],
    ) -> BenchmarkResult:
        """
        Evaluate a single GAIA example.
        
        Args:
            example: GAIA dataset example
            
        Returns:
            BenchmarkResult with evaluation outcome
        """
        augmented_question = self.format_question(example)
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        result = BenchmarkResult(
            task_id=example.get("task_id", ""),
            question=example["question"],
            augmented_question=augmented_question,
            true_answer=example.get("true_answer", ""),
            task=example.get("task", ""),
            start_time=start_time,
        )
        
        try:
            # Run the workflow
            trace = await self.workflow(
                problem=augmented_question,
                dataset_name="gaia",
                verbose=True,
            )
            
            result.prediction = trace.get("final_response", "")
            result.browsed_links = trace.get("browsed_links", [])
            result.searched_links = trace.get("searched_links", [])
            result.total_tool_calls = trace.get("total_tool_calls", 0)
            result.failed_tool_calls = trace.get("total_failed_tool_calls", 0)
            
            # Extract intermediate steps
            if "full_traces" in trace:
                full_traces = trace["full_traces"]
                if hasattr(full_traces, "generated_text"):
                    result.intermediate_steps = [full_traces.generated_text]
                if hasattr(full_traces, "total_tokens"):
                    result.input_tokens = full_traces.total_tokens // 2
                    result.output_tokens = full_traces.total_tokens // 2
            
        except Exception as e:
            print(f"Error evaluating: {example.get('task_id', 'unknown')}: {e}")
            result.agent_error = str(e)
        
        result.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Append result to file
        self.append_result(result)
        
        return result
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute GAIA-specific metrics."""
        import pandas as pd
        
        try:
            df = pd.read_json(self.output_file, lines=True)
        except Exception as e:
            print(f"Error loading results: {e}")
            return {}
        
        total = len(df)
        if total == 0:
            return {"total": 0}
        
        metrics = {
            "total": total,
            "errors": df["agent_error"].notna().sum(),
            "error_rate": df["agent_error"].notna().mean(),
            "avg_tool_calls": df["total_tool_calls"].mean(),
            "avg_failed_tool_calls": df["failed_tool_calls"].mean(),
        }
        
        # Metrics by task level
        if "task" in df.columns:
            for level in df["task"].unique():
                level_df = df[df["task"] == level]
                metrics[f"level_{level}_count"] = len(level_df)
                metrics[f"level_{level}_error_rate"] = level_df["agent_error"].notna().mean()
        
        return metrics


async def run_gaia_benchmark(
    run_name: str,
    model_id: str = "openrouter/anthropic/claude-3.5-sonnet",
    concurrency: int = 8,
    split: str = "validation",
    max_examples: Optional[int] = None,
    workflow_config_path: Optional[str] = None,
    **kwargs,
) -> List[BenchmarkResult]:
    """
    Convenience function to run GAIA benchmark.
    
    Args:
        run_name: Name for this evaluation run
        model_id: Model to use (supports litellm format, e.g., openrouter/model-name)
        concurrency: Number of parallel evaluations
        split: Dataset split ("validation" or "test")
        max_examples: Maximum examples to evaluate
        workflow_config_path: Path to workflow config YAML
        **kwargs: Additional config options
        
    Returns:
        List of BenchmarkResult objects
        
    Example:
        results = await run_gaia_benchmark(
            run_name="my-gaia-run",
            model_id="openrouter/anthropic/claude-3.5-sonnet",
            concurrency=8,
        )
    """
    config = GAIABenchmarkConfig(
        run_name=run_name,
        model_id=model_id,
        concurrency=concurrency,
        split=split,
        workflow_config_path=workflow_config_path,
        **kwargs,
    )
    
    benchmark = GAIABenchmark(config)
    return await benchmark.run(max_examples=max_examples)
