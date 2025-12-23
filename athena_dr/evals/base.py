"""
Base classes for benchmark evaluations.
"""

import asyncio
import json
import os
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Union

import datasets
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

T = TypeVar("T")


@dataclass
class BenchmarkConfig:
    """Base configuration for benchmarks."""
    
    # Run configuration
    run_name: str
    concurrency: int = 8
    output_dir: str = "output"
    
    # Model configuration - custom endpoint support
    model_id: str = "openrouter/anthropic/claude-3.5-sonnet"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    
    # Generation parameters
    max_tokens: int = 4096
    temperature: float = 0.0
    
    # Workflow configuration
    workflow_config_path: Optional[str] = None


class BenchmarkResult(BaseModel):
    """Result from a single benchmark example."""
    
    task_id: str
    question: str
    augmented_question: Optional[str] = None
    prediction: Optional[str] = None
    true_answer: Optional[str] = None
    task: Union[str, int] = ""
    
    # Execution metadata
    start_time: str = ""
    end_time: str = ""
    
    # Status flags
    parsing_error: bool = False
    iteration_limit_exceeded: bool = False
    agent_error: Optional[str] = None
    
    # Token counts
    input_tokens: int = 0
    output_tokens: int = 0
    
    # Additional metadata
    intermediate_steps: List[Any] = field(default_factory=list)
    browsed_links: List[str] = field(default_factory=list)
    searched_links: List[str] = field(default_factory=list)
    total_tool_calls: int = 0
    failed_tool_calls: int = 0


@dataclass
class BenchmarkResult:
    """Result from a single benchmark example."""
    
    task_id: str
    question: str
    augmented_question: Optional[str] = None
    prediction: Optional[str] = None
    true_answer: Optional[str] = None
    task: Union[str, int] = ""
    
    # Execution metadata
    start_time: str = ""
    end_time: str = ""
    
    # Status flags
    parsing_error: bool = False
    iteration_limit_exceeded: bool = False
    agent_error: Optional[str] = None
    
    # Token counts
    input_tokens: int = 0
    output_tokens: int = 0
    
    # Additional metadata
    intermediate_steps: List[Any] = field(default_factory=list)
    browsed_links: List[str] = field(default_factory=list)
    searched_links: List[str] = field(default_factory=list)
    total_tool_calls: int = 0
    failed_tool_calls: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "task_id": self.task_id,
            "question": self.question,
            "augmented_question": self.augmented_question,
            "prediction": self.prediction,
            "true_answer": self.true_answer,
            "task": self.task,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "parsing_error": self.parsing_error,
            "iteration_limit_exceeded": self.iteration_limit_exceeded,
            "agent_error": self.agent_error,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "intermediate_steps": self.intermediate_steps,
            "browsed_links": self.browsed_links,
            "searched_links": self.searched_links,
            "total_tool_calls": self.total_tool_calls,
            "failed_tool_calls": self.failed_tool_calls,
        }


class BaseBenchmark(ABC):
    """Base class for all benchmarks."""
    
    # Thread-safe file writing
    _append_lock = threading.Lock()
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._setup_output_dir()
    
    def _setup_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def output_file(self) -> str:
        """Get the output file path for results."""
        return os.path.join(
            self.config.output_dir,
            f"{self.config.run_name}.jsonl"
        )
    
    def append_result(self, result: BenchmarkResult) -> None:
        """Thread-safe append of result to JSONL file."""
        output_path = Path(self.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self._append_lock, open(self.output_file, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(result.to_dict()) + "\n")
        
        print(f"Result exported to: {output_path.resolve()}")
    
    def get_completed_questions(self) -> List[str]:
        """Get list of already completed questions from output file."""
        try:
            df = pd.read_json(self.output_file, lines=True)
            return df["question"].tolist()
        except Exception as e:
            print(f"No previous results found: {e}")
            return []
    
    @abstractmethod
    def load_dataset(self) -> datasets.Dataset:
        """Load the benchmark dataset."""
        pass
    
    @abstractmethod
    async def evaluate_single(
        self,
        example: Dict[str, Any],
    ) -> BenchmarkResult:
        """Evaluate a single example from the benchmark."""
        pass
    
    @abstractmethod
    def format_question(self, example: Dict[str, Any]) -> str:
        """Format the question with any augmentation."""
        pass
    
    def get_pending_examples(self) -> List[Dict[str, Any]]:
        """Get examples that haven't been evaluated yet."""
        dataset = self.load_dataset()
        completed = set(self.get_completed_questions())
        
        pending = []
        for example in dataset:
            if example["question"] not in completed:
                pending.append(example)
        
        print(f"Found {len(completed)} completed, {len(pending)} pending examples")
        return pending
    
    async def run(
        self,
        max_examples: Optional[int] = None,
        skip_completed: bool = True,
    ) -> List[BenchmarkResult]:
        """
        Run the benchmark evaluation.
        
        Args:
            max_examples: Maximum number of examples to evaluate
            skip_completed: Whether to skip already completed examples
            
        Returns:
            List of benchmark results
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        if skip_completed:
            examples = self.get_pending_examples()
        else:
            examples = self.load_dataset().to_list()
        
        if max_examples:
            examples = examples[:max_examples]
        
        print(f"Running benchmark with {len(examples)} examples")
        print(f"Concurrency: {self.config.concurrency}")
        print(f"Output file: {self.output_file}")
        
        results = []
        
        # Run evaluations with thread pool
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as executor:
            futures = [
                executor.submit(
                    lambda ex: asyncio.run(self.evaluate_single(ex)),
                    example
                )
                for example in examples
            ]
            
            for future in tqdm(
                as_completed(futures),
                total=len(examples),
                desc="Evaluating"
            ):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error in evaluation: {e}")
        
        return results
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute aggregate metrics from results file."""
        try:
            df = pd.read_json(self.output_file, lines=True)
        except Exception as e:
            print(f"Error loading results: {e}")
            return {}
        
        total = len(df)
        if total == 0:
            return {"total": 0}
        
        return {
            "total": total,
            "errors": df["agent_error"].notna().sum(),
            "error_rate": df["agent_error"].notna().mean(),
            "parsing_errors": df["parsing_error"].sum(),
            "avg_tool_calls": df["total_tool_calls"].mean(),
            "avg_input_tokens": df["input_tokens"].mean(),
            "avg_output_tokens": df["output_tokens"].mean(),
        }
