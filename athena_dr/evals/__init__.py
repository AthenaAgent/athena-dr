"""
Athena-DR Evals Module

This module provides evaluation benchmarks for testing the athena-dr agent system.
Currently supported benchmarks:
- GAIA: General AI Assistants benchmark
"""

from athena_dr.evals.benchmarks.gaia import GAIABenchmark, GAIABenchmarkConfig
from athena_dr.evals.base import BaseBenchmark, BenchmarkConfig, BenchmarkResult

__all__ = [
    # Base classes
    "BaseBenchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    # GAIA benchmark
    "GAIABenchmark",
    "GAIABenchmarkConfig",
]
