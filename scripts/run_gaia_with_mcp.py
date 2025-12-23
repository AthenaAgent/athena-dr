#!/usr/bin/env python
"""
Run GAIA Benchmark with automatic MCP server management.

This script:
1. Starts an MCP server in the background
2. Runs the GAIA benchmark with StreamableHttpTransport
3. Cleans up the MCP server when done

Usage:
    python scripts/run_gaia_with_mcp.py --run-name my-test --model-id glm-4.5-air --base-url "https://..." --concurrency 8
"""

import argparse
import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def wait_for_mcp_server(port: int, timeout: int = 30) -> bool:
    """Wait for MCP server to be ready."""
    import socket
    start = time.time()
    while time.time() - start < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            if result == 0:
                return True
        except:
            pass
        time.sleep(0.5)
    return False


def kill_processes_on_port(port: int):
    """Kill any process using the specified port."""
    try:
        result = subprocess.run(
            f"lsof -ti:{port} | xargs kill -9 2>/dev/null",
            shell=True,
            capture_output=True
        )
    except:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Run GAIA benchmark with automatic MCP server management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--model-id", type=str, default="openrouter/anthropic/claude-3.5-sonnet")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "test"])
    parser.add_argument("--use-raw-dataset", action="store_true")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=32000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tool-calls", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="output/gaia")
    parser.add_argument("--data-dir", type=str, default="data/gaia")
    parser.add_argument("--include-all-questions", action="store_true")
    parser.add_argument("--mcp-port", type=int, default=8000)
    args = parser.parse_args()

    mcp_port = args.mcp_port
    mcp_process = None

    def cleanup():
        """Cleanup MCP server on exit."""
        if mcp_process:
            print("\nShutting down MCP server...")
            mcp_process.terminate()
            try:
                mcp_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                mcp_process.kill()
        kill_processes_on_port(mcp_port)

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda s, f: sys.exit(0))

    # Step 1: Kill any existing MCP server on the port
    print(f"Checking for existing processes on port {mcp_port}...")
    kill_processes_on_port(mcp_port)
    time.sleep(1)

    # Step 2: Start MCP server
    print(f"Starting MCP server on port {mcp_port}...")
    mcp_process = subprocess.Popen(
        [sys.executable, "-m", "athena_dr.agent.mcp_backend.main"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(Path(__file__).parent.parent),
    )

    # Wait for server to be ready
    if not wait_for_mcp_server(mcp_port, timeout=30):
        print("ERROR: MCP server failed to start!")
        cleanup()
        sys.exit(1)

    print(f"✅ MCP server started successfully (PID: {mcp_process.pid})")
    time.sleep(1)  # Give it a moment to fully initialize

    # Step 3: Set environment for StreamableHttpTransport
    os.environ["MCP_TRANSPORT"] = "StreamableHttpTransport"
    os.environ["MCP_TRANSPORT_PORT"] = str(mcp_port)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Step 4: Run the benchmark
    print("\n" + "="*60)
    print("Starting GAIA Benchmark")
    print("="*60)

    # Build command for run_gaia.py
    cmd = [
        sys.executable, "scripts/run_gaia.py",
        "--run-name", args.run_name,
        "--model-id", args.model_id,
        "--concurrency", str(args.concurrency),
        "--split", args.split,
        "--max-tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--max-tool-calls", str(args.max_tool_calls),
        "--output-dir", args.output_dir,
        "--data-dir", args.data_dir,
    ]

    if args.base_url:
        cmd.extend(["--base-url", args.base_url])
    if args.api_key:
        cmd.extend(["--api-key", args.api_key])
    if args.max_examples:
        cmd.extend(["--max-examples", str(args.max_examples)])
    if args.use_raw_dataset:
        cmd.append("--use-raw-dataset")
    if args.include_all_questions:
        cmd.append("--include-all-questions")

    # Run benchmark
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))

    print("\n" + "="*60)
    print("Benchmark completed!")
    print("="*60)

    # Cleanup happens via atexit
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
