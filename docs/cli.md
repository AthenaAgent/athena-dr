# Command Line Interface Reference

Athena DR provides several command-line tools for running deep research workflows, managing MCP servers, and processing datasets.

## CLI Commands Overview

| Command | Purpose | Entry Point |
|---------|---------|-------------|
| `athena-dr` | Main workflow CLI for running auto-search workflows and dataset generation | `athena_dr.agent.workflows.cli:launch_auto_search_workflow_cli` |
| `athena-dr-mcp` | Run the MCP (Model Context Protocol) server with search and browse tools | `athena_dr.agent.mcp_backend.main:main` |
| `athena-dr-list-tools` | List all available tools from the MCP server in a formatted table | `athena_dr.utils:list_tools_from_server` |
| `athena-dr-workflows` | Web page reader CLI to read and answer questions from web pages | `athena_dr.agent.workflows.cli:launch_web_page_reader_cli` |

---

## `athena-dr`

The main workflow CLI that provides commands for running auto-search workflows and generating dataset responses.

### Commands

#### `debug`

Test the workflow setup and configuration.

```bash
athena-dr debug [--config CONFIG_FILE]
```

| Option | Description |
|--------|-------------|
| `--config` | Path to configuration file (uses default if not specified) |

#### `generate_dataset`

Generate responses for an evaluation dataset using the auto-search workflow.

```bash
athena-dr generate-dataset DATASET_NAME [OPTIONS]
```

| Argument/Option | Description |
|-----------------|-------------|
| `DATASET_NAME` | Dataset name (e.g., `simpleqa`, `browsecomp`) |
| `--num-examples`, `-n` | Number of examples to process (or `ablation`, `final_run`) |
| `--subset`, `-s` | Dataset subset to use |
| `--max-concurrent`, `-c` | Maximum concurrent tasks (default: 5) |
| `--batch-size`, `-b` | Batch size for processing (default: 20) |
| `--use-cache` | Load from existing cache and use batch processing |
| `--output`, `-o` | Output file path |
| `--config` | Configuration file path |
| `--verbose`, `-v` | Enable verbose output |
| `--config-overrides` | Override config parameters (format: `param1=value1,param2=value2`) |
| `--num_total_workers` | Total number of workers for parallel evaluation |
| `--worker_index` | Index of current worker (0-based) |

**Example:**

```bash
athena-dr generate-dataset simpleqa -n 100 -o results.jsonl --max-concurrent 10
```

#### `rejection_sampling`

Run rejection sampling iterations for dataset generation.

```bash
athena-dr rejection-sampling DATASET_NAME --iteration N --output OUTPUT_FILE [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `--iteration` | Iteration number for rejection sampling (required) |
| `--output`, `-o` | Base output file path (appended with `-iter-N`) |
| `--threshold`, `-t` | Score threshold for rejection (default: 1.0) |
| Other options | Same as `generate-dataset` |

#### `collect_rejection_sampling_data`

Collect and merge results from multiple rejection sampling iterations.

```bash
athena-dr collect-rejection-sampling-data OUTPUT_FILE [OPTIONS]
```

| Option | Description |
|--------|-------------|
| `OUTPUT_FILE` | Output file path for collected results |
| `--max-iterations` | Maximum iterations to check (default: 10) |
| `--verbose`, `-v` | Enable verbose output |

#### `generate-sft-trace`

Generate SFT (Supervised Fine-Tuning) traces from a dataset using the TraceGenerator. This command loads a dataset, generates traces using the AutoReasonSearchWorkflow, applies rejection sampling to filter out incorrect answers, and optionally exports the resulting traces to Hugging Face Hub.

```bash
athena-dr generate-sft-trace DATASET_NAME --prompt-column COLUMN --gt-answer-column COLUMN [OPTIONS]
```

| Argument/Option | Description | Default |
|-----------------|-------------|---------|
| `DATASET_NAME` | Name of the dataset to load (e.g., `hotpotqa/hotpot_qa`) | (required) |
| `--prompt-column`, `-p` | Column name containing the prompts/questions | (required) |
| `--gt-answer-column`, `-g` | Column name containing ground truth answers | (required) |
| `--auto-search-config`, `-a` | Path to auto search configuration file | `configs/auto_search_configs.yml` |
| `--rejection-sampling-config`, `-r` | Path to rejection sampling configuration file | `configs/rejection_sampling_configs.yml` |
| `--dataset-subset`, `-s` | Dataset subset/configuration to load | `None` |
| `--dataset-split` | Dataset split to use | `train` |
| `--max-examples`, `-n` | Maximum number of examples to process | `None` (all) |
| `--max-attempts`, `-m` | Maximum attempts per example for rejection sampling | `3` |
| `--export-dataset`, `-e` | Hugging Face Hub dataset name to export traces to | `None` |
| `--project` | Weave project name for tracing | `athena_dr` |

**Example:**

```bash
# Generate SFT traces from HotpotQA dataset
athena-dr generate-sft-trace hotpotqa/hotpot_qa \
    --prompt-column question \
    --gt-answer-column answer \
    --dataset-subset distractor \
    --max-examples 100 \
    --max-attempts 3 \
    --export-dataset username/hotpotqa_sft_traces
```

The command will:

1. Load the specified dataset from Hugging Face Hub
2. For each example, generate a trace using the AutoReasonSearchWorkflow
3. Check if the generated answer is correct using rejection sampling
4. Retry up to `--max-attempts` times if the answer is incorrect
5. Collect successful traces and optionally export them to Hugging Face Hub

---

## `athena-dr-mcp`

Run the MCP (Model Context Protocol) server that exposes search and web browsing tools.

### Usage

```bash
athena-dr-mcp [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--transport` | Transport protocol (`stdio`, `http`, `sse`, `streamable-http`) | `http` |
| `--port` | Port to bind to (for HTTP transports) | `8000` |
| `--host` | Host address to bind to | `127.0.0.1` |
| `--path` | Path for the HTTP endpoint | `/mcp` |
| `--log-level` | Log level (`debug`, `info`, `warning`, `error`, `critical`) | `info` |
| `--no-cache` | Disable API response caching | (caching enabled) |

### Available Tools

The MCP server exposes the following tools:

| Tool | Tags | Description |
|------|------|-------------|
| `semantic_scholar_search` | search, necessary | Search academic papers via Semantic Scholar API |
| `semantic_scholar_snippet_search` | search | Retrieve focused snippets from scientific papers |
| `pubmed_search` | search | Search medical/scientific papers via PubMed API |
| `massive_serve_search` | search, necessary | Dense passage retrieval via massive-serve API |
| `serper_google_webpage_search` | search, necessary | General web search using Google (Serper.dev) |
| `serper_google_scholar_search` | search, necessary | Academic paper search via Google Scholar |
| `serper_fetch_webpage_content` | browse, necessary | Fetch webpage content via Serper.dev API |
| `jina_fetch_webpage_content` | browse | Fetch webpage content via Jina Reader API |
| `crawl4ai_fetch_webpage_content` | browse, necessary | Extract webpage content using Crawl4AI (local) |
| `crawl4ai_docker_fetch_webpage_content` | browse, necessary | Extract webpage content using Crawl4AI (Docker) |
| `webthinker_fetch_webpage_content` | browse | Extract text from URLs using advanced web parsing |
| `webthinker_fetch_webpage_content_async` | browse | Async version of webthinker content extraction |
| `vllm_hosted_reranker` | rerank, necessary | Rerank documents using VLLM-hosted reranker |

**Example:**

```bash
# Start HTTP server on port 8000
athena-dr-mcp --transport http --port 8000

# Start with stdio transport (for local MCP clients)
athena-dr-mcp --transport stdio

# Start with caching disabled
athena-dr-mcp --no-cache
```

---

## `athena-dr-list-tools`

List all available tools from the MCP server in a formatted, readable table.

### Usage

```bash
athena-dr-list-tools
```

This command connects to the MCP server and displays all registered tools with their:

- Tool name
- Description
- Parameters (name, type, required status, default value)

---

## `athena-dr-workflows`

Launch the web page reader CLI to read web pages and answer questions based on their content.

### Usage

```bash
athena-dr-workflows URL [URL...] --question QUESTION [OPTIONS]
```

### Arguments and Options

| Argument/Option | Description |
|-----------------|-------------|
| `URL` | One or more URLs of web pages to read |
| `--question`, `-q` | The question to answer from the web pages (required) |
| `--project`, `-p` | Weave project name for tracing (default: `athena_dr`) |

**Example:**

```bash
athena-dr-workflows https://example.com/article1 https://example.com/article2 \
    --question "What are the main findings discussed in these articles?"
```

---

## Environment Variables

Several environment variables can be used to configure the CLI tools:

| Variable | Description |
|----------|-------------|
| `MCP_INCLUDE_TAGS` | Comma-separated list of tool tags to include (default: `search,browse,rerank`) |
| `SERPER_API_KEY` | API key for Serper.dev services |
| `SEMANTIC_SCHOLAR_API_KEY` | API key for Semantic Scholar |
| `JINA_API_KEY` | API key for Jina Reader |
| `CRAWL4AI_BLOCKLIST_PATH` | Path to blocklist file for Crawl4AI AI2 config |
| `AZURE_API_KEY` | API key for Azure services |