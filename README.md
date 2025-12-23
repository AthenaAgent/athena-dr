# Athena Deep Research

Our homegrown deep research model.

## Installation

```bash
git clone https://github.com/AthenaAgent/athena-dr.git
cd athena-dr
uv sync --all-extras --all-groups
```

## Usage

First, start the code execution container using the following command:

```bash
docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20250609
```

Next, set the .env file with the following variables:

```
S2_API_KEY=<SEMANTICSCHOLAR_API_KEY>
SERPAPI_API_KEY=<SERPAPI_API_KEY>
SERPER_API_KEY=<SERPER_API_KEY>
JINA_API_KEY=<JINA_API_KEY>
OPENROUTER_API_KEY=<OPENROUTER_API_KEY>
AZURE_API_KEY=<AZURE_API_KEY>
MCP_CACHE_DIR=".cache"
JUSPAY_API_KEY="dummy"
SPORTSDB_API_KEY=<SPORTSDB_API_KEY>
S2_API_TIMEOUT=10
```

To run a single-query search, use the following command:

```bash
python scripts/test_auto_search.py
```

To generate SFT traces, use the following command:

```bash
python scripts/generate_sft_traces.py
```

# Docs

1. Install the dependencies using `uv sync --group docs`
2. Serve the docs using `mkdocs serve`

The documentation is available at http://127.0.0.1:8000