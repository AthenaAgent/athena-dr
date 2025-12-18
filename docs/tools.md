# MCP Server Tools

The RL-RAG MCP server provides a collection of tools for search, browsing, and reranking. Tools are tagged to enable selective inclusion via the `MCP_INCLUDE_TAGS` environment variable.

## Tool Categories

- **search**: Tools for searching documents and web content
- **browse**: Tools for fetching and parsing webpage content
- **rerank**: Tools for reranking search results
- **necessary**: Core tools marked as essential

---

## Search Tools

### `serper_google_webpage_search`

General web search using Google Search (based on Serper.dev API).

**Tags**: `search`, `necessary`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | `str` | Yes | - | Search query string |
| `num_results` | `int` | No | `10` | Number of results to return |
| `gl` | `str` | No | `"us"` | Geolocation - country code to boost search results |
| `hl` | `str` | No | `"en"` | Host language of user interface |

**Returns**: Dictionary containing `organic` results, `knowledgeGraph`, `peopleAlsoAsk`, and `relatedSearches`.

---

### `massive_serve_search`

Search for documents using massive-serve API for dense passage retrieval.

**Tags**: `search`, `necessary`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | `str` | Yes | - | Search query string |
| `n_docs` | `int` | No | `10` | Number of documents to return |
| `domains` | `str` | No | `"dpr_wiki_contriever_ivfpq"` | Domain/index to search in |
| `base_url` | `str` | No | `None` | Base URL for the massive-serve API |
| `nprobe` | `int` | No | `None` | Number of probes for search |

**Returns**: Dictionary containing `message`, `query`, `n_docs`, `results`, and parsed `data` list.

---

## Browse Tools

### `serper_fetch_webpage_content`

Fetch the content of a webpage using Serper.dev API.

**Tags**: `browse`, `necessary`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `webpage_url` | `str` | Yes | - | The URL of the webpage to fetch |
| `include_markdown` | `bool` | No | `True` | Whether to include markdown formatting in the response |

**Returns**: Dictionary containing `text`, `markdown`, `metadata`, `url`, `success`, and optionally `error`.

---

### `jina_fetch_webpage_content`

Fetch the content of a webpage using Jina Reader API with timeout support.

**Tags**: `browse`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `webpage_url` | `str` | Yes | - | The URL of the webpage to fetch |
| `timeout` | `int` | No | `30` | Request timeout in seconds |

**Returns**: Dictionary containing `url`, `title`, `content`, `description`, `publishedTime`, `metadata`, `success`, and optionally `error`.

---

### `crawl4ai_fetch_webpage_content`

Open a specific URL and extract readable page text using Crawl4AI (local headless browser).

**Tags**: `browse`, `necessary`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | `str` | Yes | - | URL to fetch and extract content from |
| `ignore_links` | `bool` | No | `True` | If True, remove hyperlinks in markdown |
| `use_pruning` | `bool` | No | `False` | Apply pruning content filter to extract main content |
| `bm25_query` | `str` | No | `None` | Optional query to enable BM25-based content filtering |
| `bypass_cache` | `bool` | No | `True` | If True, bypass Crawl4AI cache |
| `timeout_ms` | `int` | No | `80000` | Per-page timeout in milliseconds |
| `include_html` | `bool` | No | `False` | Whether to include raw HTML in the response |

**Returns**: `Crawl4AiResult` with extracted webpage content including markdown-formatted text.

---

### `webthinker_fetch_webpage_content_async`

Asynchronously extract text content from a single URL (webpage or PDF) using advanced web parsing.

**Tags**: `browse`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `url` | `str` | Yes | - | URL to extract text from |
| `snippet` | `str` | No | `None` | Optional snippet to search for and extract context around |
| `keep_links` | `bool` | No | `False` | Whether to preserve links in the extracted text |

**Returns**: Dictionary containing `url` and extracted `text` content.

---

## Rerank Tools

### `vllm_hosted_reranker`

Rerank a list of documents based on their relevance to the query using VLLM hosted reranker.

**Tags**: `rerank`, `necessary`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | `str` | Yes | - | Search query string |
| `documents` | `List[str]` | Yes | - | List of document texts to rank |
| `top_n` | `int` | Yes | - | Number of top documents to return |
| `model_name` | `str` | Yes | - | Name of the reranker model (e.g., `"BAAI/bge-reranker-v2-m3"`) |
| `api_url` | `str` | Yes | - | Base URL for the VLLM reranker API |

**Returns**: `RerankerResult` containing reranker results with `method`, `model_name`, and ranked results.

---

## Health Check Endpoint

The server also exposes a health check endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Returns `OK` if the server is running.


## Redundant tools and why we keep them

There are 3 reduntant tools:

1. `jina_fetch_webpage_content`
2. `crawl4ai_fetch_webpage_content`
3. `webthinker_fetch_webpage_content_async`

### Comparison

| Aspect | [jina_fetch_webpage_content](cci:1://file:///Users/geekyrakshit/Workspace/athena/athena-dr/athena_dr/agent/mcp_backend/main.py:199:0-223:17) | [crawl4ai_fetch_webpage_content](cci:1://file:///Users/geekyrakshit/Workspace/athena/athena-dr/athena_dr/agent/mcp_backend/main.py:287:0-327:17) | [webthinker_fetch_webpage_content_async](cci:1://file:///Users/geekyrakshit/Workspace/athena/athena-dr/athena_dr/agent/mcp_backend/main.py:268:0-312:37) |
|--------|------------------------------|----------------------------------|------------------------------------------|
| **Backend** | Jina Reader API (remote) | Crawl4AI (local browser) | Custom aiohttp parser (local) |
| **Execution** | Sync | Async | Async |
| **Tags** | `browse` | `browse`, `necessary` | `browse` |
| **Timeout** | 30s | 80s | 240s |

#### Feature Differences

| Feature | Jina | Crawl4AI | WebThinker |
|---------|------|----------|------------|
| **BM25 query filtering** | ❌ | ✅ | ❌ |
| **Content pruning** | ❌ | ✅ | ❌ |
| **Snippet extraction** | ❌ | ❌ | ✅ |
| **Link removal option** | ❌ | ✅ | ✅ (inverse: `keep_links`) |
| **Raw HTML output** | ❌ | ✅ | ❌ |
| **PDF support** | ❌ | ❌ | ✅ |
| **Caching control** | ❌ | ✅ | ❌ |
| **Rich metadata** | ✅ (title, description, publishedTime) | ❌ | ❌ |

#### Summary

- **[jina_fetch_webpage_content](cci:1://file:///Users/geekyrakshit/Workspace/athena/athena-dr/athena_dr/agent/mcp_backend/main.py:199:0-223:17)** — Simplest option; uses external Jina API, returns rich metadata (title, description, publish time).
- **[crawl4ai_fetch_webpage_content](cci:1://file:///Users/geekyrakshit/Workspace/athena/athena-dr/athena_dr/agent/mcp_backend/main.py:287:0-327:17)** — Most feature-rich; runs a local headless browser, supports BM25 filtering and pruning to extract relevant content.
- **[webthinker_fetch_webpage_content_async](cci:1://file:///Users/geekyrakshit/Workspace/athena/athena-dr/athena_dr/agent/mcp_backend/main.py:268:0-312:37)** — Lightweight async parser; supports PDFs and snippet-based context extraction.