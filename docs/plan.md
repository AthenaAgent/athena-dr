# Plan

## Objective

What Deepresearch means to us?
 -  current gpt/gemini deepresearch dumps either too much information or too high level summary over all searched information
 -  we hope we can have progressive explaintions eg top level first and then more detailed explanations for each step
 -  by top level explantions we can skip the part not important in deeper explanations
 -  also better verifiabe citations linking what exactly its referring to no just link of referaces
 

The objective is to train a comprehensive deep research agent system in the 8B parameter range (Usuing Qwen-3 8B or a similar model as the base model) that is competitive with single agent models of similar size on the following benchmarks:
1. [Frames](./benchmarks.md#frames-factuality-retrieval-and-reasoning-measurement-set) benchmark (> 63.3 which is achieved by SFR-DR-8B).
2. [BrowseComp](./benchmarks.md#browsecomp-a-benchmark-for-browsing-agents) benchmark (> 0.434 which is achieved by Tongyi-DR-30B).
3. [Xbench DeepSearch](./benchmarks.md#xbench-deepsearch) benchmark (> 75.0 which is achieved by Tongyi-DR-30B).
4. [WebWalkerQA](https://huggingface.co/datasets/callanwu/WebWalkerQA) benchmark (> 72.2 which is achieved by Tongyi-DR-30B).
5. [SimpleQA](./benchmarks.md#simpleqa) benchmark (> 98.6 which is achieved by Tongyi-DR-30B).

## Approach

The approach involves 2 steps:

1. **Identify base model:**
    - [Qwen3 8B](https://huggingface.co/Qwen/Qwen3-8B) * <-------------------------------
        - max context length is 128k with yarn
    - [Olmo3 7b](https://huggingface.co/allenai/Olmo-3-7B-Think)
      - olmo3 8b use MHA not GQA so need more varm for inference
    - [rnj-1 8b](https://essential.ai/research/rnj-1)
        - max context length is 32k with yarn
    - [trinity 6b 1B MOE](https://huggingface.co/arcee-ai/Trinity-Nano-Base) 
        - we dont want to focus on moe, just post tranining recipe for now

    if 32b:
    - nemotron3 32b A3b
    - qwen3 32b A3b
    - olmo3 32b
    - trinity 32b

2. **Deep Research Agent Scaffold**: We will use a custom scaffold built on top of DR-Tulu's agentic scaffold, extending its capabilities with additional research-specific tools. The reason to build on top of DR-Tulu is because of its simple MCP-based ReACT architecture that's both easy to extend with more tools, and battle hardened through real-world use.
    - Available tools:
        - `semantic_scholar_search`: Search for academic papers using Semantic Scholar API.
        - `semantic_scholar_snippet_search`: Focused snippet retrieval from scientific papers using Semantic Scholar API.
        - `pubmed_search`: Search for medical and scientific papers using PubMed API.
        - `vllm_hosted_reranker`: Rerank a list of documents based on their relevance to the query using VLLM hosted reranker.
        - `massive_serve_search`: Search for documents using massive-serve API for dense passage retrieval.
        - `serper_google_webpage_search`: General web search using Google Search (based on Serper.dev API). Perform general web search to find relevant webpages, articles, and online resources.
        - `serper_google_scholar_search`: Search for academic papers using google scholar (based on Serper.dev API).
        - `serper_fetch_webpage_content`: Fetch the content of a webpage using Serper.dev API.
        - `jina_fetch_webpage_content`: Fetch the content of a webpage using Jina Reader API with timeout support.
        - `crawl4ai_fetch_webpage_content`: Open a specific URL and extract readable page text as snippets using Crawl4AI.
        - `crawl4ai_docker_fetch_webpage_content`: Open a specific URL and extract readable page text as snippets using Crawl4AI Docker API.
        - `webthinker_fetch_webpage_content`: Extract text content from a single URL (webpage or PDF) using advanced web parsing.
        - `webthinker_fetch_webpage_content_async`: Asynchronously extract text content from a single URL (webpage or PDF) using advanced web parsing.
    - Suggestions for more tools:
        - `memory_read`: Read from a memory store for persistent context.
        - `memory_write`: Write to a memory store for persistent context. The memory tools would encourage the model to perform long-horizon reasoning beyond the limitations of its effective context length.
        - `code_interpreter`: Execute shell/python code snippets in a sandboxed environment.
    - Why more tools are needed?
        - Overall, this suggests that using multiple complementary search tools and allowing the model to adaptively select among them can improve both prediction quality and cost efficiency (dr tulu page 13 para 1)
        - R1-32B is not able to decompose the complex query into individual components, consequently only making ambiguous queries that involve too many unknown information. The agent also has severe hallucinations, producing conclusions that are not supported by the search results. Finally, it fails to resolve all unknown information. This case study shows that existing online RL approaches only incentivize elementary search strategies. It is also worth noting that, since the turn limit is set as a small value, e.g. 4, during training, the model only exhibits a short tool-use horizon (Beyond Ten Turns: https://arxiv.org/pdf/2508.07976 | section 2)
        - https://www.youtube.com/watch?v=CEvIs9y1uog&t=92s

3. **Supervised Fine-tuning**: Model already knows to tool calls, we just need to teach them to work on long horizon tasks (so no mid training needed). The plan is to train the model on high-quality instruction-following and reasoning traces generated by putting strong frontier models in deep research agent scaffolds. If we decide to proceed with Tulu-DR scaffold, we can leverage their [SFT dataset](https://huggingface.co/datasets/rl-research/dr-tulu-sft-data) as a starting point. This dataset included rejection sampled traces from the following (but not limited to):
    - [OpenScholar](https://huggingface.co/datasets/allenai/openscilm_queries)
    - [Search Arena](https://huggingface.co/datasets/lmarena-ai/search-arena-24k)
    - short-form QA datasets including [WebWalker-Silver](https://huggingface.co/datasets/callanwu/WebWalkerQA)
    - [TaskCraft](https://huggingface.co/datasets/PersonalAILab/TaskCraft)
    - [PopQA](https://huggingface.co/datasets/akariasai/PopQA)
    - [TyDiQA (English)](https://github.com/google-research-datasets/tydiqa)
    - [MegaScience](https://huggingface.co/datasets/MegaScience/MegaScience)
    - [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
    - [ScholarQA](https://allenai.org/blog/ai2-scholarqa)

4. **Reinforcement Learning Fine-tuning**: If we have time and compute, we will do RL using the same tools as SFT. If we need to customize the model to for an enterprise customer with cutom tool calls, we can also introduce them in scaffold the RL step.
