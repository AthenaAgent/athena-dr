# Target Benchmarks

## [FRAMES: Factuality, Retrieval, And reasoning MEasurement Set](https://huggingface.co/datasets/google/frames-benchmark)
A comprehensive evaluation dataset designed to test the capabilities of Retrieval-Augmented Generation (RAG) systems across factuality, retrieval accuracy, and reasoning.

| Agent                  | Base models      | FRAMES |
| ---------------------- | ---------------- | ------ |
| **Proprietary Agents** |                  |        |
| Deep Research          | o3               | -      |
| GPT-5                  | GPT-5            | -      |
| GPT-5-Pro              | GPT-5-Pro        | -      |
| o4-mini                | o4-mini          | -      |
| Kimi-researcher        | Kimi-k1.5/k2     | 78.8†  |
| gpt-oss-20b            | gpt-oss-20b      | -      |
| gpt-oss-120b           | gpt-oss-120b     | -      |
| **Multi Agents**       |                  |        |
| OpenDeepSearch-R1      | Deepseek-R1-671B | 72.4*  |
| OpenDeepSearch-QwQ     | QwQ-32B          | 54.1*  |
| MiroThinker-8B         | Qwen3-8B&235B    | 64.4†  |
| MiroThinker-32B        | Qwen3-32B&235B   | 71.7†  |
| WebThinker-32B         | QwQ-32B          | 35.5*  |
| **Single Agents**      |                  |        |
| WebSailor-32B          | Qwen2.5-32B      | 69.78* |
| WebShaper-32B          | QwQ-32B          | 69.42* |
| AFM-32B                | Qwen2.5-32B      | 55.3†  |
| SFR-DR-8B              | Qwen3-8B         | 63.3   |
| SFR-DR-32B             | QwQ-32B          | 72.0   |
| SFR-DR-20B             | gpt-oss-20b      | 82.8   |
| Tongyi-DR-30B          | Qwen3-32B        | 90.6   |

## [BrowseComp: a benchmark for browsing agents](https://openai.com/index/browsecomp/)

A simple and challenging benchmark that measures the ability of AI agents to locate hard-to-find information.

| MODEL | SCORE |
| --- | --- |
| **Kimi K2-Thinking-0905**<br>Moonshot AI | 0.602 |
| **DeepSeek-V3.2 (Thinking)**<br>DeepSeek | 0.514 |
| **GLM-4.6**<br>Zhipu AI | 0.451 |
| **MiniMax M2**<br>MiniMax | 0.440 |
| **Tongyi-DR-30B**<br>Alibaba | 0.434 |
| **DeepSeek-V3.2-Exp**<br>DeepSeek | 0.401 |
| **DeepSeek-V3.1**<br>DeepSeek | 0.300 |
| **GLM-4.5**<br>Zhipu AI | 0.264 |
| **GLM-4.5-Air**<br>Zhipu AI | 0.213 |
| **DeepSeek-R1-0528**<br>DeepSeek | 0.089 |

## [Xbench DeepSearch](https://xbench.org/agi/aisearch)

DeepSearch is part of xbench's AGI-Aligned series, focused on evaluating tool usage capabilities in search and information retrieval scenarios.

| Product | Accuracy | Cost/Task | Time Cost/Task |
| --- | --- | --- | --- |
| ChatGPT-5-Pro | 75+ | ~$0.085 | 5-8min |
| Tongyi-DR-30B | 75.0 | Free | - |
| SuperGrok Expert | 40+ | ~$0.08 | 3-5min |
| Minimax Agent | 35+ | $69 | 2-3min |
| StepFun Research | 35+ | Free | 8-15min |
| Flowith | 35+ | ~$0.1 | 8-15min |
| Skywork | 35+ | ~$0.55 | 3-5min |
| Manus Agent (Quality Mode) | 35+ | ~$0.63 | 3-5min |
| Doubao (Deep Research) | 35+ | Free | 5-8min |
| Fellou | 35+ | ~$2 | 5-8min |
| Genspark Super Agent | 30+ | ~$0.15 | 3-5min |
| Coze Space | 30+ | Free | 2-3min |
