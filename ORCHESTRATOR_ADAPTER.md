# Open Deep Research Adapter for `rag_runtime_runner.py`

This adapter exposes Open Deep Research as:

- `POST /responses` (Responses-compatible client answer endpoint)
- `GET /tools` and `POST /tools` (ToolHarness contract used by Judge4)

Important behavior: `/tools` is built from **ODR's own `get_all_tools(config)`**. Judge4 only sees executable tools that ODR itself has access to.

## 1. Install dependencies

From repo root:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

From `experiments/open_deep_research` (if ODR deps are missing in root venv):

```bash
source ../../.venv/bin/activate
pip install -e .
```

## 2. Start adapter

```bash
source .venv/bin/activate
python experiments/open_deep_research/orchestrator_adapter.py \
  --host 127.0.0.1 \
  --port 4581
```

## 3. ODR runtime env (optional)

Core:

- `ODR_SEARCH_API=tavily|openai|openrouter|anthropic|none`
- `ODR_ALLOW_CLARIFICATION=false`
- `ODR_MAX_RESEARCHER_ITERATIONS=4`
- `ODR_RESEARCH_MODEL=openai:gpt-4.1`
- `ODR_FINAL_REPORT_MODEL=openai:gpt-4.1`

OpenRouter web search:

- `ODR_OPENROUTER_WEB_SEARCH_ENGINE=auto|native|exa|firecrawl|parallel`
- `ODR_OPENROUTER_WEB_SEARCH_MAX_RESULTS=5`
- `ODR_OPENROUTER_WEB_SEARCH_MAX_TOTAL_RESULTS=20`
- `ODR_OPENROUTER_WEB_SEARCH_CONTEXT_SIZE=low|medium|high`
- `ODR_OPENROUTER_WEB_SEARCH_ALLOWED_DOMAINS=arxiv.org,nature.com`
- `ODR_OPENROUTER_WEB_SEARCH_EXCLUDED_DOMAINS=reddit.com`

MCP (optional):

- `ODR_MCP_URL=https://...`
- `ODR_MCP_TOOLS=tool_a,tool_b`
- `ODR_MCP_AUTH_REQUIRED=true|false`
- `ODR_MCP_PROMPT=...`

Note: for Judge4 tool execution through `/tools`, `tavily` is the most practical mode because it yields executable function tools. Native provider and OpenRouter web-search modes may rely on non-function tool types.
