# Bitext Customer Service Data Analyst Agent

A LangGraph-based data analyst agent for the Bitext Customer Service dataset.

The agent answers structured analytical questions and qualitative dataset questions while staying scoped to:

- the Bitext Customer Service dataset
- the current conversation context
- the saved user profile

It uses a graph-owned planner → tool executor → observation reviewer loop, semantic dataset filters, visible traces, and persistent session/profile state.

## Features

- LangGraph workflow with a graph-owned planner → tool executor → observation reviewer loop
- LLM-based query router
- SQLite-backed LangGraph checkpoint persistence for conversation state
- Semantic dataset tools with Pydantic input validation
- LLM-backed qualitative row summarization with deterministic fallback
- Persistent user profile memory
- CLI chat interface
- Streamlit chat interface
- FastMCP server exposing dataset/profile tools
- Visible reasoning traces with route decisions, tool calls, reviewer decisions, inputs, and observations
- Deterministic handling for high-risk follow-up example pagination such as `Show me 3 more.`
- Tests for loader, tools, router, memory, graph behavior, MCP wrappers, and config behavior

## Dataset

Dataset source:

```text
bitext/Bitext-customer-support-llm-chatbot-training-dataset
```

The app loads the dataset through Hugging Face `datasets`, normalizes DataFrame columns, adds a stable integer `row_id`, and caches the normalized CSV locally under:

```text
data/bitext_customer_service.csv
```

Expected core analysis columns:

```text
instruction
response
category
intent
```

The code normalizes known column aliases and works from these core analysis columns.

## Project Structure

```text
bitext_agent/
├── README.md
├── pyproject.toml
├── data/
│   └── .gitkeep
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── streamlit_app.py
│   ├── mcp_server.py
│   ├── graph.py
│   ├── state.py
│   ├── router.py
│   ├── tools.py
│   ├── data_loader.py
│   ├── memory.py
│   ├── prompts.py
│   ├── config.py
│   ├── logging_utils.py
│   └── agent/
│       ├── __init__.py
│       ├── context.py
│       ├── evidence_contracts.py
│       ├── followups.py
│       ├── llm_factory.py
│       ├── loop.py
│       ├── profile.py
│       ├── schemas.py
│       └── tool_executor.py
├── tests/
│   ├── test_config.py
│   ├── test_data_loader.py
│   ├── test_tools.py
│   ├── test_router.py
│   ├── test_memory.py
│   ├── test_graph.py
│   └── test_mcp_server.py
├── .checkpoints/
│   └── checkpoint.sqlite
└── .user_profiles/
    └── <user_id>/
        └── context.md
```

Runtime folders such as `data/`, `.checkpoints/`, and `.user_profiles/` are created automatically when needed.

## Environment Setup

This project uses Nebius Token Factory through an OpenAI-compatible API.

Create a `.env` file in the project root:

```env
NEBIUS_API_KEY=your_nebius_api_key_here
```

Install dependencies:

```bash
uv sync
```

Or with pip:

```bash
pip install -e ".[dev]"
```

The model names, Nebius base URL, runtime folders, and iteration limits are configured in:

```text
app/config.py
```

Current model configuration in the codebase:

```text
router_model = nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B
agent_model = nvidia/Llama-3_1-Nemotron-Ultra-253B-v1
nebius_base_url = https://api.tokenfactory.nebius.com/v1/
```

## Nebius Model Role Split

All LLM calls use Nebius Token Factory through the OpenAI-compatible API. No non-Nebius model provider is used for routing, agent reasoning, profile-update decisions, or row summarization.

- Router model:
  - Used for structured route classification into `structured`, `unstructured`, or `out_of_scope`.
  - Uses a small direct Responses API client and requests JSON-object output.
  - Chosen because routing is a short, low-temperature classification task where a smaller model is faster and cheaper.
- Main data-analysis agent model:
  - Used by the planner and observation reviewer for tool selection, answer-readiness checks, and grounded final answers.
  - Chosen because the main loop needs stronger instruction following, reliable tool planning, and careful observation review.
- Profile-update model:
  - Used to decide whether a user message contains a durable fact/preference worth saving.
  - Uses the same larger agent model because profile updates require conservative instruction following.
- Summarizer model:
  - Used by `summarize_rows` when LLM summarization is available.
  - Uses the same larger agent model because summaries must stay grounded in selected dataset rows.
  - Falls back to deterministic summary if the summarizer is unavailable.

## Running the CLI Agent

Run:

```bash
python -m app.main --session demo --user max --max-iterations 12
```

Example interaction:

```text
Bitext Customer Service Data Analyst Agent
Session: demo
User: max
Max iterations: 12
Type 'exit' or 'quit' to stop.

You: How many refund requests did we get?

[router] structured
[router_reason] The user asks for an exact dataset count.

[tool] resolve_filter_value
[input]
{
  "query": "refund requests",
  "columns": [
    "category",
    "intent"
  ],
  "top_k": 5
}
[observation] {"query":"refund requests","candidates":[...],"recommended_filter":{"category":"REFUND","intent":null},"confidence":"high"}

[tool] count_rows
[input]
{
  "category": "REFUND",
  "intent": null,
  "text_query": null
}
[observation] {"count":2992,"applied_filters":{"category":"REFUND","intent":null,"text_query":null}}

[reviewer] answered
[reason] The observations are sufficient.

Agent: There are 2,992 refund-request rows in the dataset.
```

## Running the Streamlit App

Run:

```bash
streamlit run app/streamlit_app.py
```

The sidebar includes:

- Session ID
- User ID
- Max iterations
- Clear visible chat only
- Start fresh session

The app shows:

- chat messages
- the latest reasoning trace in a separate side panel
- router decision
- tool calls and observations
- reviewer decisions

The Streamlit app uses the same `invoke_agent()` function as the CLI.

## Running the FastMCP Server

Run:

```bash
python -m app.mcp_server
```

The MCP server exposes these tools:

```text
read_user_profile
get_dataset_schema
resolve_filter_value
count_rows
sample_examples
group_counts
summarize_rows
```

The MCP server does **not** expose:

```text
filter_rows
update_user_profile
```

The dataset tools use semantic filters directly:

```text
category
intent
text_query
```

Row-id list workflows are intentionally not exposed.

## MCP Tool Reply Format

Each MCP tool returns a plain dictionary created from the corresponding Pydantic output model with:

```python
result.model_dump()
```

### read_user_profile

```json
{
  "user_id": "max",
  "profile": "# User Profile\n\n- User prefers file-by-file implementation review.\n"
}
```

### get_dataset_schema

```json
{
  "columns": ["row_id", "instruction", "response", "category", "intent"],
  "row_count": 26872,
  "sample_values": {
    "category": ["ORDER", "SHIPPING", "ACCOUNT"],
    "intent": ["track_order", "recover_password"]
  }
}
```

If `include_sample_values=false`, then `sample_values` is `null`.

### resolve_filter_value

```json
{
  "query": "refund requests",
  "candidates": [
    {
      "column": "category",
      "value": "REFUND",
      "count": 2992,
      "score": 1.0,
      "reason": "Category alias resolves exactly to this dataset value."
    }
  ],
  "recommended_filter": {
    "category": "REFUND",
    "intent": null
  },
  "confidence": "high"
}
```

`confidence` is one of:

```text
none
low
medium
high
```

### count_rows

```json
{
  "count": 2992,
  "applied_filters": {
    "category": "REFUND",
    "intent": null,
    "text_query": null
  }
}
```

### sample_examples

```json
{
  "examples": [
    {
      "row_id": 20,
      "instruction": "Where is my package?",
      "response": "You can track it from your account.",
      "category": "SHIPPING",
      "intent": "track_order"
    }
  ],
  "next_offset": 5,
  "match_count": 6,
  "applied_filters": {
    "category": "SHIPPING",
    "intent": null,
    "text_query": null
  }
}
```

### group_counts

```json
{
  "group_by": "intent",
  "counts": [
    {
      "label": "recover_password",
      "count": 997
    },
    {
      "label": "delete_account",
      "count": 995
    }
  ],
  "match_count": 5986,
  "applied_filters": {
    "category": "ACCOUNT",
    "intent": null,
    "text_query": null
  }
}
```

### summarize_rows

```json
{
  "summary": "Agents explain cancellation policy and guide customers through next steps.",
  "row_count_used": 100,
  "match_count": 950,
  "focus": "How agents respond to cancellation requests",
  "target_field": "response",
  "applied_filters": {
    "category": "CANCEL",
    "intent": null,
    "text_query": null
  }
}
```

### MCP Input Validation

The MCP wrappers validate inputs through the same Pydantic input models used by the local tool executor before calling implementation functions.

Invalid calls fail before the implementation runs.

Examples of invalid values:

```text
resolve_filter_value(top_k=0)
sample_examples(n=999)
sample_examples(offset=-1)
group_counts(top_k=100000)
summarize_rows(max_examples=99999)
```

## Connecting an MCP Client

The default MCP server entrypoint runs FastMCP with stdio transport:

```bash
python -m app.mcp_server
```

For an external MCP client that supports stdio servers, configure the server command as:

```json
{
  "mcpServers": {
    "bitext-data-tools": {
      "command": "python",
      "args": ["-m", "app.mcp_server"]
    }
  }
}
```

A minimal direct Python smoke test can also call the same MCP wrapper functions used by the server:

```python
from app import mcp_server


def main() -> None:
    result = mcp_server.group_counts(
        group_by="category",
        top_k=10,
    )
    print(result)


if __name__ == "__main__":
    main()
```

Expected shape of the returned value:

```python
{
    "group_by": "category",
    "counts": [
        {
            "label": "REFUND",
            "count": 2992,
        }
    ],
    "match_count": 26872,
    "applied_filters": {
        "category": None,
        "intent": None,
        "text_query": None,
    },
}
```

If you run FastMCP with an HTTP/SSE transport instead, use the transport URL printed in the server startup logs. The exact URL depends on the FastMCP runtime configuration.

## Tool Reference

### get_dataset_schema

Returns dataset columns, row count, and optional sample values.

Important: sample values are examples only. They are not treated as the complete set of distinct categories or intents.

### resolve_filter_value

Resolves a natural-language phrase to actual dataset `category` and/or `intent` values.

Examples:

```text
refund requests
shipping issues
account problems
people wanting their money back
```

Use this before filtered analytical tools when the user provides or implies a category/intent value.

### count_rows

Counts all rows or rows matching semantic filters:

```text
category
intent
text_query
```

### sample_examples

Returns example rows matching semantic filters.

Supports:

```text
n
offset
```

`offset` is used for follow-up requests like:

```text
Show me 3 more.
```

### group_counts

Groups all rows or filtered rows by:

```text
category
intent
```

Supports scoped distributions:

```text
group_counts(group_by="intent", category="ACCOUNT")
group_counts(group_by="category", intent="track_refund")
```

For full distributions, use `top_k=20` or higher unless the user explicitly asks for a limited top-N result.

### summarize_rows

Summarizes matching rows for qualitative analysis.

Use for:

- summaries
- themes
- patterns
- tone
- support response patterns
- customer pain points

Arguments include:

```text
category
intent
text_query
focus
target_field
max_examples
```

`target_field` can be:

```text
instruction
response
both
```

When `NEBIUS_API_KEY` and `langchain-openai` are available, this tool uses the configured agent model to summarize only selected rows. If summarization is unavailable, it falls back to a deterministic summary.

### read_user_profile

Reads the persistent profile for a user.

### update_user_profile

Updates the profile with a durable observation.

This is called by the graph profile-update node and is not exposed as an MCP tool.

## Example Questions

Structured questions:

```text
How many refund requests did we get?
Show me 3 examples from the REFUND category.
Show me 5 examples of the SHIPPING category.
What categories exist in the dataset?
What intents exist in the dataset?
Show the distribution of categories.
Show the distribution of intents.
What is the distribution of intents in the ACCOUNT category?
Break down ACCOUNT by intent.
Intent breakdown for ACCOUNT.
What intents appear under ACCOUNT?
Break down track_refund by category.
What categories appear under track_refund?
```

Unstructured questions:

```text
Summarize the FEEDBACK category.
What are customers usually asking about refunds?
Describe common complaint themes.
What tone do customer support responses usually have?
How do customer service representatives typically respond to cancellation requests?
Summarize how agents respond to complaint intents.
```

Follow-up questions:

```text
Show me 3 more.
Show me another 5 examples.
What about refunds?
What is the total count of the last two?
```

Profile questions:

```text
What do you remember about me?
What do you know about my saved profile?
```

Out-of-scope questions:

```text
Who won the World Cup?
Who is the president of France?
What's the best CRM software for handling complaints?
Write me a poem about customer service.
```

Expected out-of-scope behavior:

```text
I can only answer questions about the Bitext Customer Service dataset or this conversation's saved profile. I can help with counts, filters, examples, summaries, and comparisons inside the dataset.
```

## Memory Design

The project has two memory layers.

### User Profile Memory

Persistent user profile memory is stored under:

```text
.user_profiles/<user_id>/context.md
```

It stores durable facts and preferences only, not a transcript.

Example:

```md
# User Profile

- User ID: max
- User prefers file-by-file implementation review.
- User likes visible reasoning traces.
```

The graph loads this profile at the start of each query and updates it at the end only when a durable observation is detected.

### Conversation / Follow-up State

The graph state stores:

- messages
- route
- route reason
- tool trace
- recent structured results
- final answer

LangGraph checkpoints are stored under:

```text
.checkpoints/checkpoint.sqlite
```

The graph uses the CLI/Streamlit `session_id` as the LangGraph `thread_id`, so the same session can restore prior conversation context and recent structured results across app restarts.

Checkpoint persistence requires:

```text
langgraph-checkpoint-sqlite
```

Recent structured results support follow-up questions such as:

```text
Show me 3 more.
What about refunds?
What is the total count of the last two?
```

For follow-up example pagination, the state stores semantic filters and offsets instead of passing large row-id lists through the LLM context.

## LangGraph Graph Design

Graph flow:

```text
START
  ↓
load_user_profile_node
  ↓
router_node
  ↓
 ┌────────────────────────────┬────────────────────────┐
 │ structured/unstructured     │ out_of_scope            │
 ↓                            ↓
data_agent_loop_node           refusal_node
  ↓                            ↓
profile_update_node            profile_update_node
  ↓                            ↓
END                          END
```

The router classifies each query into:

```text
structured
unstructured
out_of_scope
```

Both `structured` and `unstructured` dataset queries go to the same graph-owned data-agent loop node.

The data-agent node receives compact graph context:

- route
- route reason
- hidden or loaded user profile context, depending on query type
- recent structured results used for follow-up questions
- current-turn tool trace
- reviewer feedback from the previous loop step, when available

The graph-owned loop runs this cycle:

```text
planner LLM
  ↓
tool executor
  ↓
observation reviewer LLM
  ├── answered/cannot_answer → final answer
  └── needs_more → planner LLM
```

The planner chooses the next tool or decides that existing observations are enough for a final answer. The executor safely calls exactly one selected tool. The reviewer checks whether the latest observations answer the exact user question before the graph allows a final answer.

The data-agent loop can execute:

```text
get_dataset_schema
resolve_filter_value
count_rows
sample_examples
group_counts
summarize_rows
read_user_profile
```

## Evidence and Tool-Use Rules

### Filter Resolution

Before calling an analysis tool with a non-null `category` or `intent` value that comes from the current user query, the agent should call:

```text
resolve_filter_value
```

The analysis tool may then use the recommended semantic filter if resolver confidence is `medium` or `high`.

Examples:

```text
How many refund requests did we get?
Show me 5 examples of the SHIPPING category.
Summarize the FEEDBACK category.
How do agents respond to cancellation requests?
```

### Discovery

For discovery questions, the agent should use unfiltered `group_counts`, not `get_dataset_schema.sample_values`.

Examples:

```text
What categories exist in the dataset?
What intents exist in the dataset?
Show the distribution of categories.
Show the distribution of intents.
```

### Scoped Distributions

Scoped distribution questions require grouped counts with the requested scope applied.

Category → intent examples:

```text
What is the distribution of intents in the ACCOUNT category?
Break down ACCOUNT by intent.
Intent breakdown for ACCOUNT.
What intents appear under ACCOUNT?
Which intents occur inside ACCOUNT?
```

Expected final scoped tool call:

```text
group_counts(group_by="intent", category="ACCOUNT", top_k=20)
```

Intent → category examples:

```text
What is the distribution of categories in the track_refund intent?
Break down track_refund by category.
Category breakdown for track_refund.
What categories appear under track_refund?
Which categories occur inside check_refund_status?
```

Expected final scoped tool call:

```text
group_counts(group_by="category", intent="track_refund", top_k=20)
```

A global grouped call is invalid evidence for a scoped distribution.

## Graph-Owned Agent Loop Note

The project uses a custom LangGraph loop instead of delegating the full ReAct cycle to LangChain's standard agent runtime.

The goal is to keep the system agentic while making evidence checks explicit. The LLM still chooses tools and reviews observations, but the graph controls the loop boundaries:

- route before any dataset tool use
- execute one planned tool at a time
- review each observation for answer readiness
- continue when the observation is incomplete
- produce final answers only from reviewed observations
- preserve checkpointed follow-up state
- keep deterministic handling for high-risk example-pagination follow-ups
- build visible reasoning traces directly while the graph runs

This design prevents weak first observations, such as schema sample values or global distributions, from being treated as complete evidence for questions that require full grouped values or scoped grouped values.

## Debugging with LangGraph Studio

LangGraph Studio can be used to inspect:

- graph nodes
- route decisions
- state transitions
- tool traces
- reviewer decisions
- final answers
- checkpoint behavior
- agent fallback behavior

The current graph is built by:

```python
from app.graph import build_graph

graph = build_graph()
```

## Running Tests

Run the full suite:

```bash
pytest
```

Run focused suites:

```bash
pytest tests/test_graph.py -q
pytest tests/test_mcp_server.py -q
pytest tests/test_tools.py -q
```

The test suite covers:

- config and environment behavior
- data loader normalization
- stable row IDs
- tool filtering/counting/sampling/grouping/summarization
- tool input schema boundaries
- MCP wrapper validation
- LLM router behavior with mocked model calls
- persistent profile file behavior
- graph routing
- planner/reviewer loop behavior
- deterministic follow-ups
- scoped distribution guardrails
- out-of-scope refusal
- profile updates
- checkpoint config
- graceful fallback behavior
