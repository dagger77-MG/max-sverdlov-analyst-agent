# Bitext Customer Service Data Analyst Agent

A LangGraph-based educational data analyst agent for the Bitext Customer Service dataset.

The agent can answer structured analytical questions and qualitative dataset questions, while staying scoped to the dataset and the saved user profile.

## Features

- LangGraph ReAct-style agent loop
- LLM-based query router
- Structured dataset tools
- Qualitative row summarization tool
- Persistent user profile memory
- CLI chat interface
- Streamlit chat interface
- FastMCP server exposing dataset tools
- Visible reasoning traces with route decisions, tool calls, inputs, and observations
- Unit tests for loader, tools, router, memory, and graph behavior

## Dataset

Dataset source:

```text
bitext/Bitext-customer-support-llm-chatbot-training-dataset
```

The app loads the dataset through Hugging Face `datasets`, normalizes the DataFrame columns, adds a stable `row_id`, and caches the normalized CSV locally under:

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

The code inspects and normalizes the dataset instead of assuming extra columns.

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
│   └── logging_utils.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_tools.py
│   ├── test_router.py
│   ├── test_memory.py
│   └── test_graph.py
└── .user_profiles/
    └── .gitkeep
```

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

The model names and Nebius base URL are configured in:

```text
app/config.py
```

Current model configuration:

```text
router_model = meta-llama/Meta-Llama-3.1-8B-Instruct
agent_model = meta-llama/Llama-3.3-70B-Instruct
nebius_base_url = https://api.tokenfactory.nebius.com/v1/
```

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
[tool] filter_rows
[input]
{
  "category": "REFUND"
}
[observation] Found 842 matching rows. Returned 842 row IDs [1, 4, 9, 12, 20...]

[tool] count_rows
[input]
{
  "row_ids": {
    "type": "list",
    "count": 842,
    "preview": [1, 4, 9, 12, 20]
  }
}
[observation] Count = 842.

Agent: There are 842 refund-request rows in the dataset.
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
- Clear visible chat button

The main chat area shows:

- User messages
- Agent responses
- Reasoning trace expanders with route and tool steps

The Streamlit app uses the same `invoke_agent()` function as the CLI.

## Running the FastMCP Server

Run:

```bash
python -m app.mcp_server
```

The MCP server exposes these dataset tools:

```text
get_dataset_schema
filter_rows
count_rows
sample_examples
group_counts
summarize_rows
```

## Connecting an MCP Client

Example client:

```python
from __future__ import annotations

import asyncio

from fastmcp import Client


async def main() -> None:
    async with Client("http://localhost:8000/mcp") as client:
        result = await client.call_tool(
            "group_counts",
            {
                "group_by": "category",
                "top_k": 10,
            },
        )
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
```

The exact transport URL may depend on the FastMCP runtime configuration. If needed, check the server startup logs.

## Example Questions

Structured questions:

```text
How many refund requests did we get?
Show me 3 examples from the REFUND category.
How many complaints did we get?
Which intent is most common?
Count rows by category.
```

Unstructured questions:

```text
Summarize the FEEDBACK category.
What are customers usually asking about refunds?
Describe common complaint themes.
What tone do customer support responses usually have?
```

Follow-up questions:

```text
Show me 3 more.
What about refunds?
What is the total count of the last two?
```

Profile question:

```text
What do you remember about me?
```

Out-of-scope question:

```text
Who won the World Cup?
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

Examples:

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
- iteration count
- final answer

Recent structured results support follow-up questions such as:

```text
Show me 3 more.
What about refunds?
What is the total count of the last two?
```

Current limitation: durable LangGraph checkpoint persistence is not yet wired through a SQLite checkpointer. The current implementation carries state inside one graph invocation and stores persistent user profiles on disk.

## LangGraph Graph Design

Graph flow:

```text
START
  ↓
load_user_profile_node
  ↓
router_node
  ↓
 ┌───────────────────────┬────────────────────────┐
 │ structured/unstructured│ out_of_scope            │
 ↓                       ↓
react_data_agent_node     refusal_node
  ↓                       ↓
profile_update_node       profile_update_node
  ↓                       ↓
END                     END
```

The router classifies each query into:

```text
structured
unstructured
out_of_scope
```

Both `structured` and `unstructured` dataset queries go to the same route-aware ReAct agent node.

The ReAct agent receives route-specific instructions:

- Structured route: prefer exact tools such as `filter_rows`, `count_rows`, `sample_examples`, and `group_counts`.
- Unstructured route: identify relevant rows and then use `summarize_rows`.
- Out-of-scope route: skip data tools and return a scoped refusal.

## Tool Reference

### get_dataset_schema

Returns dataset columns, row count, and optional sample values.

### filter_rows

Filters rows by:

- category
- intent
- text query over instruction and response
- optional limit

Returns matching row IDs and total match count.

### count_rows

Counts all dataset rows or a provided subset of row IDs.

### sample_examples

Returns example rows from the full dataset or a filtered subset.

Supports `offset` for follow-up requests like:

```text
Show me 3 more.
```

### group_counts

Groups rows by:

```text
category
intent
```

Returns counts sorted by frequency.

### summarize_rows

Summarizes selected row IDs for qualitative analysis.

Use for:

- summaries
- themes
- patterns
- tone
- customer pain points

### read_user_profile

Reads the persistent profile for a user.

### update_user_profile

Updates the profile with a durable observation.

This is called by the graph profile-update node, not normally by the main data-analysis loop.

## Debugging with LangGraph Studio

LangGraph Studio can be used to inspect:

- graph nodes
- route decisions
- state transitions
- tool traces
- final answers
- max-iteration fallback behavior

The current graph is built by:

```python
from app.graph import build_graph

graph = build_graph()
```

## Running Tests

Run:

```bash
pytest
```

The test suite covers:

- data loader normalization
- stable row IDs
- tool filtering/counting/sampling/grouping/summarization
- LLM router behavior with mocked model calls
- persistent profile file behavior
- graph routing, tool traces, refusal, profile updates, and max-iteration fallback

## Validation Checklist

Before submission, manually validate these cases.

### Structured Count

Question:

```text
How many refund requests did we get?
```

Expected:

- Route: `structured`
- Uses `filter_rows`
- Uses `count_rows`
- Returns exact count

### Structured Examples

Question:

```text
Show me 3 examples from the REFUND category.
```

Expected:

- Route: `structured`
- Uses `filter_rows`
- Uses `sample_examples`
- Shows 3 examples

### Follow-Up Examples

Question:

```text
Show me 3 more.
```

Expected:

- Uses previous row IDs
- Uses previous offset
- Shows the next examples

### Unstructured Summary

Question:

```text
Summarize the FEEDBACK category.
```

Expected:

- Route: `unstructured`
- Uses `filter_rows`
- Uses `summarize_rows`
- Produces a dataset-grounded summary

### User Profile Memory

Question:

```text
What do you remember about me?
```

Expected:

- Reads persistent user profile
- Does not hallucinate unknown facts
- Clearly says if profile is empty

### Out-of-Scope Refusal

Question:

```text
Who won the World Cup?
```

Expected:

- Route: `out_of_scope`
- Refuses politely
- Does not answer from general knowledge

## Known Limitations

- LangGraph SQLite checkpoint persistence is not yet implemented.
- `summarize_rows` currently creates a deterministic summary from row metadata and representative instructions. It does not yet call a separate summarization LLM.
- Follow-up behavior depends on recent structured results stored in graph state. Durable cross-restart follow-up memory requires checkpoint integration.
- The router and agent rely on Nebius model support for structured output through LangChain.
- The exact FastMCP transport URL should be verified from server startup logs.
