from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

import sqlite3

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.config import settings
from app.logging_utils import summarize_row_ids
from app.memory import read_user_profile_impl, update_user_profile_impl
from app.prompts import (
    DATA_AGENT_SYSTEM_PROMPT,
    OUT_OF_SCOPE_REFUSAL,
    PROFILE_UPDATE_SYSTEM_PROMPT,
)
from app.router import RouteDecision, route_query_with_reason
from app.state import AgentState, AnalysisResult, ToolTraceItem
from app.tools import (
    count_rows_impl,
    filter_rows_impl,
    get_dataset_schema_impl,
    group_counts_impl,
    sample_examples_impl,
    summarize_rows_impl,
)


ToolName = Literal[
    "get_dataset_schema",
    "filter_rows",
    "count_rows",
    "sample_examples",
    "group_counts",
    "summarize_rows",
    "read_user_profile",
    "final_answer",
]


class AgentActionDecision(BaseModel):
    """One ReAct-style action selected by the data agent."""

    thought: str = Field(
        description="Brief reasoning about what to do next."
    )
    tool_name: ToolName = Field(
        description="Tool to call next, or final_answer when ready to answer."
    )
    tool_input: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-compatible input for the selected tool.",
    )
    final_answer: str | None = Field(
        default=None,
        description="Final user-facing answer when tool_name is final_answer.",
    )


class ProfileObservationDecision(BaseModel):
    """Decision about whether a durable profile observation should be saved."""

    observation: str = Field(
        default="",
        description="Concise durable observation to save, or empty string.",
    )


@lru_cache(maxsize=1)
def get_agent_llm():
    """Return a cached OpenAI-compatible chat model for the data agent."""
    if not settings.nebius_api_key:
        raise RuntimeError(
            "NEBIUS_API_KEY is missing. Add it to your environment or .env file "
            "before using the graph agent."
        )

    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The graph agent requires 'langchain-openai'. "
            "Install project dependencies before running the agent."
        ) from exc

    return ChatOpenAI(
        model=settings.agent_model,
        api_key=settings.nebius_api_key,
        base_url=settings.nebius_base_url,
        temperature=0,
        max_tokens=1024,
        extra_body={
            "enable_thinking": False,
            "thinking_budget": 0,
        },
    )


@lru_cache(maxsize=1)
def get_structured_action_llm():
    """Return a cached agent model configured for structured action decisions."""
    return get_agent_llm().with_structured_output(AgentActionDecision)


@lru_cache(maxsize=1)
def get_structured_profile_llm():
    """Return a cached model configured for profile-update decisions."""
    return get_agent_llm().with_structured_output(ProfileObservationDecision)


def create_initial_state(
    query: str,
    session_id: str,
    user_id: str,
    max_iterations: int | None = None,
) -> AgentState:
    """Create the initial graph state for one user query."""
    return AgentState(
        messages=[HumanMessage(content=query)],
        session_id=session_id,
        user_id=user_id,
        route=None,
        route_reason=None,
        tool_trace=[],
        last_structured_results=[],
        user_profile="",
        iteration_count=0,
        max_iterations=settings.normalize_max_iterations(max_iterations),
        final_answer=None,
    )


def _latest_user_message(messages: list[BaseMessage]) -> str:
    """Return the latest human message content from graph state."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)

    return ""


def _append_trace(
    state: AgentState,
    tool_name: str,
    tool_input: dict[str, Any],
    observation: str,
) -> None:
    """Append one tool/observation step to state."""
    state["tool_trace"].append(
        ToolTraceItem(
            tool_name=tool_name,
            tool_input=tool_input,
            observation=observation,
        )
    )


def _append_structured_result(
    state: AgentState,
    result: AnalysisResult,
    max_results: int = 10,
) -> None:
    """Store compact structured results for follow-up questions."""
    state["last_structured_results"].append(result)
    state["last_structured_results"] = state["last_structured_results"][-max_results:]


def _tool_trace_for_prompt(tool_trace: list[ToolTraceItem]) -> str:
    """Format existing trace for the next LLM action decision."""
    if not tool_trace:
        return "No tools have been called yet."

    lines: list[str] = []
    for index, trace_item in enumerate(tool_trace, start=1):
        lines.append(
            f"{index}. Tool: {trace_item['tool_name']}\n"
            f"   Input: {trace_item['tool_input']}\n"
            f"   Observation: {trace_item['observation']}"
        )

    return "\n".join(lines)


def _structured_results_for_prompt(results: list[AnalysisResult]) -> str:
    """Format recent structured results for follow-up resolution."""
    if not results:
        return "No recent structured results."

    lines: list[str] = []
    for index, result in enumerate(results[-5:], start=1):
        row_ids = result.get("row_ids")
        lines.append(
            f"{index}. label={result['label']}; "
            f"value={result['value']}; "
            f"query_type={result['query_type']}; "
            f"row_ids={summarize_row_ids(row_ids)}"
        )

    return "\n".join(lines)


def _route_specific_instructions(route: str | None) -> str:
    """Return explicit behavior instructions for the selected route."""
    if route == "structured":
        return """Route-specific instructions for STRUCTURED queries:
- The user is asking for exact dataset analysis.
- Prefer deterministic tools such as filter_rows, count_rows, sample_examples, and group_counts.
- Use filter_rows first when the user mentions a category, intent, topic, or text condition.
- Prefer exact category or intent filters over text_query when the user wording clearly maps to a known dataset category or intent.
- Use text_query only when the user asks about a topic that is not clearly represented by a known category or intent.
- For filtered count questions, use filter_rows.match_count directly in final_answer.
- Use count_rows only for whole-dataset counts or complete row_id lists.
- Use sample_examples when the user asks for examples, samples, or "show me more".
- Use group_counts when the user asks for distributions, most common categories, or counts by category/intent.
- Do not use summarize_rows unless the user explicitly asks for a summary, themes, tone, patterns, or qualitative interpretation.
- Known category mappings:
  - refund, refunds, refund requests, reimbursement, reimbursement cases, money back, guarantee -> category="REFUND"
  - feedback, product feedback, customer feedback -> category="FEEDBACK"
  - complaint, complaints -> category="COMPLAINT"
  - contact, contact support, customer service contact -> category="CONTACT"
- For example/sample requests, do not stop after filter_rows.
- If the user asks "Show me N examples from CATEGORY/INTENT/TOPIC":
  1. Call filter_rows with the category, intent, or text condition and no limit.
  2. Call sample_examples with the returned row_ids, n=N, and offset=0.
  3. Answer using the sampled examples.
- Do not use filter_rows(limit=N) as a substitute for sample_examples(n=N).
- Never repeat the same filter_rows call if it already returned matching row IDs.
- Final answers should include exact values from tool observations.
"""

    if route == "unstructured":
        return """Route-specific instructions for UNSTRUCTURED queries:
- The user is asking for qualitative analysis over dataset rows.
- First identify the relevant row subset with filter_rows when the user mentions a category, intent, topic, or text condition.
- Then use summarize_rows on the selected row IDs.
- If the user asks about the whole dataset qualitatively, you may use get_dataset_schema first, then filter_rows or summarize_rows as needed.
- Do not answer from general knowledge.
- Final answers should describe only patterns supported by the rows passed to summarize_rows.
- Mention the number of rows used when helpful.
"""

    return """Route-specific instructions:
- If the route is unclear, inspect the dataset with get_dataset_schema or use the safest relevant dataset tool.
- Do not answer from general knowledge.
"""


def _build_action_messages(state: AgentState) -> list[BaseMessage]:
    """Build messages for the ReAct-style action selector."""
    user_query = _latest_user_message(state["messages"])

    context = f"""Current route: {state["route"]}
Route reason: {state["route_reason"]}

User profile:
{state["user_profile"]}

Recent structured results:
{_structured_results_for_prompt(state["last_structured_results"])}

Tool trace so far:
{_tool_trace_for_prompt(state["tool_trace"])}

{_route_specific_instructions(state["route"])}

Available tools:
- get_dataset_schema(include_sample_values: bool = True)
  Inspect dataset columns, row count, and sample values.
- filter_rows(category: str | None = None, intent: str | None = None, text_query: str | None = None, limit: int | None = None)
  Find matching row IDs for a category, intent, or text query.
  This tool does not show examples.
  Prefer category or intent when the user phrase maps to a known dataset category/intent.
  Do not use limit to satisfy "show N examples".
  For example requests, call filter_rows without limit, then call sample_examples with n=N.
- count_rows(row_ids: list[int] | None = None)
  Count all rows or a complete row_id subset. Do not use count_rows with previewed row IDs from an observation such as "[1, 2, 3...]".
  For filtered count questions, prefer filter_rows.match_count and then final_answer.
- sample_examples(row_ids: list[int] | None = None, n: int = 3, offset: int = 0)
  Show actual example rows.
  Use this whenever the user asks for examples, samples, rows, cases, or "show me N".
  Usually call filter_rows first to get row_ids, then call sample_examples(row_ids=..., n=N, offset=0).
- group_counts(group_by: "category" | "intent", row_ids: list[int] | None = None, top_k: int = 20)
  Count rows grouped by category or intent.
- summarize_rows(row_ids: list[int], focus: str, max_examples: int = 100)
  Summarize selected rows for qualitative questions about themes, tone, or patterns.
- read_user_profile(user_id: str)
  Read the saved durable user profile.
- final_answer

Important:
- Use tools before answering dataset questions.
- For "show more" follow-ups, reuse previous row IDs and the previous next offset if available in the trace.
- For "total of the last two", use recent stored count results if available.
- After every tool observation, compare the observation with the original user question:
  - If the observation already contains the information needed to answer the question, choose final_answer.
  - If the observation is only an intermediate result, choose the next missing tool.
  - Do not call another tool just because more tools are available.
  - Do not repeat a tool call that produced the same answer-ready observation.
- If a tool call already returned useful results, do not repeat the same tool call with the same input.
- After filter_rows returns matching row IDs for an example request, the next action should usually be sample_examples.
- When enough evidence is available, choose final_answer.
"""

    return [
        SystemMessage(
            content="/no_think\n"
                    + DATA_AGENT_SYSTEM_PROMPT
        ),
        SystemMessage(content=context),
        HumanMessage(content=user_query),
    ]


def _safe_int_list(value: Any) -> list[int] | None:
    """Convert a JSON-like row_ids value into list[int] or None."""
    if value is None:
        return None

    if not isinstance(value, list):
        return None

    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue

    return result


def _normalized_tool_input(tool_input: dict[str, Any]) -> dict[str, Any]:
    """Normalize tool input for repeated-call detection."""
    normalized: dict[str, Any] = {}

    for key, value in tool_input.items():
        if value is None:
            continue

        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                normalized[key] = stripped
            continue

        normalized[key] = value

    return normalized


def _is_repeated_tool_call(
    state: AgentState,
    tool_name: ToolName,
    tool_input: dict[str, Any],
) -> bool:
    """Return True when the exact same tool call already ran this turn."""
    normalized_input = _normalized_tool_input(tool_input)

    for trace_item in state["tool_trace"]:
        if trace_item["tool_name"] != tool_name:
            continue

        if _normalized_tool_input(trace_item["tool_input"]) == normalized_input:
            return True

    return False


def _answer_from_existing_trace(state: AgentState) -> str:
    """Ask the model to answer from existing observations after a repeated call."""
    return (
        "The needed tool result is already available in the trace. "
        "Please answer using the latest relevant observation instead of repeating "
        "the same tool call."
    )


def _execute_tool(
    state: AgentState,
    tool_name: ToolName,
    tool_input: dict[str, Any],
) -> str:
    """Execute one selected tool and update trace/state."""
    if tool_name == "get_dataset_schema":
        result = get_dataset_schema_impl(
            include_sample_values=bool(tool_input.get("include_sample_values", True))
        )
        observation = (
            f"Dataset has {result.row_count} rows and columns: "
            f"{', '.join(result.columns)}."
        )
        _append_trace(state, tool_name, tool_input, observation)
        return observation

    if tool_name == "filter_rows":
        result = filter_rows_impl(
            category=tool_input.get("category"),
            intent=tool_input.get("intent"),
            text_query=tool_input.get("text_query"),
            limit=tool_input.get("limit"),
        )
        observation = (
            f"Found {result.match_count} matching rows. "
            f"match_count={result.match_count}. "
            f"Returned row_ids for the matching subset: {summarize_row_ids(result.row_ids)}."

        )

        _append_trace(state, tool_name, tool_input, observation)
        _append_structured_result(
            state,
            AnalysisResult(
                label=str(result.applied_filters),
                value=result.match_count,
                query_type="filter",
                row_ids=result.row_ids,
            ),
        )
        return observation

    if tool_name == "count_rows":
        row_ids = _safe_int_list(tool_input.get("row_ids"))
        result = count_rows_impl(row_ids=row_ids)
        observation = f"Count = {result.count}."
        _append_trace(
            state,
            tool_name,
            {"row_ids": row_ids},
            observation,
        )
        _append_structured_result(
            state,
            AnalysisResult(
                label="count_rows",
                value=result.count,
                query_type="count",
                row_ids=row_ids,
            ),
        )
        return observation

    if tool_name == "sample_examples":
        row_ids = _safe_int_list(tool_input.get("row_ids"))
        n = int(tool_input.get("n", 3))
        offset = int(tool_input.get("offset", 0))
        result = sample_examples_impl(row_ids=row_ids, n=n, offset=offset)

        example_lines = [
            f"{example.row_id}: {example.instruction}"
            for example in result.examples
        ]
        observation = (
            f"Returned {len(result.examples)} examples. "
            f"Next offset = {result.next_offset}. "
            + ("\n" + "\n".join(example_lines) if example_lines else "")
        )
        _append_trace(
            state,
            tool_name,
            {"row_ids": row_ids, "n": n, "offset": offset},
            observation,
        )
        _append_structured_result(
            state,
            AnalysisResult(
                label="sample_examples",
                value=result.next_offset,
                query_type="sample",
                row_ids=row_ids,
            ),
        )
        return observation

    if tool_name == "group_counts":
        row_ids = _safe_int_list(tool_input.get("row_ids"))
        group_by = str(tool_input.get("group_by", "category"))
        top_k = int(tool_input.get("top_k", 20))

        if group_by not in {"category", "intent"}:
            group_by = "category"

        result = group_counts_impl(
            group_by=group_by,  # type: ignore[arg-type]
            row_ids=row_ids,
            top_k=top_k,
        )
        observation = "; ".join(
            f"{row.label}: {row.count}" for row in result.counts
        )
        _append_trace(
            state,
            tool_name,
            {"group_by": group_by, "row_ids": row_ids, "top_k": top_k},
            observation,
        )
        return observation

    if tool_name == "summarize_rows":
        row_ids = _safe_int_list(tool_input.get("row_ids")) or []
        focus = str(tool_input.get("focus", "dataset summary"))
        max_examples = int(tool_input.get("max_examples", 100))
        result = summarize_rows_impl(
            row_ids=row_ids,
            focus=focus,
            max_examples=max_examples,
        )
        observation = result.summary
        _append_trace(
            state,
            tool_name,
            {
                "row_ids": row_ids,
                "focus": focus,
                "max_examples": max_examples,
            },
            observation,
        )
        return observation

    if tool_name == "read_user_profile":
        user_id = str(tool_input.get("user_id") or state["user_id"])
        result = read_user_profile_impl(user_id=user_id)
        observation = result.profile
        _append_trace(state, tool_name, {"user_id": user_id}, observation)
        return observation

    raise ValueError(f"Unsupported tool name: {tool_name}")


def load_user_profile_node(state: AgentState) -> dict[str, Any]:
    """Load persistent profile memory into graph state."""
    profile = read_user_profile_impl(state["user_id"])
    return {
        "user_profile": profile.profile,
    }


def router_node(state: AgentState) -> dict[str, Any]:
    """Classify the latest user query before tool selection."""
    user_query = _latest_user_message(state["messages"])
    decision: RouteDecision = route_query_with_reason(user_query)

    return {
        "route": decision.route,
        "route_reason": decision.reason,
    }


def refusal_node(state: AgentState) -> dict[str, Any]:
    """Return a scoped refusal for out-of-scope queries."""
    return {
        "final_answer": OUT_OF_SCOPE_REFUSAL,
        "messages": [AIMessage(content=OUT_OF_SCOPE_REFUSAL)],
    }


def react_data_agent_node(state: AgentState) -> dict[str, Any]:
    """Run a ReAct-style tool-use loop for dataset questions."""
    structured_llm = get_structured_action_llm()

    while state["iteration_count"] < state["max_iterations"]:
        state["iteration_count"] += 1

        decision = structured_llm.invoke(_build_action_messages(state))
        if not isinstance(decision, AgentActionDecision):
            decision = AgentActionDecision.model_validate(decision)

        if decision.tool_name == "final_answer":
            final_answer = decision.final_answer or (
                "I completed the analysis, but no final answer was provided."
            )
            state["final_answer"] = final_answer
            return {
                "tool_trace": state["tool_trace"],
                "last_structured_results": state["last_structured_results"],
                "iteration_count": state["iteration_count"],
                "final_answer": final_answer,
                "messages": [AIMessage(content=final_answer)],
            }

        if _is_repeated_tool_call(
            state=state,
            tool_name=decision.tool_name,
            tool_input=decision.tool_input,
        ):
            final_answer = _answer_from_existing_trace(state)
            state["final_answer"] = final_answer
            return {
                "tool_trace": state["tool_trace"],
                "last_structured_results": state["last_structured_results"],
                "iteration_count": state["iteration_count"],
                "final_answer": final_answer,
                "messages": [AIMessage(content=final_answer)],
            }

        _execute_tool(
            state=state,
            tool_name=decision.tool_name,
            tool_input=decision.tool_input,
        )

    fallback = (
        "I could not complete the analysis within the allowed number of "
        "reasoning steps. Please try asking a more specific dataset question."
    )
    state["final_answer"] = fallback
    return {
        "tool_trace": state["tool_trace"],
        "last_structured_results": state["last_structured_results"],
        "iteration_count": state["iteration_count"],
        "final_answer": fallback,
        "messages": [AIMessage(content=fallback)],
    }


def profile_update_node(state: AgentState) -> dict[str, Any]:
    """Update the persistent profile only when a durable fact is detected."""
    user_query = _latest_user_message(state["messages"])

    if not user_query.strip():
        return {}

    try:
        profile_llm = get_structured_profile_llm()
        decision = profile_llm.invoke(
            [
                SystemMessage(content=PROFILE_UPDATE_SYSTEM_PROMPT),
                HumanMessage(content=user_query),
            ]
        )
        if not isinstance(decision, ProfileObservationDecision):
            decision = ProfileObservationDecision.model_validate(decision)
    except Exception:
        return {}

    observation = decision.observation.strip()
    if not observation:
        return {}

    updated_profile = update_user_profile_impl(
        user_id=state["user_id"],
        new_observation=observation,
    )
    return {
        "user_profile": updated_profile.profile,
    }


def route_after_router(state: AgentState) -> str:
    """Choose the next graph branch after routing."""
    if state["route"] == "out_of_scope":
        return "refusal_node"

    return "react_data_agent_node"


@lru_cache(maxsize=1)
def get_checkpointer():
    """Return a persistent SQLite checkpointer for LangGraph state."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise RuntimeError(
            "Persistent LangGraph checkpoints require the SQLite checkpoint package. "
            "Install 'langgraph-checkpoint-sqlite' with the project dependencies."
        ) from exc

    settings.ensure_runtime_dirs()
    checkpoint_path = settings.checkpoint_dir / "checkpoint.sqlite"
    connection = sqlite3.connect(str(checkpoint_path), check_same_thread=False)

    checkpointer = SqliteSaver(connection)
    checkpointer.setup()

    return checkpointer


@lru_cache(maxsize=1)
def build_graph():
    """Build and compile the LangGraph agent graph."""
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("load_user_profile_node", load_user_profile_node)
    graph_builder.add_node("router_node", router_node)
    graph_builder.add_node("react_data_agent_node", react_data_agent_node)
    graph_builder.add_node("refusal_node", refusal_node)
    graph_builder.add_node("profile_update_node", profile_update_node)

    graph_builder.add_edge(START, "load_user_profile_node")
    graph_builder.add_edge("load_user_profile_node", "router_node")
    graph_builder.add_conditional_edges(
        "router_node",
        route_after_router,
        {
            "react_data_agent_node": "react_data_agent_node",
            "refusal_node": "refusal_node",
        },
    )
    graph_builder.add_edge("react_data_agent_node", "profile_update_node")
    graph_builder.add_edge("refusal_node", "profile_update_node")
    graph_builder.add_edge("profile_update_node", END)

    return graph_builder.compile(checkpointer=get_checkpointer())


def _build_graph_config(
    session_id: str,
    user_id: str,
    max_iterations: int,
) -> dict[str, Any]:
    """Build LangGraph invocation config for checkpointed sessions."""
    return {
        "configurable": {
            "thread_id": session_id,
            "user_id": user_id,
        },
        "recursion_limit": max_iterations + 5,
    }


def _create_invocation_state(
    graph,
    query: str,
    session_id: str,
    user_id: str,
    max_iterations: int,
    config: dict[str, Any],
) -> Any:
    """Create graph input while preserving checkpointed follow-up context.

    For a new thread, provide a complete initial state. For an existing
    checkpointed thread, provide only the fields that should reset for the
    current turn plus the new user message. This preserves prior messages and
    recent structured results for follow-up questions.
    """

    try:
        checkpoint_state = graph.get_state(config)
    except Exception:
        return create_initial_state(
            query=query,
            session_id=session_id,
            user_id=user_id,
            max_iterations=max_iterations,
        )

    if not checkpoint_state.values:
        return create_initial_state(
            query=query,
            session_id=session_id,
            user_id=user_id,
            max_iterations=max_iterations,
        )

    return {
        "messages": [HumanMessage(content=query)],
        "session_id": session_id,
        "user_id": user_id,
        "route": None,
        "route_reason": None,
        "tool_trace": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "final_answer": None,
    }


def invoke_agent(
    query: str,
    session_id: str | None = None,
    user_id: str | None = None,
    max_iterations: int | None = None,
) -> AgentState:
    """Invoke the compiled graph for one user query."""
    graph = build_graph()
    normalized_session_id = session_id or settings.default_session_id
    normalized_user_id = user_id or settings.default_user_id
    normalized_max_iterations = settings.normalize_max_iterations(max_iterations)

    config = _build_graph_config(
        session_id=normalized_session_id,
        user_id=normalized_user_id,
        max_iterations=normalized_max_iterations,
    )

    invocation_state = _create_invocation_state(
        graph=graph,
        query=query,
        session_id=normalized_session_id,
        user_id=normalized_user_id,
        max_iterations=normalized_max_iterations,
        config=config,
    )

    return graph.invoke(invocation_state, config=config)