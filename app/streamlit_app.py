from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_TEXT = str(PROJECT_ROOT)

if PROJECT_ROOT_TEXT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_TEXT)

from app.config import settings # noqa: E402
from app.graph import invoke_agent # noqa: E402
from app.logging_utils import format_reasoning_trace # noqa: E402

SESSION_ID_INPUT_KEY = "agent_session_id_input"


def get_chat_messages_key(session_id: str) -> str:
    """Return the Streamlit session-state key for visible chat messages."""
    normalized_session_id = session_id.strip() or settings.default_session_id
    return f"chat_messages::{normalized_session_id}"


def get_latest_trace_key(session_id: str) -> str:
    """Return the Streamlit session-state key for the latest reasoning trace."""
    normalized_session_id = session_id.strip() or settings.default_session_id
    return f"latest_trace::{normalized_session_id}"


def initialize_chat_state(chat_messages_key: str, latest_trace_key: str) -> None:
    """Initialize Streamlit session state for visible chat and latest trace."""
    if chat_messages_key not in st.session_state:
        st.session_state[chat_messages_key] = []

    if latest_trace_key not in st.session_state:
        st.session_state[latest_trace_key] = ""


def clear_visible_chat(chat_messages_key: str, latest_trace_key: str) -> None:
    """Clear only Streamlit-visible chat/trace state for the current session."""
    st.session_state[chat_messages_key] = []
    st.session_state[latest_trace_key] = ""


def make_fresh_session_id() -> str:
    """Return a new UI session ID that maps to a fresh graph checkpoint thread."""
    return f"{settings.default_session_id}_{uuid.uuid4().hex[:8]}"


def start_fresh_session(chat_messages_key: str, latest_trace_key: str) -> None:
    """Switch to a new session ID so hidden checkpointed graph state is not reused."""
    clear_visible_chat(chat_messages_key, latest_trace_key)
    st.session_state[SESSION_ID_INPUT_KEY] = make_fresh_session_id()


def render_sidebar() -> tuple[str, str, int, str, str]:
    """Render sidebar controls and return session/user/max-iteration settings."""
    st.sidebar.title("Agent Settings")

    if SESSION_ID_INPUT_KEY not in st.session_state:
        st.session_state[SESSION_ID_INPUT_KEY] = settings.default_session_id

    session_id = st.sidebar.text_input(
        "Session ID",
        key=SESSION_ID_INPUT_KEY,
        help="Conversation session identifier.",
    )
    chat_messages_key = get_chat_messages_key(session_id)
    latest_trace_key = get_latest_trace_key(session_id)

    user_id = st.sidebar.text_input(
        "User ID",
        value=settings.default_user_id,
        help="Persistent user profile identifier.",
    )

    max_iterations = st.sidebar.number_input(
        "Max iterations",
        min_value=settings.min_iterations,
        max_value=settings.max_allowed_iterations,
        value=settings.max_iterations,
        step=1,
        help="Maximum reasoning/tool-use iterations per question.",
    )

    st.sidebar.button(
        "Clear visible chat only",
        help=(
            "Clears only the messages shown in this browser tab. "
            "The graph checkpoint for this Session ID is preserved."
        ),
        on_click=clear_visible_chat,
        args=(chat_messages_key, latest_trace_key),
    )
    st.sidebar.button(
        "Start fresh session",
        help=(
            "Creates a new Session ID so previous graph checkpoint context "
            "will not affect follow-up questions."
        ),
        on_click=start_fresh_session,
        args=(chat_messages_key, latest_trace_key),
    )

    return session_id, user_id, int(max_iterations), chat_messages_key, latest_trace_key


def render_existing_messages(chat_messages_key: str) -> None:
    """Render messages currently stored in visible Streamlit chat state."""
    for message in st.session_state[chat_messages_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def render_trace_panel(trace: str) -> None:
    """Render the latest reasoning trace in a dedicated side panel."""
    st.subheader("Reasoning trace")
    st.caption("Shows only the latest request trace.")

    if trace:
        st.text_area(
            "Latest reasoning trace",
            value=trace,
            height=650,
            label_visibility="collapsed",
            key=f"latest_reasoning_trace_text::{len(trace)}::{hash(trace)}",
        )
    else:
        st.info(
            "Ask a question to see router, tool, observation, and reviewer steps."
        )


def append_user_message(chat_messages_key: str, content: str) -> None:
    st.session_state[chat_messages_key].append(
        {
            "role": "user",
            "content": content,
        }
    )


def append_assistant_message(chat_messages_key: str, content: str) -> None:
    st.session_state[chat_messages_key].append(
        {
            "role": "assistant",
            "content": content,
        }
    )


def main() -> None:

    st.set_page_config(
        page_title="Bitext Data Analyst Agent",
        page_icon="📊",
        layout="wide",
    )

    st.title("Bitext Customer Service Data Analyst Agent")
    st.caption(
        "Ask structured or qualitative questions about the Bitext customer service dataset."
    )

    (
        session_id,
        user_id,
        max_iterations,
        chat_messages_key,
        latest_trace_key,
    ) = render_sidebar()
    initialize_chat_state(chat_messages_key, latest_trace_key)

    chat_column, trace_column = st.columns([0.62, 0.38], gap="large")

    with chat_column:
        render_existing_messages(chat_messages_key)

    with trace_column:
        trace_placeholder = st.empty()
        with trace_placeholder.container():
            render_trace_panel(st.session_state[latest_trace_key])

    query = st.chat_input("Ask about the dataset...")

    if not query:
        return

    append_user_message(chat_messages_key, query)

    st.session_state[latest_trace_key] = ""
    with trace_column:
        trace_placeholder.empty()
        with trace_placeholder.container():
            st.subheader("Reasoning trace")
            st.caption("Shows only the latest request trace.")
            st.info("Running agent... trace will appear here after this request completes.")

    with chat_column:
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    result = invoke_agent(
                        query=query,
                        session_id=session_id,
                        user_id=user_id,
                        max_iterations=max_iterations,
                    )

                    trace = format_reasoning_trace(
                        route=result.get("route"),
                        route_reason=result.get("route_reason"),
                        tool_trace=result.get("tool_trace", []),
                    )

                    final_answer = result.get("final_answer") or (
                        "I could not produce a final answer."
                    )

                except RuntimeError as exc:
                    trace = ""
                    final_answer = f"Error: {exc}"

            st.markdown(final_answer)

    st.session_state[latest_trace_key] = trace

    with trace_column:
        trace_placeholder.empty()
        with trace_placeholder.container():
            render_trace_panel(trace)

    append_assistant_message(chat_messages_key, final_answer)


if __name__ == "__main__":
    main()