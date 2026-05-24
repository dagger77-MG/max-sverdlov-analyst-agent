from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_TEXT = str(PROJECT_ROOT)

if PROJECT_ROOT_TEXT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_TEXT)


import streamlit as st # noqa: E402

from app.config import settings # noqa: E402
from app.graph import invoke_agent # noqa: E402
from app.logging_utils import format_reasoning_trace # noqa: E402


def get_chat_messages_key(session_id: str) -> str:
    """Return the Streamlit session-state key for visible chat messages."""
    normalized_session_id = session_id.strip() or settings.default_session_id
    return f"chat_messages::{normalized_session_id}"

def initialize_chat_state(chat_messages_key: str) -> None:
    """Initialize Streamlit session state for visible chat messages."""
    if chat_messages_key not in st.session_state:
        st.session_state[chat_messages_key] = []


def render_sidebar() -> tuple[str, str, int, str]:
    """Render sidebar controls and return session/user/max-iteration settings."""
    st.sidebar.title("Agent Settings")

    session_id = st.sidebar.text_input(
        "Session ID",
        value=settings.default_session_id,
        help="Conversation session identifier.",
    )
    chat_messages_key = get_chat_messages_key(session_id)

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

    if st.sidebar.button("Clear visible chat"):
        st.session_state[chat_messages_key] = []
        st.rerun()

    return session_id, user_id, int(max_iterations), chat_messages_key


def render_existing_messages(chat_messages_key: str) -> None:
    """Render messages currently stored in visible Streamlit chat state."""
    for message in st.session_state[chat_messages_key]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            trace = message.get("trace")
            if trace:
                with st.expander("Reasoning trace"):
                    st.text(trace)


def append_user_message(chat_messages_key: str, content: str) -> None:
    st.session_state[chat_messages_key].append(
        {
            "role": "user",
            "content": content,
        }
    )


def append_assistant_message(chat_messages_key: str, content: str, trace: str) -> None:
    st.session_state[chat_messages_key].append(
        {
            "role": "assistant",
            "content": content,
            "trace": trace,
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

    session_id, user_id, max_iterations, chat_messages_key = render_sidebar()
    initialize_chat_state(chat_messages_key)

    render_existing_messages(chat_messages_key)

    query = st.chat_input("Ask about the dataset...")

    if not query:
        return

    append_user_message(chat_messages_key, query)

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

        if trace:
            with st.expander("Reasoning trace"):
                st.text(trace)

    append_assistant_message(chat_messages_key, final_answer, trace)


if __name__ == "__main__":
    main()