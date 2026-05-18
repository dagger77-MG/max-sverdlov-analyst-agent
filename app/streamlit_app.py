from __future__ import annotations

import streamlit as st

from app.config import settings
from app.graph import invoke_agent
from app.logging_utils import format_reasoning_trace


def initialize_chat_state() -> None:
    """Initialize Streamlit session state for visible chat messages."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []


def render_sidebar() -> tuple[str, str, int]:
    """Render sidebar controls and return session/user/max-iteration settings."""
    st.sidebar.title("Agent Settings")

    session_id = st.sidebar.text_input(
        "Session ID",
        value=settings.default_session_id,
        help="Conversation session identifier.",
    )

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
        st.session_state.chat_messages = []
        st.rerun()

    return session_id, user_id, int(max_iterations)


def render_existing_messages() -> None:
    """Render messages currently stored in visible Streamlit chat state."""
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            trace = message.get("trace")
            if trace:
                with st.expander("Reasoning trace"):
                    st.text(trace)


def append_user_message(content: str) -> None:
    st.session_state.chat_messages.append(
        {
            "role": "user",
            "content": content,
        }
    )


def append_assistant_message(content: str, trace: str) -> None:
    st.session_state.chat_messages.append(
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

    initialize_chat_state()

    st.title("Bitext Customer Service Data Analyst Agent")
    st.caption(
        "Ask structured or qualitative questions about the Bitext customer service dataset."
    )

    session_id, user_id, max_iterations = render_sidebar()

    render_existing_messages()

    query = st.chat_input("Ask about the dataset...")

    if not query:
        return

    append_user_message(query)

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

    append_assistant_message(final_answer, trace)


if __name__ == "__main__":
    main()