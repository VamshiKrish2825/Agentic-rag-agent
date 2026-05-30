"""
app.py
------
Gradio chat interface for the Agentic RAG pipeline.

Run:
    python app.py

Then open http://127.0.0.1:7860 in your browser.
"""

from __future__ import annotations

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from src.rag_agent import run_agent  # noqa: E402

# ---------------------------------------------------------------------------
# State: conversation history is kept in a Gradio State object so each
# browser session has its own independent history.
# ---------------------------------------------------------------------------

def chat(
    message: str,
    history: list[list[str]] | None,
    session_history: list[dict] | None,
):
    """
    Gradio chat handler.

    Args:
        message:        Latest user message.
        history:        Gradio's [[user, assistant], ...] display list.
        session_history: Internal [{role, content}, ...] list for the agent.

    Returns:
        Tuple of ("", updated_history, updated_session_history).
        The empty string clears the input box.
    """
    history = history or []
    agent_history = list(session_history or [])
    message = (message or "").strip()

    if not message:
        return "", history, agent_history

    try:
        answer = run_agent(query=message, history=agent_history)
    except Exception as exc:
        answer = f"Error: {exc}"

    agent_history.append({"role": "user", "content": message})
    agent_history.append({"role": "assistant", "content": answer})
    history.append([message, answer])

    return "", history, agent_history


def clear_session():
    """Reset the chat and conversation memory."""
    return [], [], ""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="Agentic RAG",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:
    gr.Markdown(
        """
        # Agentic RAG - LangGraph + FAISS
        Ask questions about the documents in `data/sample_docs/`.
        The agent will **rewrite your query**, **retrieve relevant chunks**,
        **grade their relevance**, and **generate a grounded answer**.
        """
    )

    session_history = gr.State([])  # internal [{role, content}] history

    chatbot = gr.Chatbot(height=450, label="Conversation")
    msg_box = gr.Textbox(
        placeholder="Ask a question about your documents...",
        label="Your question",
        show_label=False,
        lines=2,
    )

    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear")

    # Wire up events
    submit_btn.click(
        fn=chat,
        inputs=[msg_box, chatbot, session_history],
        outputs=[msg_box, chatbot, session_history],
    )
    msg_box.submit(
        fn=chat,
        inputs=[msg_box, chatbot, session_history],
        outputs=[msg_box, chatbot, session_history],
    )
    clear_btn.click(
        fn=clear_session,
        outputs=[chatbot, session_history, msg_box],
    )

    gr.Markdown(
        """
        ---
        Built by Vamshi - [GitHub](https://github.com/VamshiKrish2825)
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
