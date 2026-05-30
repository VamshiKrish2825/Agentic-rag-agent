"""
rag_agent.py
------------
LangGraph-based Agentic RAG pipeline.

Flow:
    User Query
        ->
    query_rewriter  - cleans / expands the query
        ->
    retriever       - fetches top-k chunks from the vector store
        ->
    relevance_check - grades each chunk; routes to generate or rewrite
        ->
    generator       - synthesizes a final answer from accepted chunks
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.prompts import (
    GENERATE_PROMPT,
    QUERY_REWRITE_PROMPT,
    RELEVANCE_PROMPT,
)
from src.retriever import get_retriever
from src.state import RAGState

load_dotenv()

# ---------------------------------------------------------------------------
# LLM setup - swap model or base_url for any OpenAI-compatible endpoint
# ---------------------------------------------------------------------------

MAX_REWRITES = int(os.getenv("MAX_REWRITES", "2"))


def _get_llm() -> ChatOpenAI:
    """Return the configured OpenAI-compatible chat model."""
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment or .env file. "
            "For Ollama, set OPENAI_API_KEY=ollama."
        )

    kwargs: dict[str, Any] = {
        "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0")),
        "api_key": api_key,
    }
    if base_url:
        kwargs["base_url"] = base_url

    return ChatOpenAI(**kwargs)


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def query_rewriter(state: RAGState) -> dict[str, Any]:
    """Rewrite the user query so it is self-contained and retrieval-friendly."""
    llm = _get_llm()
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in state["history"]
    )
    prompt = QUERY_REWRITE_PROMPT.format(
        history=history_text or "No prior conversation.",
        query=state["query"],
    )
    result = llm.invoke(prompt)
    rewritten = result.content.strip()
    return {"rewritten_query": rewritten}


def retriever_node(state: RAGState) -> dict[str, Any]:
    """Retrieve top-k documents for the (rewritten) query."""
    retriever = get_retriever()
    docs: list[Document] = retriever.invoke(state["rewritten_query"])
    return {"documents": docs}


def relevance_check(state: RAGState) -> dict[str, Any]:
    """Grade each retrieved document and keep only relevant chunks."""
    llm = _get_llm()
    relevant: list[Document] = []

    for doc in state["documents"]:
        prompt = RELEVANCE_PROMPT.format(
            query=state["rewritten_query"],
            content=doc.page_content[:800],  # truncate long chunks
        )
        verdict = llm.invoke(prompt).content.strip().lower()
        if verdict.startswith("yes"):
            relevant.append(doc)

    return {"relevant_docs": relevant}


def increment_rewrite_count(state: RAGState) -> dict[str, Any]:
    """Track retry attempts as a normal graph state update."""
    return {"rewrites": state.get("rewrites", 0) + 1}


def use_fallback_documents(state: RAGState) -> dict[str, Any]:
    """Use retrieved documents when strict grading rejects everything."""
    return {"relevant_docs": state.get("documents", [])}


def generator(state: RAGState) -> dict[str, Any]:
    """Generate the final answer from relevant documents."""
    llm = _get_llm()
    context = "\n\n---\n\n".join(d.page_content for d in state["relevant_docs"])
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in state["history"]
    )
    prompt = GENERATE_PROMPT.format(
        history=history_text or "No prior conversation.",
        context=context,
        query=state["query"],
    )
    answer = llm.invoke(prompt).content.strip()
    return {"answer": answer}


# ---------------------------------------------------------------------------
# Routing logic
# ---------------------------------------------------------------------------

def route_after_relevance(state: RAGState) -> str:
    """Route to generation, retry with a rewritten query, or fallback context."""
    if state["relevant_docs"]:
        return "generate"
    if state.get("rewrites", 0) >= MAX_REWRITES:
        return "fallback"
    return "retry"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    """Compile and return the LangGraph agent."""
    graph = StateGraph(RAGState)

    graph.add_node("rewrite", query_rewriter)
    graph.add_node("retrieve", retriever_node)
    graph.add_node("grade", relevance_check)
    graph.add_node("increment_rewrite", increment_rewrite_count)
    graph.add_node("fallback", use_fallback_documents)
    graph.add_node("generate", generator)

    graph.set_entry_point("rewrite")

    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_edge("increment_rewrite", "rewrite")
    graph.add_edge("fallback", "generate")

    graph.add_conditional_edges(
        "grade",
        route_after_relevance,
        {
            "generate": "generate",
            "retry": "increment_rewrite",
            "fallback": "fallback",
        },
    )

    graph.add_edge("generate", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def run_agent(query: str, history: list[dict] | None = None) -> str:
    """
    Run the RAG agent for a single query.

    Args:
        query:   The user's question.
        history: List of {"role": "user"|"assistant", "content": "..."} dicts.

    Returns:
        The generated answer string.
    """
    app = build_graph()
    initial_state: RAGState = {
        "query": query,
        "rewritten_query": "",
        "history": history or [],
        "documents": [],
        "relevant_docs": [],
        "answer": "",
        "rewrites": 0,
    }
    final_state = app.invoke(initial_state)
    return final_state.get("answer", "I could not find a relevant answer.")
