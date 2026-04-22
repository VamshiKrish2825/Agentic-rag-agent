"""
rag_agent.py
------------
LangGraph-based Agentic RAG pipeline.

Flow:
    User Query
        ↓
    query_rewriter  — cleans / expands the query
        ↓
    retriever       — fetches top-k chunks from the vector store
        ↓
    relevance_check — grades each chunk; routes to generate or rewrite
        ↓
    generator       — synthesises a final answer from accepted chunks
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.retriever import get_retriever
from src.state import RAGState
from src.prompts import (
    QUERY_REWRITE_PROMPT,
    RELEVANCE_PROMPT,
    GENERATE_PROMPT,
)

# ---------------------------------------------------------------------------
# LLM setup — swap model or base_url for any OpenAI-compatible endpoint
# ---------------------------------------------------------------------------

def _get_llm() -> ChatOpenAI:
    """Return the LLM.  Set OPENAI_API_KEY (or GROQ_API_KEY + base_url) in .env."""
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


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
    return {"rewritten_query": rewritten, "rewrites": state.get("rewrites", 0)}


def retriever_node(state: RAGState) -> dict[str, Any]:
    """Retrieve top-k documents for the (rewritten) query."""
    retriever = get_retriever()
    docs: list[Document] = retriever.invoke(state["rewritten_query"])
    return {"documents": docs}


def relevance_check(state: RAGState) -> dict[str, Any]:
    """Grade each retrieved document; keep only relevant ones."""
    llm = _get_llm()
    relevant: list[Document] = []

    for doc in state["documents"]:
        prompt = RELEVANCE_PROMPT.format(
            query=state["rewritten_query"],
            content=doc.page_content[:800],  # truncate long chunks
        )
        verdict = llm.invoke(prompt).content.strip().lower()
        if "yes" in verdict:
            relevant.append(doc)

    return {"relevant_docs": relevant}


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

MAX_REWRITES = 2


def route_after_relevance(state: RAGState) -> str:
    """
    If enough relevant docs exist → generate.
    If we've already rewritten too many times → generate anyway (best-effort).
    Otherwise → rewrite the query and try again.
    """
    if state["relevant_docs"]:
        return "generate"
    if state.get("rewrites", 0) >= MAX_REWRITES:
        # Fall back: use all retrieved docs so the user still gets an answer
        state["relevant_docs"] = state["documents"]
        return "generate"
    # Increment rewrite counter and loop back
    state["rewrites"] = state.get("rewrites", 0) + 1
    return "rewrite"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    """Compile and return the LangGraph agent."""
    graph = StateGraph(RAGState)

    # Register nodes
    graph.add_node("rewrite", query_rewriter)
    graph.add_node("retrieve", retriever_node)
    graph.add_node("grade", relevance_check)
    graph.add_node("generate", generator)

    # Entry point
    graph.set_entry_point("rewrite")

    # Edges
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade")

    # Conditional routing after grading
    graph.add_conditional_edges(
        "grade",
        route_after_relevance,
        {
            "generate": "generate",
            "rewrite": "rewrite",
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
