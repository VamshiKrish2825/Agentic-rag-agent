"""
state.py
--------
Typed state object shared across all LangGraph nodes.

Every key is explicitly typed so the graph's data-flow is transparent.
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document


class RAGState(TypedDict):
    """
    The single mutable bag of data that flows through the LangGraph pipeline.

    Attributes
    ----------
    query          : Original user question, never mutated.
    rewritten_query: Query after the rewriter node processes it.
    history        : Conversation turns [{role, content}, ...].
    documents      : Raw chunks returned by the retriever.
    relevant_docs  : Subset of documents that passed the relevance check.
    answer         : Final generated response.
    rewrites       : Number of query-rewrite iterations performed so far.
    """

    query: str
    rewritten_query: str
    history: list[dict]       # [{"role": "user"|"assistant", "content": str}]
    documents: list[Document]
    relevant_docs: list[Document]
    answer: str
    rewrites: int
