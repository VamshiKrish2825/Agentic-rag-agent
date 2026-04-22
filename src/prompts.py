"""
prompts.py
----------
All system / user prompt templates in one place.

Keeping prompts separate from logic makes it easy to:
  - tune them without touching pipeline code
  - unit-test them in isolation
  - swap them for domain-specific variants
"""

# ---------------------------------------------------------------------------
# Query Rewriter
# ---------------------------------------------------------------------------

QUERY_REWRITE_PROMPT = """You are a search query optimizer.

Given the conversation history and the user's latest message, rewrite the
query so it is:
  - Self-contained (no pronouns that refer to earlier messages)
  - Optimised for semantic similarity search
  - Concise (one sentence)

Do NOT answer the question — only return the rewritten query.

Conversation history:
{history}

User query: {query}

Rewritten query:"""


# ---------------------------------------------------------------------------
# Relevance Grader
# ---------------------------------------------------------------------------

RELEVANCE_PROMPT = """You are a strict relevance grader.

Decide whether the document below is useful for answering the query.
Reply with ONLY "yes" or "no".

Query: {query}

Document:
{content}

Is this document relevant? (yes/no):"""


# ---------------------------------------------------------------------------
# Answer Generator
# ---------------------------------------------------------------------------

GENERATE_PROMPT = """You are a helpful assistant that answers questions strictly
based on the provided context documents.

Rules:
  1. If the context contains the answer, provide it clearly and concisely.
  2. If the context is insufficient, say "I don't have enough information in
     the provided documents to answer this question."
  3. Never make up facts that aren't in the context.
  4. You may use the conversation history to maintain coherent dialogue.

Conversation history:
{history}

Context documents:
{context}

User question: {query}

Answer:"""
