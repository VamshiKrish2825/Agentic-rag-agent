# Introduction to Retrieval-Augmented Generation (RAG)

## What is RAG?

Retrieval-Augmented Generation (RAG) is an AI framework that combines two key capabilities:
1. **Retrieval** — searching a knowledge base for relevant information
2. **Generation** — using a large language model (LLM) to produce a coherent answer

The core idea is simple: instead of relying solely on knowledge baked into the LLM during training,
RAG allows the model to look up fresh, domain-specific information at query time.

## Why RAG Matters

Traditional LLMs have two key limitations:
- **Knowledge cutoff**: they only know what was in their training data
- **Hallucination**: they can confidently state incorrect facts

RAG addresses both by grounding the LLM's response in retrieved documents that are passed as
context. If the document says X, the model says X — rather than guessing.

## Key Components of a RAG System

### 1. Document Store
All your source documents (PDFs, web pages, databases) are preprocessed and stored.
Documents are split into smaller **chunks** — typically 200–1000 tokens — to enable precise retrieval.

### 2. Embedding Model
Each chunk is converted into a dense vector (embedding) that captures its semantic meaning.
Common choices: `all-MiniLM-L6-v2`, `text-embedding-ada-002`, `e5-large`.

### 3. Vector Database
Embeddings are stored in a vector database for fast similarity search.
Popular options: FAISS (local), Qdrant, Chroma, Pinecone, Weaviate.

### 4. Retriever
At query time, the user's question is embedded and the vector database is searched for the
top-k most similar chunks. These become the **context** for the LLM.

### 5. Generator (LLM)
The LLM receives: system prompt + retrieved context + user query → and produces the final answer.

## What Makes RAG "Agentic"?

A standard RAG pipeline is a fixed sequence: retrieve → generate.
An **Agentic RAG** adds reasoning loops:

- **Query rewriting**: the agent rewrites a vague query before retrieval
- **Relevance grading**: the agent evaluates whether retrieved chunks actually help
- **Iterative retrieval**: if the first retrieval fails, the agent tries a different query
- **Tool use**: the agent can call external APIs, run code, or search the web

This makes the system more robust — it doesn't give up if the first retrieval is imperfect.

## LangGraph and RAG

LangGraph is a framework for building stateful, graph-structured AI agents.
Each node in the graph is a function. Edges (including conditional edges) define the flow.

A typical LangGraph RAG graph looks like:

```
query_rewriter → retriever → relevance_grader
                                    ↓ (relevant found)
                              generator → END
                                    ↓ (not relevant, retry allowed)
                              query_rewriter  (loop back)
```

The graph keeps state in a `TypedDict` that all nodes read from and write to.

## Evaluation Metrics (RAGAS Framework)

Once built, RAG systems are evaluated using metrics like:

| Metric | What it measures |
|---|---|
| **Faithfulness** | Does the answer stick to the retrieved context? |
| **Answer Relevance** | Does the answer address the actual question? |
| **Context Recall** | Did we retrieve all the important information? |
| **Context Precision** | Are the retrieved chunks actually useful? |

## Common RAG Failure Modes

1. **Chunking too coarsely** — chunks so large that irrelevant text drowns out relevant text
2. **Chunking too finely** — chunks so small that context is lost
3. **Bad embeddings** — domain mismatch between embedding model and document type
4. **Missing reranker** — top-k by cosine similarity is not always the best k
5. **Prompt stuffing** — passing too much context confuses the LLM

## Summary

RAG is the practical bridge between raw LLM capability and real-world knowledge requirements.
Agentic RAG extends this with self-correcting loops that make the system more reliable.
LangGraph is an excellent framework for implementing agentic RAG because it makes the
control flow explicit, debuggable, and easy to extend.
