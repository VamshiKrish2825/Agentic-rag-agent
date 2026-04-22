# LangGraph: Building Stateful AI Agents

## Overview

LangGraph is an open-source library from LangChain that enables developers to build
stateful, multi-actor AI applications using a graph-based control flow.

It is designed for situations where a simple linear chain of LLM calls is not enough —
for example, when you need loops, conditional branching, human approval steps, or
parallel execution.

## Core Concepts

### State
Every LangGraph application revolves around a shared **state** object — typically a `TypedDict`.
All nodes in the graph read from and write to this state.

```python
from typing import TypedDict

class MyState(TypedDict):
    query: str
    documents: list
    answer: str
```

### Nodes
Nodes are plain Python functions that accept the state and return a partial update to it.

```python
def my_node(state: MyState) -> dict:
    # do work ...
    return {"answer": "some result"}
```

### Edges
Edges define which node runs after which.

- **Fixed edge**: always go from A to B
- **Conditional edge**: choose the next node based on state at runtime

```python
graph.add_edge("nodeA", "nodeB")                           # fixed
graph.add_conditional_edges("nodeB", my_router_function)   # conditional
```

### StateGraph
`StateGraph` is the graph builder class. You register nodes and edges, set an entry point,
then call `.compile()` to get a runnable agent.

## Why Use LangGraph over Plain LangChain?

| Feature | LangChain (LCEL) | LangGraph |
|---|---|---|
| Control flow | Linear / sequential | Graph: loops, branches |
| State management | Manual | Built-in shared state |
| Human-in-the-loop | Hard | First-class support |
| Parallel execution | Limited | Native fan-out/fan-in |
| Debugging | Difficult | LangSmith traces + checkpoints |

## Building a Minimal RAG Agent in LangGraph

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(MyState)

graph.add_node("rewrite",  query_rewriter)
graph.add_node("retrieve", retriever_node)
graph.add_node("grade",    relevance_check)
graph.add_node("generate", generator)

graph.set_entry_point("rewrite")
graph.add_edge("rewrite", "retrieve")
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges(
    "grade",
    route_after_relevance,
    {"generate": "generate", "rewrite": "rewrite"},
)
graph.add_edge("generate", END)

app = graph.compile()
result = app.invoke({"query": "What is RAG?"})
```

## Persistence and Memory

LangGraph supports **checkpointing** — saving state between graph runs.
This enables:
- Multi-turn conversations where the agent remembers prior exchanges
- Long-running tasks that can be paused and resumed
- Audit trails for production systems

Common checkpointer backends: `MemorySaver` (in-RAM), `SqliteSaver`, `PostgresSaver`.

## Human-in-the-Loop

LangGraph's `interrupt_before` and `interrupt_after` parameters let you pause execution
and wait for human input before continuing.

```python
app = graph.compile(interrupt_before=["generate"])
# run until the interrupt, show state to human, then resume
app.invoke(state, config={"configurable": {"thread_id": "1"}})
```

## LangSmith Integration

LangGraph integrates directly with LangSmith for tracing.
Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` in your environment to get
full step-by-step traces of every node execution, LLM call, and state transition.

## When NOT to Use LangGraph

LangGraph adds complexity. Use a plain LCEL chain when:
- Your pipeline is strictly linear with no branching
- You don't need loops or retries
- You want minimal dependencies

Use LangGraph when:
- You need self-correcting loops (e.g., retry retrieval if results are poor)
- You need parallel steps (e.g., query multiple data sources simultaneously)
- You need human approval gates in the workflow

## Summary

LangGraph gives you explicit, inspectable control flow for AI agents.
For RAG systems, it's particularly powerful because it lets the agent decide whether
to trust its retrieval results or try again — making the system much more robust
than a naive retrieve-then-generate pipeline.
