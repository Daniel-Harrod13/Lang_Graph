"""Assemble the adaptive RAG state graph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from adaptive_rag.edges import decide_to_generate, grade_generation, route_question
from adaptive_rag.nodes import (
    generate,
    generate_direct,
    grade_documents,
    retrieve,
    web_search,
)
from adaptive_rag.state import GraphState


def build_graph() -> CompiledStateGraph:
    """Construct and compile the adaptive RAG graph.

    Graph topology::

        START ──route_question──┬── retrieve ── grade_documents ──decide──┬── generate ──grade_generation──┬── END
                                │                                         │       ▲                        │
                                ├── web_search ───────────────────────────►│       └────────────────────────┤
                                │                                                  (hallucination retry)   │
                                └── generate_direct ── END                                                 │
                                                                                   web_search ◄────────────┘
                                                                                   (answer miss fallback)
    """
    workflow = StateGraph(GraphState)

    # -- nodes --
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("web_search", web_search)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("generate_direct", generate_direct)

    # -- entry: route the question --
    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "retrieve": "retrieve",
            "web_search": "web_search",
            "generate_direct": "generate_direct",
        },
    )

    # -- retrieval path --
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "web_search": "web_search",
        },
    )

    # -- web search feeds directly into generation --
    workflow.add_edge("web_search", "generate")

    # -- post-generation quality gate --
    workflow.add_conditional_edges(
        "generate",
        grade_generation,
        {
            "__end__": END,
            "generate": "generate",
            "web_search": "web_search",
        },
    )

    # -- direct answer exits immediately --
    workflow.add_edge("generate_direct", END)

    return workflow.compile()
