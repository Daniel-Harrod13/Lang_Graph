"""Conditional edge functions — each one inspects the current state and
returns a string key that the graph uses to select the next node."""

from __future__ import annotations

from typing import Literal

from adaptive_rag.chains import (
    answer_grader,
    hallucination_grader,
    question_router,
)
from adaptive_rag.state import GraphState

MAX_RETRIES = 3


def route_question(
    state: GraphState,
) -> Literal["retrieve", "web_search", "generate_direct"]:
    """Classify the question and pick a retrieval strategy."""
    print("--- EDGE: ROUTE QUESTION ---")
    question = state["question"]
    result = question_router.invoke({"question": question})
    datasource = result.datasource
    print(f"  → routing to: {datasource}")

    if datasource == "vectorstore":
        return "retrieve"
    if datasource == "web_search":
        return "web_search"
    return "generate_direct"


def decide_to_generate(
    state: GraphState,
) -> Literal["generate", "web_search"]:
    """After grading, decide whether we have enough relevant docs."""
    print("--- EDGE: DECIDE TO GENERATE ---")
    documents = state.get("documents", [])

    if not documents:
        print("  → no relevant documents — falling back to web search")
        return "web_search"

    print(f"  → {len(documents)} relevant document(s) — proceeding to generate")
    return "generate"


def grade_generation(
    state: GraphState,
) -> Literal["__end__", "generate", "web_search"]:
    """Two-stage quality gate: check hallucination, then answer relevance.

    Returns ``__end__`` when the answer is grounded *and* addresses the
    question, ``generate`` to retry if hallucinated, or ``web_search``
    to gather fresh context if the answer misses the question.
    """
    print("--- EDGE: GRADE GENERATION ---")
    question = state["question"]
    generation = state["generation"]
    documents = state.get("documents", [])
    retry_count = state.get("retry_count", 0)

    if retry_count >= MAX_RETRIES:
        print("  → max retries reached — returning best effort answer")
        return "__end__"

    # Stage 1: hallucination check
    hall = hallucination_grader.invoke(
        {"documents": "\n\n---\n\n".join(documents), "generation": generation}
    )
    if hall.score == "no":
        print("  ✗ hallucination detected — regenerating")
        return "generate"

    # Stage 2: answer relevance check
    ans = answer_grader.invoke(
        {"question": question, "generation": generation}
    )
    if ans.score == "no":
        print("  ✗ answer doesn't address question — trying web search")
        return "web_search"

    print("  ✓ answer is grounded and relevant")
    return "__end__"
