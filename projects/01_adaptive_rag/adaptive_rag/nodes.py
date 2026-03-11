"""Graph nodes — each function receives the current state and returns a
partial dict that gets merged back into the state."""

from __future__ import annotations

from typing import Any, Dict

from langchain_community.tools.tavily_search import TavilySearchResults

from adaptive_rag.chains import direct_chain, rag_chain, retrieval_grader
from adaptive_rag.retriever import get_retriever
from adaptive_rag.state import GraphState

web_search_tool = TavilySearchResults(max_results=3)


def retrieve(state: GraphState) -> Dict[str, Any]:
    """Retrieve relevant chunks from the vector store."""
    print("--- NODE: RETRIEVE FROM VECTORSTORE ---")
    question = state["question"]
    retriever = get_retriever()
    docs = retriever.invoke(question)
    return {"documents": [doc.page_content for doc in docs]}


def web_search(state: GraphState) -> Dict[str, Any]:
    """Fetch results from the web via Tavily."""
    print("--- NODE: WEB SEARCH ---")
    question = state["question"]
    results = web_search_tool.invoke({"query": question})
    documents = [r["content"] for r in results]
    return {"documents": documents}


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """Score each retrieved document for relevance; keep only relevant ones."""
    print("--- NODE: GRADE DOCUMENTS ---")
    question = state["question"]
    documents = state.get("documents", [])

    relevant: list[str] = []
    for doc in documents:
        result = retrieval_grader.invoke(
            {"question": question, "document": doc}
        )
        if result.score == "yes":
            print("  ✓ document relevant")
            relevant.append(doc)
        else:
            print("  ✗ document not relevant")

    return {"documents": relevant}


def generate(state: GraphState) -> Dict[str, Any]:
    """Generate an answer grounded in the retrieved documents."""
    print("--- NODE: GENERATE (RAG) ---")
    question = state["question"]
    documents = state.get("documents", [])
    retry_count = state.get("retry_count", 0)

    generation = rag_chain.invoke(
        {"question": question, "documents": "\n\n---\n\n".join(documents)}
    )
    return {"generation": generation, "retry_count": retry_count + 1}


def generate_direct(state: GraphState) -> Dict[str, Any]:
    """Answer directly without retrieval."""
    print("--- NODE: GENERATE (DIRECT) ---")
    question = state["question"]
    generation = direct_chain.invoke({"question": question})
    return {"generation": generation}
