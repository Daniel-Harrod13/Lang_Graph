from __future__ import annotations

from typing import List

from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):
    """Shared state that flows through every node in the adaptive RAG graph.

    Fields are populated incrementally — only ``question`` is required at
    invocation time; the remaining fields are filled by nodes as execution
    proceeds through the graph.
    """

    question: str
    generation: str
    documents: List[str]
    datasource: str
    retry_count: int
