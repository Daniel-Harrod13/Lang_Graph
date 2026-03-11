"""LLM chains used by the adaptive RAG graph.

Each chain pairs a prompt template with a model invocation.  Structured-output
chains use Pydantic models so that downstream nodes receive typed objects
rather than raw text.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------------------
# Query router
# ---------------------------------------------------------------------------

class RouteQuery(BaseModel):
    """Classify the best datasource for answering a user question."""

    datasource: Literal["vectorstore", "web_search", "direct"] = Field(
        description=(
            "Route to 'vectorstore' for questions about LLM agents, prompt "
            "engineering, or RAG.  Route to 'web_search' for questions "
            "requiring current or recent information.  Route to 'direct' "
            "for straightforward general-knowledge questions."
        )
    )


question_router = (
    ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert at routing user questions.\n\n"
            "Available datasources:\n"
            "- **vectorstore** – contains documents about LLM agents, "
            "prompt engineering techniques, and retrieval-augmented generation.\n"
            "- **web_search** – use for current events, recent news, or any "
            "question that needs up-to-date information.\n"
            "- **direct** – use for simple factual or general-knowledge "
            "questions that don't require retrieval.\n\n"
            "Pick the single best datasource for the question.",
        ),
        ("human", "{question}"),
    ])
    | llm.with_structured_output(RouteQuery)
)

# ---------------------------------------------------------------------------
# Retrieval grader
# ---------------------------------------------------------------------------

class GradeDocuments(BaseModel):
    """Binary relevance score for a retrieved document."""

    score: Literal["yes", "no"] = Field(
        description="Is the document relevant to the question? 'yes' or 'no'."
    )


retrieval_grader = (
    ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a grader assessing whether a retrieved document is "
            "relevant to a user question.  If the document contains keywords "
            "or semantic meaning related to the question, grade it as "
            "relevant.  Give a binary 'yes' or 'no' score.",
        ),
        ("human", "Document:\n\n{document}\n\nQuestion: {question}"),
    ])
    | llm.with_structured_output(GradeDocuments)
)

# ---------------------------------------------------------------------------
# RAG generator
# ---------------------------------------------------------------------------

rag_chain = (
    ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an assistant answering questions using retrieved context. "
            "Use the following documents to answer concisely and accurately.  "
            "If the documents don't contain enough information, say so.\n\n"
            "Documents:\n{documents}",
        ),
        ("human", "{question}"),
    ])
    | llm
    | StrOutputParser()
)

# ---------------------------------------------------------------------------
# Hallucination grader
# ---------------------------------------------------------------------------

class GradeHallucinations(BaseModel):
    """Binary score: is the generation grounded in the provided facts?"""

    score: Literal["yes", "no"] = Field(
        description=(
            "'yes' if the answer is grounded in / supported by the documents, "
            "'no' otherwise."
        )
    )


hallucination_grader = (
    ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a grader assessing whether an LLM generation is "
            "grounded in a set of retrieved facts.  Give a binary 'yes' or "
            "'no' score.  'yes' means every claim in the generation is "
            "supported by the facts.",
        ),
        ("human", "Facts:\n\n{documents}\n\nGeneration:\n\n{generation}"),
    ])
    | llm.with_structured_output(GradeHallucinations)
)

# ---------------------------------------------------------------------------
# Answer grader
# ---------------------------------------------------------------------------

class GradeAnswer(BaseModel):
    """Binary score: does the generation address the original question?"""

    score: Literal["yes", "no"] = Field(
        description="'yes' if the answer resolves the question, 'no' otherwise."
    )


answer_grader = (
    ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a grader assessing whether an answer addresses a user "
            "question.  Give a binary 'yes' or 'no' score.  'yes' means the "
            "answer resolves the question.",
        ),
        ("human", "Question: {question}\n\nAnswer: {generation}"),
    ])
    | llm.with_structured_output(GradeAnswer)
)

# ---------------------------------------------------------------------------
# Direct answer (no retrieval)
# ---------------------------------------------------------------------------

direct_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question directly and concisely."),
        ("human", "{question}"),
    ])
    | llm
    | StrOutputParser()
)
