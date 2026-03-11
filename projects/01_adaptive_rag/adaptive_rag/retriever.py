"""Vector store construction and retriever factory.

On first call, loads markdown documents from the ``docs/`` directory,
chunks them, embeds them with OpenAI embeddings, and stores them in an
in-memory Chroma collection.  Subsequent calls return the cached retriever.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

_retriever = None


def _load_documents() -> list[Document]:
    """Read every ``.md`` file under *DOCS_DIR* into LangChain Documents."""
    documents: list[Document] = []
    for filepath in sorted(DOCS_DIR.glob("*.md")):
        text = filepath.read_text(encoding="utf-8")
        documents.append(
            Document(page_content=text, metadata={"source": filepath.name})
        )
    return documents


def get_retriever(k: int = 4):
    """Return a Chroma retriever, building the index on first call."""
    global _retriever
    if _retriever is not None:
        return _retriever

    documents = _load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=splits,
        collection_name="adaptive-rag",
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    )

    _retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return _retriever
