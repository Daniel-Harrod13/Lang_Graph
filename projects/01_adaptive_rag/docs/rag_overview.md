# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is an architecture that enhances LLM outputs by grounding them in retrieved external knowledge. Instead of relying solely on parameters learned during pre-training, RAG systems fetch relevant documents at inference time and incorporate them into the generation process.

## Why RAG?

LLMs face fundamental limitations that RAG addresses:

- **Knowledge cutoff**: Models only know what existed in their training data. RAG provides access to current information.
- **Hallucination**: Models can generate plausible but incorrect information. Retrieved context provides factual grounding.
- **Domain specificity**: Fine-tuning on proprietary data is expensive. RAG enables knowledge injection without retraining.
- **Traceability**: Retrieved documents provide citations, making it possible to verify and audit generated answers.

## Architecture

A standard RAG pipeline consists of three stages: indexing, retrieval, and generation.

### Indexing

Documents are preprocessed and stored in a searchable index:

1. **Document loading**: Ingest raw documents from various sources (PDFs, web pages, databases, APIs).
2. **Chunking**: Split documents into smaller segments. Common strategies include fixed-size chunks with overlap, recursive character splitting, and semantic chunking that respects document structure.
3. **Embedding**: Convert text chunks into dense vector representations using embedding models (e.g., OpenAI text-embedding-3-small, Cohere embed-v3, open-source models like BGE or E5).
4. **Storage**: Store vectors in a vector database (Chroma, Pinecone, Weaviate, FAISS) alongside the original text and metadata.

### Retrieval

Given a user query, the retriever finds the most relevant document chunks:

- **Dense retrieval** computes the query embedding and finds nearest neighbors in vector space using cosine similarity or maximum inner product search.
- **Sparse retrieval** uses traditional keyword matching (BM25, TF-IDF) which excels at exact term matching.
- **Hybrid retrieval** combines dense and sparse methods, often using Reciprocal Rank Fusion (RRF) to merge ranked lists.

The number of retrieved chunks (top-k) balances between recall and context window utilization. Too few chunks risk missing relevant information; too many dilute the signal and waste context.

### Generation

Retrieved chunks are formatted into a prompt alongside the user's question. The LLM generates an answer grounded in the provided context. A typical RAG prompt structure includes a system instruction, the retrieved context block, and the user question.

## Advanced RAG Techniques

### Query Transformation

Improve retrieval by reformulating the query:
- **HyDE (Hypothetical Document Embeddings)**: Generate a hypothetical answer, embed it, and use that embedding for retrieval — often more effective than embedding the raw question.
- **Multi-query**: Generate multiple query variants and retrieve documents for each, then deduplicate results.
- **Step-back prompting**: Abstract the query to a higher-level question to retrieve broader context.

### Adaptive RAG

Adaptive RAG dynamically chooses the retrieval strategy based on query characteristics. A routing mechanism classifies the query and directs it to the appropriate pipeline — vector retrieval for domain-specific questions, web search for current events, or direct generation for simple queries. This prevents unnecessary retrieval overhead and improves answer quality.

### Self-RAG

Self-RAG adds reflection mechanisms where the model decides whether retrieval is needed, evaluates the relevance of retrieved documents, and checks whether its generation is supported by the evidence. These self-assessment steps create a quality-controlled pipeline.

### Corrective RAG (CRAG)

CRAG evaluates retrieved documents for relevance before generation. If documents are irrelevant, the system falls back to alternative retrieval strategies (e.g., web search). A knowledge refinement step extracts and filters relevant information from retrieved documents.

## Evaluation

RAG systems require evaluation at multiple levels:

- **Retrieval quality**: Precision@k, Recall@k, Mean Reciprocal Rank (MRR), Normalized Discounted Cumulative Gain (NDCG)
- **Generation quality**: Faithfulness (is the answer grounded in context?), answer relevance (does it address the question?), and completeness
- **End-to-end**: RAGAS framework provides composite metrics combining context relevance, faithfulness, and answer quality

## Common Pitfalls

1. Chunks too large or too small — losing either context or granularity
2. Poor embedding model choice for the domain
3. Not handling the "no relevant documents found" case gracefully
4. Stuffing too many chunks into context, causing the model to lose focus
5. Ignoring metadata filtering opportunities (date ranges, document types, categories)
