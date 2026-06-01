"""Agentic RAG experiment.

An agent LLM (openai/gpt-oss-20b via an OpenAI-compatible LM Studio endpoint)
iteratively decides what to search; the only tool is dense semantic search over
Qdrant (e5-large embeddings, NO reranker). Isolated from src/ — it only imports
the reusable building blocks (STEmbedder, Retriever, build_context, RAGConfig).

See README.md for how to run; the experiment is described in text/main.tex.
"""
