from __future__ import annotations

import json
from typing import Dict, List

from src.rag.core import RAGConfig, STEmbedder, Retriever, LlamaCppChatClient, rag_answer


def main() -> None:
    cfg = RAGConfig()

    embedder = STEmbedder(cfg.embed_model_name)
    retriever = Retriever(cfg, embedder)
    llm = LlamaCppChatClient(cfg)

    history: List[Dict[str, str]] = []

    print("RAG chat. Команды: /reset, /debug, /exit")
    debug = False

    while True:
        q = input("\n> ").strip()
        if not q:
            continue
        if q == "/exit":
            break
        if q == "/reset":
            history = []
            print("history cleared")
            continue
        if q == "/debug":
            debug = not debug
            print(f"debug={debug}")
            continue

        try:
            result = rag_answer(cfg, retriever, llm, q, chat_history=history)
        except Exception as e:
            print(f"\n[ошибка]: {e}")
            continue

        # обновляем историю (минимально)
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": result["answer"] or ""})
        history = history[-20:]

        print("\n" + result["answer"])

        if debug:
            print("\n--- debug:sources ---")
            print(json.dumps(result["sources"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()