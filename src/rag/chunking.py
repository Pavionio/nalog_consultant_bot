from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_method: str
    chunk_size: int = 1024
    chunk_overlap: int = 128
    chunk_tokenizer: str = "character"
    chunk_min_sentences: int = 2
    chunk_min_characters_per_sentence: int = 12
    semantic_threshold: float = 0.8
    semantic_similarity_window: int = 3
    semantic_skip_window: int = 0
    semantic_embedding_model: str = "minishlab/potion-base-32M"
    parent_chunk_size: int = 3072
    parent_chunk_overlap: int = 256
    child_chunk_size: int = 768
    child_chunk_overlap: int = 96
    parent_chunker_method: str = "recursive_legal"
    child_chunker_method: str = "sentence"


LEGAL_DELIMITERS = [
    "\n\n",
    "\nРаздел ",
    "\nГлава ",
    "\nСтатья ",
    "\nПункт ",
    "\nПодпункт ",
    "\nПриложение ",
    "\nПисьмо ",
    "\nПриказ ",
    "\n1. ",
    "\n2. ",
    "\n3. ",
    "\nа) ",
    "\nб) ",
    "; ",
    ". ",
]


def stable_chunk_point_id(chunk: dict[str, Any]) -> str:
    if chunk.get("chunk_method") == "parent_child":
        raw = (
            f"{chunk.get('source_code')}:{chunk.get('external_id')}:"
            f"{chunk.get('chunk_method')}:{chunk.get('parent_i')}:{chunk.get('child_i')}:"
            f"{chunk.get('parent_chunk_size')}:{chunk.get('child_chunk_size')}"
        )
    else:
        raw = (
            f"{chunk.get('source_code')}:{chunk.get('external_id')}:"
            f"{chunk.get('chunk_method')}:{chunk.get('chunk_size')}:{chunk.get('chunk_overlap')}:"
            f"{chunk.get('chunk_i')}"
        )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


def _chunk_text_obj(ch: Any) -> str:
    if isinstance(ch, dict):
        return str(ch.get("text") or ch.get("chunk") or "")
    return str(getattr(ch, "text", "") or "")


def _chunk_start(ch: Any) -> Optional[int]:
    if isinstance(ch, dict):
        value = ch.get("start_index") or ch.get("start")
    else:
        value = getattr(ch, "start_index", None)
        if value is None:
            value = getattr(ch, "start", None)
    return int(value) if value is not None else None


def _chunk_end(ch: Any) -> Optional[int]:
    if isinstance(ch, dict):
        value = ch.get("end_index") or ch.get("end")
    else:
        value = getattr(ch, "end_index", None)
        if value is None:
            value = getattr(ch, "end", None)
    return int(value) if value is not None else None


def _run_chunker(chunker: Any, text: str) -> list[Any]:
    if hasattr(chunker, "chunk"):
        return list(chunker.chunk(text))
    out = chunker(text)
    return list(out)


def _make_recursive_rules() -> Any:
    from chonkie.types import RecursiveLevel, RecursiveRules

    return RecursiveRules(
        levels=[
            RecursiveLevel(delimiters=LEGAL_DELIMITERS, include_delim="prev"),
            RecursiveLevel(delimiters=[". ", "! ", "? ", "\n"], include_delim="prev"),
            RecursiveLevel(delimiters=[":", ";", ",", " - ", " -"], include_delim="prev"),
            RecursiveLevel(delimiters=None, whitespace=True, include_delim="prev"),
            RecursiveLevel(delimiters=None, whitespace=False, include_delim="prev"),
        ]
    )


def _make_chunker(cfg: ChunkingConfig) -> tuple[Any, dict[str, Any]]:
    meta: dict[str, Any] = {}
    method = cfg.chunk_method
    if method == "token":
        from chonkie import TokenChunker

        return TokenChunker(
            tokenizer=cfg.chunk_tokenizer or "character",
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        ), meta
    if method == "sentence":
        from chonkie import SentenceChunker

        return SentenceChunker(
            tokenizer=cfg.chunk_tokenizer or "character",
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            min_sentences_per_chunk=cfg.chunk_min_sentences,
            min_characters_per_sentence=cfg.chunk_min_characters_per_sentence,
        ), meta
    if method == "recursive":
        from chonkie import RecursiveChunker

        if cfg.chunk_overlap:
            meta["overlap_applied_posthoc"] = True
        return RecursiveChunker(
            tokenizer=cfg.chunk_tokenizer or "character",
            chunk_size=cfg.chunk_size,
            min_characters_per_chunk=24,
        ), meta
    if method == "recursive_legal":
        from chonkie import RecursiveChunker

        try:
            rules = _make_recursive_rules()
            meta["recursive_rules_name"] = "russian_tax_legal"
            if cfg.chunk_overlap:
                meta["overlap_applied_posthoc"] = True
            return RecursiveChunker(
                tokenizer=cfg.chunk_tokenizer or "character",
                chunk_size=cfg.chunk_size,
                rules=rules,
                min_characters_per_chunk=24,
            ), meta
        except Exception as exc:
            print(f"[warn] Chonkie custom RecursiveRules unavailable, fallback to default: {exc}")
            meta["recursive_rules_name"] = "fallback_default"
            if cfg.chunk_overlap:
                meta["overlap_applied_posthoc"] = True
            return RecursiveChunker(
                tokenizer=cfg.chunk_tokenizer or "character",
                chunk_size=cfg.chunk_size,
                min_characters_per_chunk=24,
            ), meta
    if method == "semantic":
        try:
            from chonkie import SemanticChunker
        except Exception as exc:
            raise RuntimeError('SemanticChunker requires optional dependencies: pip install "chonkie[semantic]"') from exc
        return SemanticChunker(
            embedding_model=cfg.semantic_embedding_model,
            threshold=cfg.semantic_threshold,
            chunk_size=cfg.chunk_size,
            similarity_window=cfg.semantic_similarity_window,
            skip_window=cfg.semantic_skip_window,
            min_sentences_per_chunk=cfg.chunk_min_sentences,
            min_characters_per_sentence=cfg.chunk_min_characters_per_sentence,
        ), meta
    raise ValueError(f"Unsupported chunk_method={method!r}")


def _apply_posthoc_overlap(texts: list[str], overlap: int) -> list[str]:
    if overlap <= 0:
        return texts
    out: list[str] = []
    prev = ""
    for i, text in enumerate(texts):
        if i == 0 or not prev:
            out.append(text)
        else:
            tail = prev[-overlap:]
            out.append((tail + "\n" + text).strip())
        prev = text
    return out


def _base_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_code": metadata.get("source_code"),
        "external_id": metadata.get("external_id"),
        "title": metadata.get("title"),
        "canonical_url": metadata.get("canonical_url"),
        "url": metadata.get("canonical_url") or metadata.get("url"),
        "document_date": metadata.get("document_date"),
        "publication_date": metadata.get("publication_date"),
        "document_number": metadata.get("document_number"),
    }


def _chunk_once(text: str, metadata: dict[str, Any], cfg: ChunkingConfig) -> list[dict[str, Any]]:
    chunker, extra_meta = _make_chunker(cfg)
    raw_chunks = _run_chunker(chunker, text)
    texts = [_chunk_text_obj(ch).strip() for ch in raw_chunks]
    texts = [t for t in texts if t]
    if cfg.chunk_method in ("recursive", "recursive_legal") and cfg.chunk_overlap:
        texts = _apply_posthoc_overlap(texts, cfg.chunk_overlap)

    base = _base_metadata(metadata)
    chunks: list[dict[str, Any]] = []
    cursor = 0
    for i, chunk_text in enumerate(texts):
        raw = raw_chunks[i] if i < len(raw_chunks) else {}
        start = _chunk_start(raw)
        end = _chunk_end(raw)
        if start is None:
            found = text.find(chunk_text[:80], cursor)
            start = found if found >= 0 else None
        if end is None and start is not None:
            end = start + len(chunk_text)
        if end is not None:
            cursor = max(cursor, end)
        chunks.append(
            {
                **base,
                **extra_meta,
                "text": chunk_text,
                "chunk_i": i,
                "chunk_method": cfg.chunk_method,
                "chunk_size": cfg.chunk_size,
                "chunk_overlap": cfg.chunk_overlap,
                "chunk_tokenizer": cfg.chunk_tokenizer,
                "chunk_char_len": len(chunk_text),
                "chunk_token_count": len(chunk_text) if cfg.chunk_tokenizer == "character" else None,
                "chunk_start_index": start,
                "chunk_end_index": end,
            }
        )
    return chunks


def chunk_document(text: str, metadata: dict[str, Any], cfg: ChunkingConfig) -> list[dict[str, Any]]:
    if cfg.chunk_method != "parent_child":
        return _chunk_once(text, metadata, cfg)

    print("[warn] parent_child stores parent_text in every child payload; Qdrant collection will be larger.")
    parent_cfg = ChunkingConfig(
        chunk_method=cfg.parent_chunker_method,
        chunk_size=cfg.parent_chunk_size,
        chunk_overlap=cfg.parent_chunk_overlap,
        chunk_tokenizer=cfg.chunk_tokenizer,
        chunk_min_sentences=cfg.chunk_min_sentences,
        chunk_min_characters_per_sentence=cfg.chunk_min_characters_per_sentence,
        semantic_threshold=cfg.semantic_threshold,
        semantic_similarity_window=cfg.semantic_similarity_window,
        semantic_skip_window=cfg.semantic_skip_window,
        semantic_embedding_model=cfg.semantic_embedding_model,
    )
    child_cfg = ChunkingConfig(
        chunk_method=cfg.child_chunker_method,
        chunk_size=cfg.child_chunk_size,
        chunk_overlap=cfg.child_chunk_overlap,
        chunk_tokenizer=cfg.chunk_tokenizer,
        chunk_min_sentences=cfg.chunk_min_sentences,
        chunk_min_characters_per_sentence=cfg.chunk_min_characters_per_sentence,
        semantic_threshold=cfg.semantic_threshold,
        semantic_similarity_window=cfg.semantic_similarity_window,
        semantic_skip_window=cfg.semantic_skip_window,
        semantic_embedding_model=cfg.semantic_embedding_model,
    )

    parent_chunks = _chunk_once(text, metadata, parent_cfg)
    base = _base_metadata(metadata)
    out: list[dict[str, Any]] = []
    global_i = 0
    source_code = str(base.get("source_code") or "")
    external_id = str(base.get("external_id") or "")
    for parent_i, parent in enumerate(parent_chunks):
        parent_text = parent["text"]
        parent_id = f"{source_code}:{external_id}:parent:{parent_i}"
        child_chunks = _chunk_once(parent_text, metadata, child_cfg)
        for child_i, child in enumerate(child_chunks):
            child_text = child["text"]
            out.append(
                {
                    **base,
                    "text": child_text,
                    "child_text": child_text,
                    "parent_text": parent_text,
                    "chunk_i": global_i,
                    "chunk_method": "parent_child",
                    "chunk_role": "child",
                    "parent_id": parent_id,
                    "parent_i": parent_i,
                    "child_i": child_i,
                    "parent_chunk_size": cfg.parent_chunk_size,
                    "parent_chunk_overlap": cfg.parent_chunk_overlap,
                    "child_chunk_size": cfg.child_chunk_size,
                    "child_chunk_overlap": cfg.child_chunk_overlap,
                    "parent_chunker_method": cfg.parent_chunker_method,
                    "child_chunker_method": cfg.child_chunker_method,
                    "chunk_size": cfg.child_chunk_size,
                    "chunk_overlap": cfg.child_chunk_overlap,
                    "chunk_tokenizer": cfg.chunk_tokenizer,
                    "chunk_char_len": len(child_text),
                    "chunk_token_count": len(child_text) if cfg.chunk_tokenizer == "character" else None,
                    "chunk_start_index": child.get("chunk_start_index"),
                    "chunk_end_index": child.get("chunk_end_index"),
                    "parent_char_len": len(parent_text),
                }
            )
            global_i += 1
    return out

