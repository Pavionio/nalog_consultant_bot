from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


TEXT_KEYS = ("text", "content", "cleaned_text", "extracted_text", "body")
SOURCE_KEYS = ("source_code", "source")
EXTERNAL_ID_KEYS = ("external_id", "doc_id")
URL_KEYS = ("canonical_url", "url", "source_url", "doc_url")


@dataclass(frozen=True)
class LocalDocument:
    source_code: str
    external_id: str
    text: str
    title: Optional[str] = None
    canonical_url: Optional[str] = None
    document_date: Optional[str] = None
    publication_date: Optional[str] = None
    document_number: Optional[str] = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class LocalCorpusStats:
    scanned_files: int
    loaded_documents: int
    skipped_empty: int
    skipped_duplicates: int
    by_source_code: dict[str, int]
    used_dir: Optional[str]


class LocalCorpusNotFound(RuntimeError):
    pass


def _first(mapping: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "":
            return value
    return None


def _stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _source_from_path(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else "local"


def _external_id_from_path(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    if path.name == "doc.txt" and len(rel.parts) >= 3:
        return rel.parts[-2]
    return str(rel.with_suffix("")).replace("\\", "/")


def _title_from_text(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:500]
    return fallback


def _read_text(path: Path) -> str:
    return path.read_text("utf-8", errors="replace")


def _doc_from_mapping(root: Path, path: Path, item: dict[str, Any]) -> Optional[LocalDocument]:
    text = _first(item, TEXT_KEYS)
    if not isinstance(text, str):
        return None
    source_code = str(_first(item, SOURCE_KEYS) or _source_from_path(root, path))
    external_id = str(_first(item, EXTERNAL_ID_KEYS) or _external_id_from_path(root, path))
    if not external_id:
        external_id = _stable_id(str(path.relative_to(root)))
    canonical_url = _first(item, URL_KEYS)
    return LocalDocument(
        source_code=source_code,
        external_id=external_id,
        text=text,
        title=item.get("title") or _title_from_text(text, path.stem),
        canonical_url=str(canonical_url) if canonical_url else None,
        document_date=item.get("document_date") or item.get("doc_date"),
        publication_date=item.get("publication_date") or item.get("published_at"),
        document_number=item.get("document_number") or item.get("doc_number"),
        metadata={k: v for k, v in item.items() if k not in TEXT_KEYS},
    )


def _iter_json_docs(root: Path, path: Path) -> Iterable[LocalDocument]:
    data = json.loads(_read_text(path))
    if isinstance(data, dict):
        if any(k in data for k in TEXT_KEYS):
            doc = _doc_from_mapping(root, path, data)
            if doc:
                yield doc
            return
        for key in ("documents", "items", "rows"):
            rows = data.get(key)
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        doc = _doc_from_mapping(root, path, row)
                        if doc:
                            yield doc
                return
    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                doc = _doc_from_mapping(root, path, row)
                if doc:
                    yield doc


def _iter_jsonl_docs(root: Path, path: Path) -> Iterable[LocalDocument]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                doc = _doc_from_mapping(root, path, row)
                if doc:
                    if not doc.external_id or doc.external_id == _external_id_from_path(root, path):
                        doc = LocalDocument(**{**doc.__dict__, "external_id": f"{doc.external_id}:{line_no}"})
                    yield doc


def _iter_text_doc(root: Path, path: Path) -> Iterable[LocalDocument]:
    text = _read_text(path)
    source_code = _source_from_path(root, path)
    external_id = _external_id_from_path(root, path)
    yield LocalDocument(
        source_code=source_code,
        external_id=external_id,
        text=text,
        title=_title_from_text(text, path.stem),
        metadata={"local_path": str(path.relative_to(root))},
    )


def _html_to_text(path: Path) -> str:
    from fetch.init_fetch import extract_main_text, extract_html_from_mime, html_to_text

    blob = path.read_bytes()
    head = blob[:2048]
    is_mime = bool(
        re.search(br"^\s*MIME-Version\s*:", head, re.I)
        or re.search(br"Content-Type\s*:\s*multipart/", head, re.I)
    )
    if is_mime:
        html, _ = extract_html_from_mime(blob)
        return html_to_text(html)
    return extract_main_text(blob)


def _iter_raw_doc(root: Path, path: Path) -> Iterable[LocalDocument]:
    text = _html_to_text(path)
    source_code = _source_from_path(root, path)
    external_id = _external_id_from_path(root, path)
    if path.name in ("page.html", "file.mhtml", "file.xml") and len(path.relative_to(root).parts) >= 3:
        external_id = path.relative_to(root).parts[-2]
    yield LocalDocument(
        source_code=source_code,
        external_id=external_id,
        text=text,
        title=_title_from_text(text, path.stem),
        metadata={"local_path": str(path.relative_to(root)), "raw_format": path.suffix.lower()},
    )


def _iter_docs(root: Path, *, include_raw: bool) -> tuple[list[LocalDocument], int]:
    docs: list[LocalDocument] = []
    scanned = 0
    suffixes = {".txt", ".md", ".json", ".jsonl"}
    if include_raw:
        suffixes |= {".html", ".htm", ".mhtml", ".xml"}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        scanned += 1
        try:
            if path.suffix.lower() in (".txt", ".md"):
                docs.extend(_iter_text_doc(root, path))
            elif path.suffix.lower() == ".json":
                docs.extend(_iter_json_docs(root, path))
            elif path.suffix.lower() == ".jsonl":
                docs.extend(_iter_jsonl_docs(root, path))
            else:
                docs.extend(_iter_raw_doc(root, path))
        except Exception as exc:
            print(f"[warn] skip unreadable local file {path}: {exc}")
    return docs, scanned


def _iter_docs_limited(root: Path, *, include_raw: bool, max_docs: int | None) -> tuple[list[LocalDocument], int]:
    if max_docs is None:
        return _iter_docs(root, include_raw=include_raw)
    docs: list[LocalDocument] = []
    scanned = 0
    suffixes = {".txt", ".md", ".json", ".jsonl"}
    if include_raw:
        suffixes |= {".html", ".htm", ".mhtml", ".xml"}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        scanned += 1
        try:
            before = len(docs)
            if path.suffix.lower() in (".txt", ".md"):
                docs.extend(_iter_text_doc(root, path))
            elif path.suffix.lower() == ".json":
                docs.extend(_iter_json_docs(root, path))
            elif path.suffix.lower() == ".jsonl":
                docs.extend(_iter_jsonl_docs(root, path))
            else:
                docs.extend(_iter_raw_doc(root, path))
            if len(docs) >= max_docs and len(docs) != before:
                return docs, scanned
        except Exception as exc:
            print(f"[warn] skip unreadable local file {path}: {exc}")
    return docs, scanned


def load_local_documents(
    input_dir: Path,
    fallback_raw_dir: Path | None = None,
    *,
    min_chars: int = 100,
    max_docs: int | None = None,
    print_stats: bool = True,
) -> list[LocalDocument]:
    docs, scanned = _iter_docs_limited(input_dir, include_raw=False, max_docs=max_docs) if input_dir.exists() else ([], 0)
    used_dir: Optional[Path] = input_dir if docs else None
    if not docs and fallback_raw_dir and fallback_raw_dir.exists():
        docs, scanned = _iter_docs_limited(fallback_raw_dir, include_raw=True, max_docs=max_docs)
        used_dir = fallback_raw_dir if docs else None

    if not docs:
        raise LocalCorpusNotFound(
            "Не найдены локальные тексты в data/text/ или data/raw/. "
            "Сначала запустите обычный ingest/fetch pipeline."
        )

    seen: set[tuple[str, str]] = set()
    loaded: list[LocalDocument] = []
    skipped_empty = 0
    skipped_duplicates = 0
    for doc in docs:
        text = (doc.text or "").strip()
        if len(text) < min_chars:
            skipped_empty += 1
            continue
        key = (doc.source_code, doc.external_id)
        if key in seen:
            skipped_duplicates += 1
            continue
        seen.add(key)
        loaded.append(LocalDocument(**{**doc.__dict__, "text": text}))
        if max_docs is not None and len(loaded) >= max_docs:
            break

    by_source = dict(Counter(d.source_code for d in loaded))
    if print_stats:
        print(f"Local corpus dir: {used_dir}")
        print(f"Files scanned: {scanned}")
        print(f"Documents loaded: {len(loaded)}")
        print(f"Skipped empty/short: {skipped_empty}")
        print(f"Skipped duplicates: {skipped_duplicates}")
        print(f"By source_code: {json.dumps(by_source, ensure_ascii=False, sort_keys=True)}")
    return loaded
