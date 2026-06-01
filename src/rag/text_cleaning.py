from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Sequence


SUPPORTED_CLEANING_PROFILES = (
    "raw",
    "clean_basic",
    "clean_legal",
    "clean_aggressive",
    "clean_no_boilerplate",
)

_TAX_TERMS = (
    "ндфл",
    "ндс",
    "усн",
    "нпд",
    "псн",
    "енвд",
    "налоговый агент",
    "налогоплательщик",
    "декларация",
    "уведомление",
    "вычет",
    "штраф",
    "пени",
    "срок",
    "доход",
    "расход",
)

_LEGAL_KEYWORDS = (
    "нк рф",
    "налоговый кодекс",
    "федеральный закон",
    "фз",
    "письмо",
    "приказ",
    "постановление",
    "определение",
    "решение",
    "статья",
    "пункт",
    "подпункт",
    "раздел",
    "глава",
    "приложение",
)

_NAV_NOISE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bверсия\s+для\s+печати\b",
        r"\bподелиться\b",
        r"\bнаверх\b",
        r"\bглавная\b",
        r"\bпоиск\b",
        r"\bличный\s+кабинет\b",
        r"\bподписаться\b",
        r"\bофициальный\s+сайт\b",
        r"\bcookie\b",
    )
]

_DATE_DOTTED_RE = re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b")
_DATE_TEXT_RE = re.compile(
    r"\b\d{1,2}\s+(?:январ[ья]|феврал[ья]|марта?|апрел[ья]|ма[йя]|"
    r"июн[ья]|июл[ья]|август[ае]?|сентябр[ья]|октябр[ья]|ноябр[ья]|декабр[ья])\s+\d{4}\b",
    re.IGNORECASE,
)
_DOC_NUMBER_RE = re.compile(r"(?:№|(?<!\w)N(?!\w))\s*[\w\-/\.]+", re.IGNORECASE)
_ARTICLE_RE = re.compile(r"\b(?:статья|ст\.)\s*\d+(?:\.\d+)?", re.IGNORECASE)
_POINT_RE = re.compile(r"\b(?:пункт|п\.|подпункт|подп\.)\s*[а-яa-z0-9\.\)\(]+", re.IGNORECASE)
_ENUM_LIST_RE = re.compile(r"^\s*(?:\d+\.\s+|[а-яa-z]\)\s+)", re.IGNORECASE)
_MONEY_RE = re.compile(r"\b\d[\d\s]*(?:[.,]\d+)?\s*(?:руб|руб\.|₽)\b", re.IGNORECASE)
_BREADCRUMB_RE = re.compile(r"^\s*[^\n]{1,160}(?:\s*[>›»]\s*[^\n]{1,80}){1,}\s*$")
_ALNUM_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]")
_PUNCT_RE = re.compile(r"[^\w\sА-Яа-яЁё]")


@dataclass
class CleaningResult:
    text: str
    profile: str
    original_char_len: int
    cleaned_char_len: int
    removed_char_ratio: float
    removed_line_count: int
    warning_count: int
    warnings: list[str]
    stats: dict[str, Any]


def _safe_text(text: str | None) -> str:
    return str(text or "")


def _line_key(line: str) -> str:
    return normalize_whitespace(line).lower().replace("ё", "е")


def _is_navigation_noise_line(line: str) -> bool:
    if not line:
        return False
    if _BREADCRUMB_RE.match(line):
        return True
    return any(pattern.search(line) for pattern in _NAV_NOISE_PATTERNS)


def _is_aggressive_noise_line(line: str) -> bool:
    if not line:
        return False
    letters_or_digits = len(_ALNUM_RE.findall(line))
    punctuation = len(_PUNCT_RE.findall(line))
    if len(line) <= 3 and letters_or_digits <= 2:
        return True
    if punctuation >= 4 and punctuation > letters_or_digits:
        return True
    if len(line) <= 24 and letters_or_digits <= 4:
        return True
    return False


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (len(ordered) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def normalize_whitespace(text: str) -> str:
    cleaned = unicodedata.normalize("NFC", _safe_text(text))
    cleaned = cleaned.replace("\ufeff", "").replace("\u00a0", " ")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    out_chars: list[str] = []
    for ch in cleaned:
        if ch in ("\n", "\t"):
            out_chars.append(ch)
            continue
        if unicodedata.category(ch).startswith("C"):
            continue
        out_chars.append(ch)
    cleaned = "".join(out_chars)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n"))
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def remove_html_artifacts(text: str) -> str:
    cleaned = _safe_text(text)
    for src, dst in (
        ("&nbsp;", " "),
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&quot;", '"'),
    ):
        cleaned = cleaned.replace(src, dst)
    cleaned = html.unescape(cleaned)
    return cleaned


def is_legal_significant_line(line: str) -> bool:
    stripped = normalize_whitespace(_safe_text(line))
    if not stripped:
        return False
    lowered = stripped.lower().replace("ё", "е")
    if any(term in lowered for term in _LEGAL_KEYWORDS):
        return True
    if any(term in lowered for term in _TAX_TERMS):
        return True
    if _DATE_DOTTED_RE.search(stripped) or _DATE_TEXT_RE.search(stripped):
        return True
    if _DOC_NUMBER_RE.search(stripped):
        return True
    if _ARTICLE_RE.search(stripped) or _POINT_RE.search(stripped):
        return True
    if _ENUM_LIST_RE.search(stripped):
        return True
    if _MONEY_RE.search(stripped):
        return True
    return False


def remove_navigation_noise(text: str) -> tuple[str, dict[str, Any]]:
    source_lines = _safe_text(text).splitlines()
    kept_lines: list[str] = []
    removed_lines: list[str] = []
    for line in source_lines:
        stripped = normalize_whitespace(line)
        if not stripped:
            kept_lines.append("")
            continue
        if is_legal_significant_line(stripped):
            kept_lines.append(stripped)
            continue
        if _is_navigation_noise_line(stripped.lower()):
            removed_lines.append(stripped)
            continue
        kept_lines.append(stripped)
    cleaned = normalize_whitespace("\n".join(kept_lines))
    stats = {
        "removed_navigation_line_count": len(removed_lines),
        "removed_navigation_samples": removed_lines[:10],
    }
    return cleaned, stats


def remove_repeated_lines(text: str) -> tuple[str, dict[str, Any]]:
    source_lines = _safe_text(text).splitlines()
    seen: set[str] = set()
    kept_lines: list[str] = []
    removed = 0
    for line in source_lines:
        stripped = normalize_whitespace(line)
        if not stripped:
            if kept_lines and kept_lines[-1] == "":
                continue
            kept_lines.append("")
            continue
        if is_legal_significant_line(stripped):
            kept_lines.append(stripped)
            continue
        key = _line_key(stripped)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        kept_lines.append(stripped)
    cleaned = normalize_whitespace("\n".join(kept_lines))
    return cleaned, {"removed_repeated_line_count": removed}


def remove_corpus_boilerplate(text: str, corpus_stats: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    source_lines = _safe_text(text).splitlines()
    boilerplate_lines = set(corpus_stats.get("boilerplate_lines") or [])
    kept_lines: list[str] = []
    removed = 0
    for line in source_lines:
        stripped = normalize_whitespace(line)
        if not stripped:
            kept_lines.append("")
            continue
        if is_legal_significant_line(stripped):
            kept_lines.append(stripped)
            continue
        if _line_key(stripped) in boilerplate_lines:
            removed += 1
            continue
        kept_lines.append(stripped)
    cleaned = normalize_whitespace("\n".join(kept_lines))
    return cleaned, {"removed_corpus_boilerplate_line_count": removed}


def build_corpus_boilerplate_stats(documents: Sequence[Any]) -> dict[str, Any]:
    doc_line_counts: dict[str, int] = {}
    total_docs = 0
    for doc in documents:
        text = _safe_text(getattr(doc, "text", None))
        if not text.strip():
            continue
        total_docs += 1
        seen_in_doc: set[str] = set()
        for line in text.splitlines():
            stripped = normalize_whitespace(line)
            if not stripped or len(stripped) > 120:
                continue
            if is_legal_significant_line(stripped):
                continue
            seen_in_doc.add(_line_key(stripped))
        for key in seen_in_doc:
            doc_line_counts[key] = doc_line_counts.get(key, 0) + 1

    if total_docs <= 0:
        return {
            "total_docs": 0,
            "boilerplate_threshold_ratio": 0.05,
            "boilerplate_threshold_docs": 0,
            "boilerplate_lines": [],
            "boilerplate_line_count": 0,
        }

    threshold_ratio = 0.05
    threshold_docs_float = total_docs * threshold_ratio
    boilerplate = [
        key
        for key, count in doc_line_counts.items()
        if count > threshold_docs_float
    ]
    boilerplate.sort(key=lambda line: doc_line_counts.get(line, 0), reverse=True)
    return {
        "total_docs": total_docs,
        "boilerplate_threshold_ratio": threshold_ratio,
        "boilerplate_threshold_docs": max(1, int(threshold_docs_float)),
        "boilerplate_lines": boilerplate,
        "boilerplate_line_count": len(boilerplate),
    }


def _clean_basic(text: str) -> tuple[str, dict[str, Any]]:
    html_cleaned = remove_html_artifacts(text)
    normalized = normalize_whitespace(html_cleaned)
    return normalized, {"profile_step": "clean_basic"}


def _clean_legal(text: str) -> tuple[str, dict[str, Any]]:
    basic_text, basic_stats = _clean_basic(text)
    no_nav_text, nav_stats = remove_navigation_noise(basic_text)
    stats = {"basic": basic_stats, "navigation": nav_stats}
    return no_nav_text, stats


def _clean_aggressive(text: str) -> tuple[str, dict[str, Any]]:
    legal_text, legal_stats = _clean_legal(text)
    source_lines = legal_text.splitlines()
    filtered: list[str] = []
    removed_aggressive = 0
    for line in source_lines:
        stripped = normalize_whitespace(line)
        if not stripped:
            filtered.append("")
            continue
        if is_legal_significant_line(stripped):
            filtered.append(stripped)
            continue
        if _is_navigation_noise_line(stripped.lower()) or _is_aggressive_noise_line(stripped):
            removed_aggressive += 1
            continue
        filtered.append(stripped)
    filtered_text = normalize_whitespace("\n".join(filtered))
    deduped_text, dedupe_stats = remove_repeated_lines(filtered_text)
    stats = {
        "legal": legal_stats,
        "removed_aggressive_noise_line_count": removed_aggressive,
        "dedupe": dedupe_stats,
    }
    return deduped_text, stats


def clean_text(text: str, profile: str, corpus_stats: dict[str, Any] | None = None) -> CleaningResult:
    source = _safe_text(text)
    profile_name = str(profile or "raw")
    if profile_name not in SUPPORTED_CLEANING_PROFILES:
        raise ValueError(
            f"Unsupported cleaning profile {profile_name!r}. "
            f"Expected one of: {', '.join(SUPPORTED_CLEANING_PROFILES)}"
        )

    if profile_name == "raw":
        cleaned = source.strip()
        stats: dict[str, Any] = {"profile_step": "raw"}
    elif profile_name == "clean_basic":
        cleaned, stats = _clean_basic(source)
    elif profile_name == "clean_legal":
        cleaned, stats = _clean_legal(source)
    elif profile_name == "clean_aggressive":
        cleaned, stats = _clean_aggressive(source)
    else:
        legal_text, legal_stats = _clean_legal(source)
        no_boilerplate_text, boilerplate_stats = remove_corpus_boilerplate(legal_text, corpus_stats or {})
        cleaned = no_boilerplate_text
        stats = {
            "legal": legal_stats,
            "boilerplate": boilerplate_stats,
            "boilerplate_profile_stats": {
                "total_docs": int((corpus_stats or {}).get("total_docs") or 0),
                "boilerplate_line_count": int((corpus_stats or {}).get("boilerplate_line_count") or 0),
            },
        }

    original_char_len = len(source)
    cleaned_char_len = len(cleaned)
    if original_char_len <= 0:
        removed_char_ratio = 0.0
    else:
        removed_char_ratio = max(0.0, (original_char_len - cleaned_char_len) / original_char_len)
    removed_line_count = max(0, len(source.splitlines()) - len(cleaned.splitlines()))

    warnings: list[str] = []
    if removed_char_ratio > 0.5:
        warnings.append("High removed_char_ratio; cleaning may be too aggressive.")
    if cleaned_char_len < 200:
        warnings.append("Cleaned text is very short.")

    stats.update(
        {
            "original_line_count": len(source.splitlines()),
            "cleaned_line_count": len(cleaned.splitlines()),
            "removed_char_p95_guardrail": _percentile([removed_char_ratio], 0.95),
        }
    )

    return CleaningResult(
        text=cleaned,
        profile=profile_name,
        original_char_len=original_char_len,
        cleaned_char_len=cleaned_char_len,
        removed_char_ratio=removed_char_ratio,
        removed_line_count=removed_line_count,
        warning_count=len(warnings),
        warnings=warnings,
        stats=stats,
    )
