from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.rag.core import RAGConfig, STEmbedder, Retriever, LlamaCppChatClient, build_context, rag_answer
# мб не работает, не запускал еще

# ----------------------------
# Dataset loading
# ----------------------------

def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


# ----------------------------
# Retrieval metrics
# ----------------------------

def _match_rel(rel: Dict[str, Any], cand_payload: Dict[str, Any]) -> bool:
    """
    Определяет, совпадает ли кандидат с релевантным таргетом.
    Поддерживает doc_id / external_id / url / chunk_i.
    """
    # candidate fields
    c_doc_id = cand_payload.get("doc_id")
    c_external_id = cand_payload.get("external_id")
    c_url = cand_payload.get("url") or cand_payload.get("source_url") or cand_payload.get("doc_url")
    c_chunk_i = cand_payload.get("chunk_i")

    # relevant fields
    r_doc_id = rel.get("doc_id")
    r_external_id = rel.get("external_id")
    r_url = rel.get("url")
    r_chunk_i = rel.get("chunk_i")

    # Matching logic: if a key is specified in rel, it must match.
    if r_doc_id is not None and c_doc_id != r_doc_id:
        return False
    if r_external_id is not None and c_external_id != r_external_id:
        return False
    if r_url is not None and c_url != r_url:
        return False
    if r_chunk_i is not None and c_chunk_i != r_chunk_i:
        return False

    # if rel has no keys (bad rel), treat as non-match
    if all(rel.get(k) is None for k in ("doc_id", "external_id", "url", "chunk_i")):
        return False

    return True


def _binary_relevance_list(
    retrieved: List[Dict[str, Any]],
    relevant: List[Dict[str, Any]],
) -> List[int]:
    """
    Returns a list of 0/1 for each retrieved item whether it matches ANY relevant target.
    """
    rels: List[int] = []
    for r in retrieved:
        payload = r.get("payload") or {}
        hit = any(_match_rel(gt, payload) for gt in relevant)
        rels.append(1 if hit else 0)
    return rels


def precision_at_k(bin_rels: List[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = bin_rels[:k]
    if not top:
        return 0.0
    return sum(top) / len(top)


def recall_at_k(bin_rels: List[int], relevant_total: int, k: int) -> float:
    if relevant_total <= 0:
        return 0.0
    return min(1.0, sum(bin_rels[:k]) / relevant_total)


def mrr_at_k(bin_rels: List[int], k: int) -> float:
    for i, rel in enumerate(bin_rels[:k], start=1):
        if rel == 1:
            return 1.0 / i
    return 0.0


def ndcg_at_k(bin_rels: List[int], k: int) -> float:
    """
    Binary nDCG@k with log2 discount.
    """
    def dcg(rels: List[int]) -> float:
        s = 0.0
        for i, r in enumerate(rels[:k], start=1):
            if r:
                s += 1.0 / math.log2(i + 1)
        return s

    actual = dcg(bin_rels)
    ideal = dcg(sorted(bin_rels, reverse=True))
    if ideal == 0.0:
        return 0.0
    return actual / ideal


# ----------------------------
# Generation metrics (heuristics + LLM judge)
# ----------------------------

CIT_PATTERN = re.compile(r"\[(\d+)\]")

def extract_citations(answer: str) -> List[int]:
    return [int(x) for x in CIT_PATTERN.findall(answer)]


def citation_validity(answer: str, n_sources: int) -> float:
    cites = extract_citations(answer)
    if not cites:
        return 0.0
    ok = sum(1 for c in cites if 1 <= c <= n_sources)
    return ok / len(cites)


def citation_presence(answer: str) -> float:
    # simple: 1 if has any [n], else 0
    return 1.0 if extract_citations(answer) else 0.0


def citation_density(answer: str) -> float:
    # citations per 1000 chars
    cites = extract_citations(answer)
    if not answer:
        return 0.0
    return len(cites) * 1000.0 / max(1, len(answer))


JUDGE_SYSTEM = """Ты — строгий оценщик качества ответа RAG-системы.
Оценивай ТОЛЬКО по вопросу пользователя и предоставленному контексту (выдержки источников).
Нельзя домысливать факты вне контекста.

Верни JSON строго следующего вида:
{
  "faithfulness": <0|1|2>,     // 0 = есть недоказанные утверждения; 1 = сомнительно; 2 = все утверждения опираются на контекст
  "relevance": <0|1|2>,        // 0 = не отвечает; 1 = частично; 2 = отвечает по сути
  "notes": "<коротко, 1-3 предложения>"
}
"""

def judge_with_llm(
    llm: LlamaCppChatClient,
    question: str,
    context: str,
    answer: str,
) -> Dict[str, Any]:
    msgs = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"Вопрос:\n{question}\n\nКонтекст:\n{context}\n\nОтвет:\n{answer}\n\nВерни JSON:"},
    ]
    raw = llm.chat(msgs)

    # попытка вытащить JSON, даже если LLM обрамил текстом
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return {"faithfulness": None, "relevance": None, "notes": f"Judge parse error: {raw[:200]}"}

    try:
        return json.loads(m.group(0))
    except Exception:
        return {"faithfulness": None, "relevance": None, "notes": f"Judge JSON error: {m.group(0)[:200]}"}


# ----------------------------
# Runner
# ----------------------------

@dataclass
class EvalResult:
    id: str
    query: str

    # retrieval
    p_at_k: float
    r_at_k: float
    mrr: float
    ndcg: float

    # generation
    cite_presence: float
    cite_validity: float
    cite_density: float
    judge_faithfulness: Optional[int]
    judge_relevance: Optional[int]
    judge_notes: str

    retrieved_count: int


def evaluate_dataset(
    dataset_path: str,
    *,
    k: Optional[int] = None,
    use_llm_judge: bool = True,
) -> Dict[str, Any]:
    cfg = RAGConfig()
    if k is None:
        k = cfg.top_k

    embedder = STEmbedder(cfg.embed_model_name)
    retriever = Retriever(cfg, embedder)
    llm = LlamaCppChatClient(cfg)

    ds = load_jsonl(dataset_path)

    per_item: List[EvalResult] = []

    for item in ds:
        qid = str(item.get("id", ""))
        query = str(item["query"])
        relevant = item.get("relevant") or []
        relevant_total = len(relevant)

        # retrieval only
        retrieved = retriever.search(query)
        bin_rels = _binary_relevance_list(retrieved, relevant)

        p = precision_at_k(bin_rels, k)
        r = recall_at_k(bin_rels, relevant_total, k)
        mrr = mrr_at_k(bin_rels, k)
        nd = ndcg_at_k(bin_rels, k)

        # generation
        # To evaluate generation, reuse your context builder (same as prod)
        context, sources = build_context(cfg, retrieved)
        out = rag_answer(cfg, retriever, llm, query, chat_history=None)
        answer = out["answer"]

        pres = citation_presence(answer)
        val = citation_validity(answer, n_sources=len(sources))
        dens = citation_density(answer)

        jf = None
        jr = None
        jn = ""

        if use_llm_judge:
            j = judge_with_llm(llm, query, context, answer)
            jf = j.get("faithfulness")
            jr = j.get("relevance")
            jn = str(j.get("notes", ""))

        per_item.append(
            EvalResult(
                id=qid,
                query=query,
                p_at_k=p,
                r_at_k=r,
                mrr=mrr,
                ndcg=nd,
                cite_presence=pres,
                cite_validity=val,
                cite_density=dens,
                judge_faithfulness=jf,
                judge_relevance=jr,
                judge_notes=jn,
                retrieved_count=len(retrieved),
            )
        )

    # aggregate
    def avg(xs: List[float]) -> float:
        xs2 = [x for x in xs if x is not None]
        return sum(xs2) / max(1, len(xs2))

    metrics = {
        f"precision@{k}": avg([x.p_at_k for x in per_item]),
        f"recall@{k}": avg([x.r_at_k for x in per_item]),
        f"mrr@{k}": avg([x.mrr for x in per_item]),
        f"ndcg@{k}": avg([x.ndcg for x in per_item]),
        "citation_presence": avg([x.cite_presence for x in per_item]),
        "citation_validity": avg([x.cite_validity for x in per_item]),
        "citation_density_per_1k_chars": avg([x.cite_density for x in per_item]),
    }

    if use_llm_judge:
        # faithfulness/relevance are 0..2, average them
        f_vals = [x.judge_faithfulness for x in per_item if isinstance(x.judge_faithfulness, int)]
        r_vals = [x.judge_relevance for x in per_item if isinstance(x.judge_relevance, int)]
        metrics["judge_faithfulness_avg_0_2"] = avg([float(v) for v in f_vals]) if f_vals else None
        metrics["judge_relevance_avg_0_2"] = avg([float(v) for v in r_vals]) if r_vals else None

    # dump detailed
    detailed = [
        {
            "id": x.id,
            "query": x.query,
            f"precision@{k}": x.p_at_k,
            f"recall@{k}": x.r_at_k,
            f"mrr@{k}": x.mrr,
            f"ndcg@{k}": x.ndcg,
            "citation_presence": x.cite_presence,
            "citation_validity": x.cite_validity,
            "citation_density_per_1k_chars": x.cite_density,
            "judge_faithfulness": x.judge_faithfulness,
            "judge_relevance": x.judge_relevance,
            "judge_notes": x.judge_notes,
            "retrieved_count": x.retrieved_count,
        }
        for x in per_item
    ]

    return {"metrics": metrics, "detailed": detailed}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to eval_dataset.jsonl")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--no-judge", action="store_true", help="Disable LLM-as-judge scoring")
    ap.add_argument("--out", default="eval_report.json", help="Output JSON report")
    args = ap.parse_args()

    report = evaluate_dataset(args.dataset, k=args.k, use_llm_judge=not args.no_judge)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    print(f"\nSaved report to: {args.out}")


if __name__ == "__main__":
    main()
