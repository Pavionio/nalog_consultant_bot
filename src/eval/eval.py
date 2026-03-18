from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.rag.core import RAGConfig, STEmbedder, Retriever, LlamaCppChatClient, build_context, rag_answer, rewrite_query
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
    source_code: str = ""


def evaluate_dataset(
    dataset_path: str,
    *,
    k: Optional[int] = None,
    use_llm_judge: bool = True,
    use_rewrite: bool = False,
    use_hyde: bool = False,
    use_reranker: bool = False,
    # pre-loaded components (pass to avoid reloading across runs)
    embedder=None,
    retriever=None,
    llm=None,
    # precomputed query transformations {original_query: transformed_query}
    precomputed_queries: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    import dataclasses
    cfg = RAGConfig()
    cfg = dataclasses.replace(cfg, use_rewrite=use_rewrite, use_hyde=use_hyde, use_reranker=use_reranker)
    if k is None:
        k = cfg.top_k

    if embedder is None:
        embedder = STEmbedder(cfg.embed_model_name)
    if retriever is None:
        retriever = Retriever(cfg, embedder)
    else:
        # update flags on existing retriever's config
        retriever.cfg = cfg
    if llm is None:
        llm = LlamaCppChatClient(cfg)

    from tqdm import tqdm

    ds = load_jsonl(dataset_path)

    per_item: List[EvalResult] = []

    for item in tqdm(ds, desc="eval", unit="q"):
        qid = str(item.get("id", ""))
        query = str(item["query"])
        relevant = item.get("relevant") or []
        relevant_total = len(relevant)
        source_code = relevant[0].get("source_code", "") if relevant else ""

        # retrieval only
        from src.rag.core import hyde_query
        if precomputed_queries and query in precomputed_queries:
            search_query = precomputed_queries[query]
        elif use_hyde:
            search_query = hyde_query(llm, query)
        elif use_rewrite:
            search_query = rewrite_query(llm, query)
        else:
            search_query = query
        retrieved = retriever.search(search_query)
        bin_rels = _binary_relevance_list(retrieved, relevant)

        p = precision_at_k(bin_rels, k)
        r = recall_at_k(bin_rels, relevant_total, k)
        mrr = mrr_at_k(bin_rels, k)
        nd = ndcg_at_k(bin_rels, k)

        pres = val = dens = 0.0
        jf = jr = None
        jn = ""

        if use_llm_judge:
            context, sources = build_context(cfg, retrieved)
            out = rag_answer(cfg, retriever, llm, query, chat_history=None, chunks=retrieved)
            answer = out["answer"]

            pres = citation_presence(answer)
            val = citation_validity(answer, n_sources=len(sources))
            dens = citation_density(answer)

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
                source_code=source_code,
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
        f_vals = [x.judge_faithfulness for x in per_item if isinstance(x.judge_faithfulness, int)]
        r_vals = [x.judge_relevance for x in per_item if isinstance(x.judge_relevance, int)]
        metrics["judge_faithfulness_avg_0_2"] = avg([float(v) for v in f_vals]) if f_vals else None
        metrics["judge_relevance_avg_0_2"] = avg([float(v) for v in r_vals]) if r_vals else None

    # per-source breakdown
    from collections import defaultdict
    by_source: Dict[str, List[EvalResult]] = defaultdict(list)
    for x in per_item:
        by_source[x.source_code or "unknown"].append(x)

    per_source = {}
    for sc, items in sorted(by_source.items()):
        per_source[sc] = {
            "n": len(items),
            f"hit_rate@{k}": avg([x.r_at_k for x in items]),
            f"mrr@{k}":      avg([x.mrr   for x in items]),
            f"ndcg@{k}":     avg([x.ndcg  for x in items]),
        }
    metrics["per_source"] = per_source

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


EVAL_LOG = "data/metrics/eval_log.jsonl"
EVAL_LOG_SOURCE = "data/metrics/eval_log_per_source.jsonl"


def _append_log(row: Dict[str, Any], log_path: str = EVAL_LOG) -> None:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_log(log_path: str = EVAL_LOG) -> List[Dict[str, Any]]:
    p = Path(log_path)
    if not p.exists():
        return []
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    return rows


def _print_log_table(log_path: str = EVAL_LOG) -> None:
    rows = _load_log(log_path)
    if not rows:
        return

    # Columns to display
    cols = [
        ("timestamp",       "time",         19),
        ("model",           "model",        28),
        ("dataset",         "dataset",      24),
        ("k",               "k",             3),
        ("n",               "n",             5),
        ("hit_rate",        "hit@k",         6),
        ("mrr",             "mrr@k",         6),
        ("ndcg",            "ndcg@k",        7),
        ("precision",       "prec@k",        7),
        ("judge_faith",     "faith",         5),
        ("judge_rel",       "rel",           5),
    ]

    def cell(row: Dict, key: str, width: int) -> str:
        v = row.get(key)
        if v is None:
            s = "-"
        elif isinstance(v, float):
            s = f"{v:.3f}"
        else:
            s = str(v)
        return s[:width].ljust(width)

    header = "  ".join(label.ljust(w) for _, label, w in cols)
    sep    = "  ".join("-" * w for _, _, w in cols)
    print("\n=== eval log ===")
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(cell(row, key, w) for key, _, w in cols))
    print()


def main() -> None:
    import argparse
    import datetime

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to eval_dataset.jsonl")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--no-judge", action="store_true", help="Disable LLM-as-judge scoring")
    ap.add_argument("--rewrite", action="store_true", help="Enable query rewriting before retrieval")
    ap.add_argument("--hyde", action="store_true", help="Use HyDE: generate hypothetical document passage for retrieval")
    ap.add_argument("--reranker", action="store_true", help="Use cross-encoder reranker (BAAI/bge-reranker-v2-m3)")
    ap.add_argument("--out", default="eval_report.json", help="Output JSON report")
    ap.add_argument("--model", default="", help="Short description of model/config for the log (e.g. 'Qwen3-8B bge-m3 top6')")
    ap.add_argument("--log", default=EVAL_LOG, help=f"Path to eval log file (default: {EVAL_LOG})")
    args = ap.parse_args()

    report = evaluate_dataset(args.dataset, k=args.k, use_llm_judge=not args.no_judge, use_rewrite=args.rewrite, use_hyde=args.hyde, use_reranker=args.reranker)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    m = report["metrics"]
    k_used = args.k or RAGConfig().top_k
    n = len(report["detailed"])

    log_row: Dict[str, Any] = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model":     args.model or "-",
        "dataset":   Path(args.dataset).name,
        "k":         k_used,
        "n":         n,
        "hit_rate":  m.get(f"recall@{k_used}"),
        "mrr":       m.get(f"mrr@{k_used}"),
        "ndcg":      m.get(f"ndcg@{k_used}"),
        "precision": m.get(f"precision@{k_used}"),
        "judge_faith": m.get("judge_faithfulness_avg_0_2"),
        "judge_rel":   m.get("judge_relevance_avg_0_2"),
    }
    _append_log(log_row, args.log)

    # per-source log: one row per source_code
    if m.get("per_source"):
        source_log = Path(args.log).parent / Path(EVAL_LOG_SOURCE).name
        for sc, sm in m["per_source"].items():
            _append_log({
                "timestamp": log_row["timestamp"],
                "model":     log_row["model"],
                "dataset":   log_row["dataset"],
                "k":         k_used,
                "source_code": sc,
                **sm,
            }, str(source_log))

    # print overall metrics (without per_source noise)
    m_print = {kk: vv for kk, vv in m.items() if kk != "per_source"}
    print(json.dumps(m_print, ensure_ascii=False, indent=2))

    # per-source table
    if m.get("per_source"):
        k_used = args.k or RAGConfig().top_k
        print(f"\n--- per source (hit_rate@{k_used}) ---")
        rows = sorted(m["per_source"].items(), key=lambda x: x[1].get(f"hit_rate@{k_used}", 0))
        for sc, sm in rows:
            bar = "█" * int(sm.get(f"hit_rate@{k_used}", 0) * 20)
            print(f"  {sc:<28} n={sm['n']:<4} hit={sm.get(f'hit_rate@{k_used}', 0):.3f}  {bar}")

    print(f"\nSaved report to: {args.out}")

    _print_log_table(args.log)


if __name__ == "__main__":
    main()
