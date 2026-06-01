#!/usr/bin/env python3
from __future__ import annotations

"""
Smoke-test embedding backends without Qdrant or a local corpus.

Examples:
python scripts/smoke_test_embedders.py \
  --embedders BAAI/bge-m3 intfloat/multilingual-e5-base deepvk/USER-base \
  --device cuda
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rag.embedders import EMBEDDER_REGISTRY, HEAVY_EMBEDDERS, config_from_registry, build_embedder


QUERY = "какой срок подачи уведомления по НДФЛ при выплате дивидендов"
POSITIVE = "Уведомление об исчисленных суммах НДФЛ подается налоговым агентом в установленный срок при выплате доходов, включая дивиденды."
NEGATIVE = "Налог на имущество организаций рассчитывается исходя из кадастровой стоимости объекта недвижимости."


def _score(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.reshape(-1), b.reshape(-1)))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Smoke-test embedding model loading and query/passage encoding.", epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedders", nargs="+", default=list(EMBEDDER_REGISTRY))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-seq-length", type=int, default=None)
    ap.add_argument("--no-heavy", action="store_true")
    return ap


def _run_one(model_name: str, args: argparse.Namespace) -> dict[str, Any]:
    row: dict[str, Any] = {"embedder": model_name, "status": "ok"}
    try:
        cfg = config_from_registry(
            model_name,
            device=args.device,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            backend=args.backend,
        )
        embedder = build_embedder(cfg)
        q = embedder.encode_queries([QUERY])
        p = embedder.encode_passages([POSITIVE])
        n = embedder.encode_passages([NEGATIVE])
        pos = _score(q[0], p[0])
        neg = _score(q[0], n[0])
        row.update(
            {
                "dim": int(q.shape[1]),
                "positive_score": pos,
                "negative_score": neg,
                "passed": bool(pos > neg),
                "metadata": getattr(embedder, "metadata", {}),
            }
        )
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "passed": False,
                "error": str(exc),
                "traceback": traceback.format_exc()[-3000:],
            }
        )
    return row


def main() -> None:
    args = build_parser().parse_args()
    embedders = [m for m in args.embedders if not (args.no_heavy and m in HEAVY_EMBEDDERS)]
    for model_name in embedders:
        row = _run_one(model_name, args)
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()

