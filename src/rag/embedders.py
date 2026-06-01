from __future__ import annotations

import inspect
import os
import warnings
from dataclasses import dataclass, replace
from typing import Any, Optional, Sequence

import numpy as np


QWEN3_QUERY_INSTRUCTION = (
    "Given a Russian tax/legal user query, retrieve official Russian tax document "
    "passages that directly answer the query."
)
E5_MISTRAL_QUERY_INSTRUCTION = (
    "Given a Russian tax question, retrieve official Russian tax law and tax authority "
    "passages that answer the question."
)
GIGA_QUERY_INSTRUCTION = (
    "Найди официальный фрагмент налогового документа, который отвечает на вопрос пользователя."
)


EMBEDDER_REGISTRY: dict[str, dict[str, Any]] = {
    "BAAI/bge-m3": {
        "short_name": "bge_m3",
        "backend": "sentence_transformers",
        "trust_remote_code": False,
        "normalize": True,
        "max_seq_length": 8192,
        "batch_size": 32,
    },
    "intfloat/multilingual-e5-base": {
        "short_name": "e5_base",
        "backend": "sentence_transformers",
        "trust_remote_code": False,
        "normalize": True,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "max_seq_length": 512,
        "batch_size": 64,
    },
    "intfloat/multilingual-e5-large": {
        "short_name": "e5_large",
        "backend": "sentence_transformers",
        "trust_remote_code": False,
        "normalize": True,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "max_seq_length": 512,
        "batch_size": 32,
    },
    "ai-forever/sbert_large_nlu_ru": {
        "short_name": "sbert_large_nlu_ru",
        "backend": "sentence_transformers_or_mean_pooling",
        "trust_remote_code": False,
        "normalize": True,
        "pooling": "mean",
        "batch_size": 32,
    },
    "ai-sage/Giga-Embeddings-instruct": {
        "short_name": "giga_embeddings_instruct",
        "backend": "sentence_transformers",
        "trust_remote_code": True,
        "normalize": True,
        "query_instruction": GIGA_QUERY_INSTRUCTION,
        "batch_size": 16,
    },
    "Qwen/Qwen3-Embedding-4B": {
        "short_name": "qwen3_embedding_4b",
        "backend": "sentence_transformers",
        "trust_remote_code": True,
        "normalize": True,
        "query_prompt_name": "query",
        "query_instruction": QWEN3_QUERY_INSTRUCTION,
        "max_seq_length": 8192,
        "batch_size": 4,
    },
    "Qwen/Qwen3-Embedding-8B": {
        "short_name": "qwen3_embedding_8b",
        "backend": "sentence_transformers",
        "trust_remote_code": True,
        "normalize": True,
        "query_prompt_name": "query",
        "query_instruction": QWEN3_QUERY_INSTRUCTION,
        "max_seq_length": 8192,
        "batch_size": 2,
    },
    "jinaai/jina-embeddings-v3": {
        "short_name": "jina_embeddings_v3",
        "backend": "jina",
        "trust_remote_code": True,
        "normalize": True,
        "query_prompt_name": "retrieval.query",
        "passage_prompt_name": "retrieval.passage",
        "max_seq_length": 8192,
        "batch_size": 16,
    },
    "deepvk/USER-base": {
        "short_name": "user_base",
        "backend": "sentence_transformers",
        "trust_remote_code": False,
        "normalize": True,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "batch_size": 64,
    },
    "intfloat/e5-mistral-7b-instruct": {
        "short_name": "e5_mistral_7b_instruct",
        "backend": "sentence_transformers",
        "trust_remote_code": False,
        "normalize": True,
        "query_prompt_name": "web_search_query",
        "query_instruction": E5_MISTRAL_QUERY_INSTRUCTION,
        "max_seq_length": 4096,
        "batch_size": 2,
    },
}


HEAVY_EMBEDDERS = {
    "ai-sage/Giga-Embeddings-instruct",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
    "intfloat/e5-mistral-7b-instruct",
}


@dataclass
class EmbedderConfig:
    model_name: str
    device: str = "cuda"
    batch_size: int = 32
    max_seq_length: int | None = None
    normalize_embeddings: bool = True
    trust_remote_code: bool = False
    query_prompt: str | None = None
    passage_prompt: str | None = None
    query_prompt_name: str | None = None
    passage_prompt_name: str | None = None
    query_prefix: str | None = None
    passage_prefix: str | None = None
    query_instruction: str | None = None
    passage_instruction: str | None = None
    backend: str = "auto"
    dtype: str = "auto"
    attn_implementation: str | None = None


class BaseEmbedder:
    config: EmbedderConfig
    metadata: dict[str, Any]

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_passages(self, passages: list[str]) -> np.ndarray:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        raise NotImplementedError


def registry_entry(model_name: str) -> dict[str, Any]:
    return dict(EMBEDDER_REGISTRY.get(model_name, {}))


def embedder_short_name(model_name: str) -> str:
    entry = EMBEDDER_REGISTRY.get(model_name)
    if entry and entry.get("short_name"):
        return str(entry["short_name"])
    return (
        model_name.rsplit("/", 1)[-1]
        .lower()
        .replace("-", "_")
        .replace(".", "_")
        .replace("@", "_")
    )


def chunking_tag(
    *,
    chunk_method: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    parent_chunk_size: int = 3072,
    child_chunk_size: int = 768,
) -> str:
    if chunk_method == "parent_child":
        return f"parent{parent_chunk_size}_child{child_chunk_size}"
    return f"{chunk_method}_{chunk_size}_{chunk_overlap}"


def collection_name_for_embedder(
    model_name: str,
    *,
    chunk_method: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    parent_chunk_size: int = 3072,
    child_chunk_size: int = 768,
    prefix: str = "rag_chunks",
) -> str:
    return (
        f"{prefix}_{embedder_short_name(model_name)}_"
        f"{chunking_tag(chunk_method=chunk_method, chunk_size=chunk_size, chunk_overlap=chunk_overlap, parent_chunk_size=parent_chunk_size, child_chunk_size=child_chunk_size)}"
    )


def config_from_registry(
    model_name: str,
    *,
    device: str | None = None,
    batch_size: int | None = None,
    max_seq_length: int | None = None,
    normalize_embeddings: bool | None = None,
    trust_remote_code: bool | None = None,
    backend: str | None = None,
    query_prefix: str | None = None,
    passage_prefix: str | None = None,
    query_instruction: str | None = None,
    passage_instruction: str | None = None,
    query_prompt_name: str | None = None,
    passage_prompt_name: str | None = None,
    dtype: str | None = None,
    attn_implementation: str | None = None,
) -> EmbedderConfig:
    entry = registry_entry(model_name)
    cfg = EmbedderConfig(
        model_name=model_name,
        device=device or os.getenv("EMBED_DEVICE", "cuda"),
        batch_size=int(batch_size or entry.get("batch_size") or os.getenv("EMBED_BATCH_SIZE", "32")),
        max_seq_length=max_seq_length if max_seq_length is not None else entry.get("max_seq_length"),
        normalize_embeddings=bool(entry.get("normalize", True)),
        trust_remote_code=bool(entry.get("trust_remote_code", False)),
        query_prompt_name=entry.get("query_prompt_name"),
        passage_prompt_name=entry.get("passage_prompt_name"),
        query_prefix=entry.get("query_prefix"),
        passage_prefix=entry.get("passage_prefix"),
        query_instruction=entry.get("query_instruction"),
        passage_instruction=entry.get("passage_instruction"),
        backend=str(entry.get("backend") or "auto"),
        dtype=dtype or "auto",
        attn_implementation=attn_implementation or os.getenv("EMBED_ATTN_IMPLEMENTATION") or "eager",
    )
    if normalize_embeddings is not None:
        cfg.normalize_embeddings = normalize_embeddings
    if trust_remote_code is not None:
        cfg.trust_remote_code = trust_remote_code
    if backend:
        cfg.backend = backend
    if query_prefix is not None:
        cfg.query_prefix = query_prefix or None
    if passage_prefix is not None:
        cfg.passage_prefix = passage_prefix or None
    if query_instruction is not None:
        cfg.query_instruction = query_instruction or None
    if passage_instruction is not None:
        cfg.passage_instruction = passage_instruction or None
    if query_prompt_name is not None:
        cfg.query_prompt_name = query_prompt_name or None
    if passage_prompt_name is not None:
        cfg.passage_prompt_name = passage_prompt_name or None
    return cfg


def _as_texts(texts: Sequence[str]) -> list[str]:
    return [str(t or "") for t in texts]


def _normalize_np(arr: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(arr, axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return arr / denom


def _float32_2d(value: Any, *, normalize: bool = False) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if normalize:
        arr = _normalize_np(arr)
    return arr.astype(np.float32, copy=False)


def _has_encode_kw(model: Any, key: str) -> bool:
    try:
        sig = inspect.signature(model.encode)
    except Exception:
        return True
    return key in sig.parameters or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())


class SentenceTransformersEmbedder(BaseEmbedder):
    def __init__(self, config: EmbedderConfig) -> None:
        self.config = config
        self.metadata: dict[str, Any] = {
            "backend": "sentence_transformers",
            "model_name": config.model_name,
            "query_mode": "raw",
            "passage_mode": "raw",
        }
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install sentence-transformers: pip install -U sentence-transformers") from exc

        kwargs: dict[str, Any] = {
            "device": config.device,
            "trust_remote_code": config.trust_remote_code,
        }
        model_kwargs: dict[str, Any] = {}
        if config.attn_implementation:
            model_kwargs["attn_implementation"] = config.attn_implementation
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs
        token = os.getenv("HF_TOKEN")
        if token:
            kwargs["token"] = token
        try:
            self.model = SentenceTransformer(config.model_name, **kwargs)
        except TypeError:
            kwargs.pop("model_kwargs", None)
            self.model = SentenceTransformer(config.model_name, **kwargs)
        except Exception as exc:
            extra = ""
            if config.model_name.startswith("Qwen/Qwen3-Embedding"):
                extra = (
                    " Qwen3 embedding models may require recent transformers and "
                    "sentence-transformers. Try: pip install -U transformers sentence-transformers accelerate"
                )
            if "trust_remote_code" in str(exc) and not config.trust_remote_code:
                extra += " This model requires trust_remote_code=True."
            raise RuntimeError(f"Failed to load embedding model {config.model_name!r}.{extra}") from exc

        if config.max_seq_length:
            try:
                self.model.max_seq_length = int(config.max_seq_length)
            except Exception:
                pass

    @property
    def dim(self) -> int:
        try:
            dim = self.model.get_sentence_embedding_dimension()
            if dim:
                return int(dim)
        except Exception:
            pass
        return int(self.encode_passages(["dimension probe"]).shape[1])

    def _manual_texts(self, texts: list[str], mode: str) -> tuple[list[str], str]:
        if mode == "query":
            if self.config.query_prompt:
                return [self.config.query_prompt.format(query=t, text=t) for t in texts], "query_prompt"
            if self.config.query_instruction:
                if self.config.model_name.startswith("Qwen/Qwen3-Embedding"):
                    prefix = f"Instruct: {self.config.query_instruction}\nQuery: "
                elif self.config.model_name == "intfloat/e5-mistral-7b-instruct":
                    prefix = f"Instruct: {self.config.query_instruction}\nQuery: "
                else:
                    prefix = f"{self.config.query_instruction}\n"
                return [prefix + t for t in texts], "query_instruction"
            if self.config.query_prefix:
                return [self.config.query_prefix + t for t in texts], "query_prefix"
            return texts, "raw"
        if self.config.passage_prompt:
            return [self.config.passage_prompt.format(passage=t, text=t) for t in texts], "passage_prompt"
        if self.config.passage_instruction:
            return [f"{self.config.passage_instruction}\n{t}" for t in texts], "passage_instruction"
        if self.config.passage_prefix:
            return [self.config.passage_prefix + t for t in texts], "passage_prefix"
        return texts, "raw"

    def _encode_plain(self, texts: list[str], **extra: Any) -> np.ndarray:
        try:
            try:
                import torch
            except Exception:
                torch = None
            if torch is not None:
                with torch.inference_mode():
                    embs = self.model.encode(
                        texts,
                        batch_size=self.config.batch_size,
                        normalize_embeddings=self.config.normalize_embeddings,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        **extra,
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            else:
                embs = self.model.encode(
                    texts,
                    batch_size=self.config.batch_size,
                    normalize_embeddings=self.config.normalize_embeddings,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    **extra,
                )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    f"CUDA OOM while encoding with {self.config.model_name!r}. "
                    "Reduce --embed-batch-size or use --no-heavy."
                ) from exc
            raise
        arr = _float32_2d(embs)
        del embs
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return arr

    def _encode_mode(self, texts: Sequence[str], mode: str) -> np.ndarray:
        clean = _as_texts(texts)
        if not clean:
            return np.empty((0, self.dim), dtype=np.float32)

        prompt_name = self.config.query_prompt_name if mode == "query" else self.config.passage_prompt_name
        if prompt_name and _has_encode_kw(self.model, "prompt_name"):
            try:
                out = self._encode_plain(clean, prompt_name=prompt_name)
                self.metadata[f"{mode}_mode"] = f"prompt_name:{prompt_name}"
                return out
            except (TypeError, ValueError, KeyError):
                pass

        manual, manual_mode = self._manual_texts(clean, mode)
        self.metadata[f"{mode}_mode"] = manual_mode
        return self._encode_plain(manual)

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self._encode_mode(queries, "query")

    def encode_passages(self, passages: list[str]) -> np.ndarray:
        return self._encode_mode(passages, "passage")


class JinaEmbedder(SentenceTransformersEmbedder):
    def __init__(self, config: EmbedderConfig) -> None:
        super().__init__(replace(config, trust_remote_code=True))
        self.metadata["backend"] = "jina"

    def _encode_mode(self, texts: Sequence[str], mode: str) -> np.ndarray:
        clean = _as_texts(texts)
        if not clean:
            return np.empty((0, self.dim), dtype=np.float32)
        task = "retrieval.query" if mode == "query" else "retrieval.passage"
        if _has_encode_kw(self.model, "task"):
            try:
                out = self._encode_plain(clean, task=task)
                self.metadata[f"{mode}_mode"] = f"jina_task:{task}"
                self.metadata["jina_task_mode"] = True
                return out
            except (TypeError, ValueError):
                pass
        prompt_name = self.config.query_prompt_name if mode == "query" else self.config.passage_prompt_name
        if prompt_name and _has_encode_kw(self.model, "prompt_name"):
            try:
                out = self._encode_plain(clean, prompt_name=prompt_name)
                self.metadata[f"{mode}_mode"] = f"jina_prompt:{prompt_name}"
                self.metadata["jina_prompt_mode"] = True
                return out
            except (TypeError, ValueError, KeyError):
                pass
        self.metadata[f"{mode}_mode"] = "jina_plain"
        self.metadata["jina_plain_mode"] = True
        return self._encode_plain(clean)


class MeanPoolingTransformersEmbedder(BaseEmbedder):
    def __init__(self, config: EmbedderConfig) -> None:
        self.config = config
        self.metadata = {
            "backend": "transformers_mean_pooling",
            "model_name": config.model_name,
            "query_mode": "raw",
            "passage_mode": "raw",
        }
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install transformers and torch: pip install -U transformers torch") from exc
        self.torch = torch
        token = os.getenv("HF_TOKEN")
        kwargs = {"trust_remote_code": config.trust_remote_code}
        if token:
            kwargs["token"] = token
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name, **kwargs)
        if config.attn_implementation:
            kwargs["attn_implementation"] = config.attn_implementation
        try:
            self.model = AutoModel.from_pretrained(config.model_name, **kwargs).to(config.device)
        except TypeError:
            kwargs.pop("attn_implementation", None)
            self.model = AutoModel.from_pretrained(config.model_name, **kwargs).to(config.device)
        self.model.eval()

    @property
    def dim(self) -> int:
        hidden = getattr(getattr(self.model, "config", None), "hidden_size", None)
        if hidden:
            return int(hidden)
        return int(self.encode_passages(["dimension probe"]).shape[1])

    def _format(self, texts: Sequence[str], mode: str) -> tuple[list[str], str]:
        st = SentenceTransformersEmbedder.__new__(SentenceTransformersEmbedder)
        st.config = self.config
        return st._manual_texts(_as_texts(texts), mode)

    def _encode(self, texts: Sequence[str], mode: str) -> np.ndarray:
        clean, fmt_mode = self._format(texts, mode)
        self.metadata[f"{mode}_mode"] = fmt_mode
        if not clean:
            return np.empty((0, self.dim), dtype=np.float32)
        import torch.nn.functional as F

        outs = []
        max_length = self.config.max_seq_length or 512
        with self.torch.no_grad():
            for i in range(0, len(clean), self.config.batch_size):
                batch = clean[i:i + self.config.batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.config.device) for k, v in encoded.items()}
                outputs = self.model(**encoded)
                mask = encoded["attention_mask"].unsqueeze(-1).float()
                summed = (outputs.last_hidden_state * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                emb = summed / counts
                if self.config.normalize_embeddings:
                    emb = F.normalize(emb, p=2, dim=1)
                outs.append(emb.detach().cpu().numpy())
        return _float32_2d(np.vstack(outs))

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self._encode(queries, "query")

    def encode_passages(self, passages: list[str]) -> np.ndarray:
        return self._encode(passages, "passage")


class SentenceTransformersOrMeanPoolingEmbedder(BaseEmbedder):
    def __init__(self, config: EmbedderConfig) -> None:
        try:
            self.inner: BaseEmbedder = SentenceTransformersEmbedder(config)
        except Exception as exc:
            warnings.warn(
                f"SentenceTransformer load failed for {config.model_name!r}; falling back to transformers mean pooling: {exc}",
                RuntimeWarning,
            )
            self.inner = MeanPoolingTransformersEmbedder(config)
        self.config = config
        self.metadata = self.inner.metadata

    @property
    def dim(self) -> int:
        return self.inner.dim

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        return self.inner.encode_queries(queries)

    def encode_passages(self, passages: list[str]) -> np.ndarray:
        return self.inner.encode_passages(passages)


def build_embedder(config: EmbedderConfig) -> BaseEmbedder:
    entry = registry_entry(config.model_name)
    if entry.get("trust_remote_code") and not config.trust_remote_code:
        warnings.warn(
            f"{config.model_name} requires trust_remote_code=True; enabling it automatically.",
            RuntimeWarning,
        )
        config = replace(config, trust_remote_code=True)
    backend = (config.backend or "auto").strip().lower()
    if backend == "auto":
        backend = str(entry.get("backend") or "sentence_transformers")
    if backend == "sentence_transformers":
        return SentenceTransformersEmbedder(config)
    if backend == "jina":
        return JinaEmbedder(config)
    if backend == "sentence_transformers_or_mean_pooling":
        return SentenceTransformersOrMeanPoolingEmbedder(config)
    if backend == "transformers_mean_pooling":
        return MeanPoolingTransformersEmbedder(config)
    raise ValueError(f"Unsupported embedder backend={backend!r}")
