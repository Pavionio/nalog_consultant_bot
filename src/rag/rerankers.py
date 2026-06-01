from __future__ import annotations

import logging
import math
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)

_QWEN3_RERANKER_RE = re.compile(r"^Qwen/Qwen3-Reranker-[^/]+$")
_JINA_RERANKER_RE = re.compile(r"^jinaai/jina-reranker-[^/]+$")
_MIXEDBREAD_RERANKER_V2_RE = re.compile(r"^mixedbread-ai/mxbai-rerank-[^/]+-v2$")
_ALIBABA_GTE_RERANKER_BASE = "Alibaba-NLP/gte-multilingual-reranker-base"

_RERANKER_MODEL_CACHE: dict[tuple[Any, ...], "BaseRerankerBackend"] = {}
_RERANKER_MODEL_CACHE_LOCK = threading.Lock()


def _is_qwen3_reranker(model_name: str) -> bool:
    return bool(_QWEN3_RERANKER_RE.match(model_name))


def _is_jina_reranker(model_name: str) -> bool:
    return bool(_JINA_RERANKER_RE.match(model_name))


def _is_mixedbread_reranker_v2(model_name: str) -> bool:
    return bool(_MIXEDBREAD_RERANKER_V2_RE.match(model_name))


def _reranker_model_cache_enabled() -> bool:
    val = os.getenv("RAG_RERANKER_MODEL_CACHE", "1").strip().lower()
    return val not in ("0", "false", "no", "off")


def _hf_cache_dir() -> str:
    if os.getenv("RAG_RERANKER_CACHE_DIR"):
        cache_dir = os.environ["RAG_RERANKER_CACHE_DIR"]
    elif os.getenv("HF_HUB_CACHE"):
        cache_dir = os.environ["HF_HUB_CACHE"]
    elif os.getenv("HF_HOME"):
        cache_dir = str(Path(os.environ["HF_HOME"]) / "hub")
    else:
        cache_dir = str(Path.home() / ".cache" / "huggingface" / "hub")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    return cache_dir


def _hf_model_cache_exists(model_name: str) -> bool:
    model_dir = Path(_hf_cache_dir()) / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return False
    for snapshot in snapshots_dir.iterdir():
        if not snapshot.is_dir():
            continue
        if (snapshot / "config.json").exists():
            return True
    return False


def _hf_local_files_only(model_name: Optional[str] = None) -> bool:
    if os.getenv("RAG_RERANKER_LOCAL_FILES_ONLY", "").strip().lower() in ("1", "true", "yes"):
        return True
    if os.getenv("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes"):
        return True
    if os.getenv("TRANSFORMERS_OFFLINE", "").strip().lower() in ("1", "true", "yes"):
        return True
    return bool(model_name and _hf_model_cache_exists(model_name))


def _hf_common_kwargs(model_name: Optional[str] = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"cache_dir": _hf_cache_dir()}
    if _hf_local_files_only(model_name):
        kwargs["local_files_only"] = True
    token = os.getenv("HF_TOKEN")
    if token:
        kwargs["token"] = token
    return kwargs


def _cross_encoder_common_kwargs(model_name: Optional[str] = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"cache_folder": _hf_cache_dir()}
    if _hf_local_files_only(model_name):
        kwargs["local_files_only"] = True
    token = os.getenv("HF_TOKEN")
    if token:
        kwargs["token"] = token
    return kwargs


@dataclass
class RerankerResult:
    index: int
    score: float


class BaseRerankerBackend:
    reranker_type = "base"

    def __init__(self, config: Any) -> None:
        self.config = config
        self.model_name = config.reranker_model
        self.device = config.reranker_device
        self.batch_size = int(config.reranker_batch_size)
        self.max_length = int(_effective_max_length(config))
        self.normalize = bool(config.reranker_normalize)

    def score(self, query: str, passages: List[str]) -> List[float]:
        raise NotImplementedError


def auto_detect_reranker_type(model_name: str) -> str:
    if model_name == "BAAI/bge-reranker-v2-m3":
        return "flagembedding"
    if model_name == "BAAI/bge-reranker-v2-gemma":
        return "flagembedding_llm"
    if model_name == "BAAI/bge-reranker-v2.5-gemma2-lightweight":
        return "flagembedding_lightweight"
    if _is_qwen3_reranker(model_name):
        return "qwen3"
    if _is_jina_reranker(model_name):
        return "jina"
    if _is_mixedbread_reranker_v2(model_name):
        return "mixedbread"
    if model_name == _ALIBABA_GTE_RERANKER_BASE:
        return "transformers_sequence_classification"
    if model_name.startswith("cross-encoder/"):
        return "sentence_transformers"
    return "sentence_transformers_or_transformers"


def _build_reranker_uncached(config: Any, reranker_type: str) -> BaseRerankerBackend:
    if reranker_type == "flagembedding":
        return FlagEmbeddingReranker(config)
    if reranker_type == "flagembedding_llm":
        return FlagEmbeddingLLMReranker(config)
    if reranker_type == "flagembedding_lightweight":
        return LightWeightFlagEmbeddingLLMReranker(config)
    if reranker_type == "sentence_transformers":
        return SentenceTransformersReranker(config)
    if reranker_type == "transformers_sequence_classification":
        try:
            trust_remote_code = config.reranker_model == _ALIBABA_GTE_RERANKER_BASE
            return SentenceTransformersReranker(config, trust_remote_code=trust_remote_code)
        except Exception as exc:
            logger.warning("SentenceTransformers load failed, falling back to transformers: %s", exc)
            return TransformersSequenceClassificationReranker(config)
    if reranker_type == "jina":
        return JinaReranker(config)
    if reranker_type == "mixedbread":
        return MixedbreadReranker(config)
    if reranker_type == "qwen3":
        return Qwen3Reranker(config)
    if reranker_type == "sentence_transformers_or_transformers":
        try:
            return SentenceTransformersReranker(config)
        except Exception as exc:
            logger.warning("SentenceTransformers load failed, falling back to transformers: %s", exc)
            return TransformersSequenceClassificationReranker(config)
    raise ValueError(f"Unsupported reranker_type={reranker_type!r}")


def _reranker_cache_key(config: Any, reranker_type: str) -> tuple[Any, ...]:
    return (
        reranker_type,
        str(getattr(config, "reranker_model", "")),
        str(getattr(config, "reranker_device", "cpu")),
        int(getattr(config, "reranker_batch_size", 8) or 8),
        int(_effective_max_length(config)),
        bool(getattr(config, "reranker_use_fp16", False)),
        bool(getattr(config, "reranker_normalize", False)),
        str(getattr(config, "reranker_instruction", "") or ""),
        os.getenv("RAG_RERANKER_ATTN_IMPLEMENTATION", "eager").strip(),
        os.getenv("RAG_QWEN3_USE_CROSS_ENCODER", "").strip().lower(),
        os.getenv("RAG_JINA_ALLOW_LONG_CONTEXT", "").strip().lower(),
        os.getenv("RAG_ALIBABA_GTE_ALLOW_CUDA", "").strip().lower(),
        _hf_cache_dir(),
        os.getenv("RAG_RERANKER_LOCAL_FILES_ONLY", "").strip().lower(),
    )


def build_reranker(config: Any) -> BaseRerankerBackend:
    requested_type = (config.reranker_type or "auto").strip().lower()
    reranker_type = auto_detect_reranker_type(config.reranker_model) if requested_type == "auto" else requested_type

    if not _reranker_model_cache_enabled():
        return _build_reranker_uncached(config, reranker_type)

    cache_key = _reranker_cache_key(config, reranker_type)
    with _RERANKER_MODEL_CACHE_LOCK:
        cached = _RERANKER_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    reranker = _build_reranker_uncached(config, reranker_type)
    with _RERANKER_MODEL_CACHE_LOCK:
        return _RERANKER_MODEL_CACHE.setdefault(cache_key, reranker)


def _effective_max_length(config: Any) -> int:
    explicit = int(getattr(config, "reranker_max_length", 1024) or 1024)
    return explicit


def _is_cuda(device: str) -> bool:
    return str(device).startswith("cuda")


def _as_float_list(scores: Any) -> List[float]:
    if scores is None:
        return []
    if isinstance(scores, (float, int)):
        return [float(scores)]
    try:
        import torch
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().reshape(-1).tolist()
    except Exception:
        pass
    try:
        import numpy as np
        if isinstance(scores, np.ndarray):
            scores = scores.reshape(-1).tolist()
    except Exception:
        pass
    if isinstance(scores, list):
        out: List[float] = []
        for x in scores:
            if isinstance(x, (list, tuple)) and x:
                x = x[-1]
            out.append(float(x))
        return out
    return [float(x) for x in list(scores)]


def _sigmoid_scores(scores: Sequence[float]) -> List[float]:
    return [1.0 / (1.0 + math.exp(-max(min(float(s), 60.0), -60.0))) for s in scores]


def _validate_scores(scores: List[float], passages: List[str]) -> List[float]:
    if len(scores) != len(passages):
        raise RuntimeError(f"Reranker returned {len(scores)} scores for {len(passages)} passages")
    return [float(s) for s in scores]


def _ensure_padding_token(model: Any) -> None:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None) or getattr(tokenizer, "unk_token", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        return
    for obj_name in ("model", "automodel"):
        obj = getattr(model, obj_name, None)
        config = getattr(obj, "config", None)
        if config is not None:
            config.pad_token_id = pad_token_id


def _with_attention_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    attn_impl = os.getenv("RAG_RERANKER_ATTN_IMPLEMENTATION", "eager").strip()
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    return kwargs


def _from_pretrained_retry_without_attn(loader: Any, model_name: str, **kwargs: Any) -> Any:
    try:
        return loader.from_pretrained(model_name, **kwargs)
    except TypeError:
        if "attn_implementation" not in kwargs:
            raise
        kwargs = dict(kwargs)
        kwargs.pop("attn_implementation", None)
        return loader.from_pretrained(model_name, **kwargs)


class FlagEmbeddingReranker(BaseRerankerBackend):
    reranker_type = "flagembedding"

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise RuntimeError("Install FlagEmbedding: pip install FlagEmbedding") from exc
        devices = [self.device] if _is_cuda(self.device) else None
        self.model = FlagReranker(self.model_name, devices=devices, use_fp16=config.reranker_use_fp16)
        self._fallback: Optional[BaseRerankerBackend] = None

    def score(self, query: str, passages: List[str]) -> List[float]:
        if not passages:
            return []
        pairs = [[query, passage] for passage in passages]
        try:
            scores = self.model.compute_score(
                pairs,
                batch_size=self.batch_size,
                max_length=self.max_length,
                normalize=self.normalize,
            )
        except AttributeError as exc:
            if "prepare_for_model" not in str(exc):
                raise
            logger.warning("FlagEmbedding tokenizer API mismatch; falling back to SentenceTransformers CrossEncoder: %s", exc)
            if self._fallback is None:
                self._fallback = SentenceTransformersReranker(self.config, trust_remote_code=True)
                self._fallback.reranker_type = self.reranker_type
            return self._fallback.score(query, passages)
        return _validate_scores(_as_float_list(scores), passages)


class FlagEmbeddingLLMReranker(FlagEmbeddingReranker):
    reranker_type = "flagembedding_llm"

    def __init__(self, config: Any) -> None:
        BaseRerankerBackend.__init__(self, config)
        try:
            from FlagEmbedding import FlagLLMReranker
        except ImportError as exc:
            raise RuntimeError("Install FlagEmbedding: pip install FlagEmbedding") from exc
        devices = [self.device] if _is_cuda(self.device) else None
        self.model = FlagLLMReranker(self.model_name, devices=devices, use_fp16=config.reranker_use_fp16)


class LightWeightFlagEmbeddingLLMReranker(FlagEmbeddingLLMReranker):
    reranker_type = "flagembedding_lightweight"

    def __init__(self, config: Any) -> None:
        BaseRerankerBackend.__init__(self, config)
        try:
            from FlagEmbedding import LightWeightFlagLLMReranker
        except ImportError as exc:
            raise RuntimeError("Install FlagEmbedding: pip install FlagEmbedding") from exc
        devices = [self.device] if _is_cuda(self.device) else None
        try:
            self.model = LightWeightFlagLLMReranker(self.model_name, devices=devices, use_fp16=config.reranker_use_fp16)
        except Exception as exc:
            raise RuntimeError(
                "BAAI/bge-reranker-v2.5-gemma2-lightweight failed to load with this FlagEmbedding/transformers stack. "
                "Treat it as optional heavy model or try a newer FlagEmbedding release."
            ) from exc

    def score(self, query: str, passages: List[str]) -> List[float]:
        if not passages:
            return []
        pairs = [[query, passage] for passage in passages]
        kwargs = {
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "normalize": self.normalize,
            "cutoff_layers": [28],
            "compress_ratio": 2,
            "compress_layers": [24, 40],
        }
        try:
            scores = self.model.compute_score(pairs, **kwargs)
        except TypeError:
            scores = self.model.compute_score(
                pairs,
                batch_size=self.batch_size,
                max_length=self.max_length,
                normalize=self.normalize,
            )
        except Exception as exc:
            raise RuntimeError(
                "BAAI/bge-reranker-v2.5-gemma2-lightweight scoring failed with this FlagEmbedding/transformers stack."
            ) from exc
        return _validate_scores(_as_float_list(scores), passages)


class SentenceTransformersReranker(BaseRerankerBackend):
    reranker_type = "sentence_transformers"

    def __init__(self, config: Any, *, trust_remote_code: bool = False) -> None:
        super().__init__(config)
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("Install sentence-transformers: pip install sentence-transformers") from exc
        kwargs = {
            "device": self.device,
            "trust_remote_code": trust_remote_code,
            "max_length": self.max_length,
            **_cross_encoder_common_kwargs(self.model_name),
        }
        try:
            self.model = CrossEncoder(self.model_name, **kwargs)
        except TypeError:
            kwargs.pop("token", None)
            self.model = CrossEncoder(self.model_name, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"Could not load SentenceTransformers CrossEncoder {self.model_name!r}: {exc}") from exc
        _ensure_padding_token(self.model)

    def score(self, query: str, passages: List[str]) -> List[float]:
        if not passages:
            return []
        pairs = [(query, passage) for passage in passages]
        scores = _as_float_list(self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False))
        if self.normalize:
            scores = _sigmoid_scores(scores)
        return _validate_scores(scores, passages)


class Qwen3Reranker(SentenceTransformersReranker):
    reranker_type = "qwen3"

    DEFAULT_INSTRUCTION = (
        "Given a Russian tax/legal user query, retrieve official Russian tax document passages that directly answer the query."
    )

    def __init__(self, config: Any) -> None:
        BaseRerankerBackend.__init__(self, config)
        self.instruction = config.reranker_instruction or self.DEFAULT_INSTRUCTION
        self.model = None
        # In some local stacks CrossEncoder loads Qwen3 rerankers as
        # Qwen3ForSequenceClassification with a newly initialized score head
        # ("score.weight MISSING"). That is not a valid reranker. Use the
        # official yes/no logits path by default; keep CrossEncoder opt-in for
        # environments where sentence-transformers handles Qwen3 correctly.
        if os.getenv("RAG_QWEN3_USE_CROSS_ENCODER", "").strip().lower() not in ("1", "true", "yes"):
            self._raw_fallback = Qwen3TransformersFallback(self)
            return
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("Install sentence-transformers: pip install sentence-transformers") from exc
        try:
            kwargs = {
                "device": self.device,
                "trust_remote_code": True,
                "max_length": self.max_length,
                **_cross_encoder_common_kwargs(self.model_name),
            }
            if self.instruction:
                kwargs["prompts"] = {"classification": self.instruction}
                kwargs["default_prompt_name"] = "classification"
            try:
                self.model = CrossEncoder(self.model_name, **kwargs)
            except TypeError:
                kwargs.pop("token", None)
                try:
                    self.model = CrossEncoder(self.model_name, **kwargs)
                except TypeError:
                    kwargs.pop("prompts", None)
                    kwargs.pop("default_prompt_name", None)
                    self.model = CrossEncoder(self.model_name, **kwargs)
            _ensure_padding_token(self.model)
            self._raw_fallback = None
        except Exception as exc:
            logger.warning("Qwen3 CrossEncoder load failed; trying raw transformers fallback: %s", exc)
            self.model = None
            self._raw_fallback = Qwen3TransformersFallback(self)

    def score(self, query: str, passages: List[str]) -> List[float]:
        if not passages:
            return []
        if self._raw_fallback is not None:
            return self._raw_fallback.score(query, passages)
        try:
            return SentenceTransformersReranker.score(self, query, passages)
        except Exception as exc:
            logger.warning("Qwen3 CrossEncoder scoring failed; switching to raw transformers fallback: %s", exc)
            self.model = None
            self._raw_fallback = Qwen3TransformersFallback(self)
            return self._raw_fallback.score(query, passages)


class Qwen3TransformersFallback:
    def __init__(self, parent: Qwen3Reranker) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install transformers>=4.51.0 for Qwen3 rerankers") from exc

        self.parent = parent
        self.torch = torch
        dtype = torch.float16 if _is_cuda(parent.device) and parent.config.reranker_use_fp16 else torch.float32
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                parent.model_name,
                padding_side="left",
                trust_remote_code=True,
                **_hf_common_kwargs(parent.model_name),
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
            self.model = _from_pretrained_retry_without_attn(
                AutoModelForCausalLM,
                parent.model_name,
                **_with_attention_kwargs({
                    "torch_dtype": dtype,
                    "trust_remote_code": True,
                    **_hf_common_kwargs(parent.model_name),
                }),
            )
        except Exception as exc:
            raise RuntimeError("Qwen3 rerankers require transformers>=4.51.0 and recent sentence-transformers.") from exc
        self.model.to(parent.device)
        if self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        self.token_false_id = self._single_token_id(("no", "No", " no", " No"))
        self.token_true_id = self._single_token_id(("yes", "Yes", " yes", " Yes"))
        self.prefix = (
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
            'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_ids = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def _single_token_id(self, candidates: Sequence[str]) -> int:
        for text in candidates:
            token_id = self.tokenizer.convert_tokens_to_ids(text)
            if token_id is not None and token_id != self.tokenizer.unk_token_id:
                return int(token_id)
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if ids:
                return int(ids[-1])
        raise RuntimeError(f"Could not resolve yes/no token id for {self.parent.model_name}")

    def score(self, query: str, passages: List[str]) -> List[float]:
        out: List[float] = []
        for start in range(0, len(passages), self.parent.batch_size):
            batch = passages[start:start + self.parent.batch_size]
            inputs = self._tokenize_batch(query, batch)
            with self.torch.no_grad():
                logits = self.model(**inputs).logits[:, -1, :]
                yes_no = logits[:, [self.token_false_id, self.token_true_id]]
                probs = self.torch.nn.functional.log_softmax(yes_no, dim=1).exp()[:, 1]
            out.extend(_as_float_list(probs))
        return _validate_scores(out, passages)

    def _tokenize_batch(self, query: str, passages: List[str]) -> Any:
        # Preserve the chat prefix/suffix exactly. If tokenizer truncation is
        # applied to the whole prompt, long documents can remove the assistant
        # suffix and make the yes/no next-token score invalid.
        max_body_length = self.parent.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        if max_body_length < 1:
            raise RuntimeError(
                f"Qwen3 reranker max_length={self.parent.max_length} is too small for the yes/no prompt; "
                f"use at least {len(self.prefix_ids) + len(self.suffix_ids) + 1}."
            )
        features = []
        for doc in passages:
            body = f"<Instruct>: {self.parent.instruction}\n<Query>: {query}\n<Document>: {doc}"
            body_ids = self.tokenizer.encode(
                body,
                add_special_tokens=False,
                truncation=True,
                max_length=max_body_length,
            )
            input_ids = self.prefix_ids + body_ids + self.suffix_ids
            features.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
        return self.tokenizer.pad(features, padding=True, return_tensors="pt").to(self.parent.device)


class CausalYesNoReranker(BaseRerankerBackend):
    reranker_type = "causal_yes_no"

    def __init__(self, config: Any, *, instruction: Optional[str] = None) -> None:
        super().__init__(config)
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install transformers>=4.51.0 for causal rerankers") from exc
        self.torch = torch
        self.instruction = instruction or (
            "Judge whether the document directly answers the Russian tax/legal query. Answer only yes or no."
        )
        dtype = torch.float16 if _is_cuda(self.device) and self.config.reranker_use_fp16 else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            padding_side="left",
            trust_remote_code=True,
            **_hf_common_kwargs(self.model_name),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.model = _from_pretrained_retry_without_attn(
            AutoModelForCausalLM,
            self.model_name,
            **_with_attention_kwargs({
                "torch_dtype": dtype,
                "trust_remote_code": True,
                **_hf_common_kwargs(self.model_name),
            }),
        )
        self.model.to(self.device)
        if self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()
        self.no_id = self._single_token_id(("no", "No", " no", " No"))
        self.yes_id = self._single_token_id(("yes", "Yes", " yes", " Yes"))

    def _single_token_id(self, candidates: Sequence[str]) -> int:
        for text in candidates:
            token_id = self.tokenizer.convert_tokens_to_ids(text)
            if token_id is not None and token_id != self.tokenizer.unk_token_id:
                return int(token_id)
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if ids:
                return int(ids[-1])
        raise RuntimeError(f"Could not resolve yes/no token id for {self.model_name}")

    def _format(self, query: str, passage: str) -> str:
        return (
            "<|im_start|>system\nYou are a search relevance judge. Answer only yes or no.<|im_end|>\n"
            "<|im_start|>user\n"
            f"Instruction: {self.instruction}\n"
            f"Query: {query}\n"
            f"Document: {passage}\n"
            "Is the document relevant to the query?<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def score(self, query: str, passages: List[str]) -> List[float]:
        if not passages:
            return []
        out: List[float] = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start:start + self.batch_size]
            texts = [self._format(query, doc) for doc in batch]
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with self.torch.no_grad():
                logits = self.model(**inputs).logits[:, -1, :]
                yes_no = logits[:, [self.no_id, self.yes_id]]
                probs = self.torch.nn.functional.log_softmax(yes_no, dim=1).exp()[:, 1]
            out.extend(_as_float_list(probs))
        return _validate_scores(out, passages)


class JinaReranker(BaseRerankerBackend):
    reranker_type = "jina"

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        if (
            _is_cuda(self.device)
            and self.max_length > 512
            and os.getenv("RAG_JINA_ALLOW_LONG_CONTEXT", "").strip().lower() not in ("1", "true", "yes")
        ):
            logger.warning(
                "Jina reranker max_length is capped to 512 on CUDA to avoid native-attention driver failures. "
                "Set RAG_JINA_ALLOW_LONG_CONTEXT=1 to use the configured max_length=%s.",
                self.max_length,
            )
            self.max_length = 512
        logger.warning(
            "jinaai/jina-reranker-v2-base-multilingual is CC-BY-NC-4.0; treat as research/evaluation unless license is acceptable."
        )
        try:
            import torch
            from transformers import AutoConfig, AutoModelForSequenceClassification
        except ImportError as exc:
            raise RuntimeError("Jina reranker may require transformers and einops: pip install transformers einops") from exc
        model_config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True, **_hf_common_kwargs(self.model_name))
        if hasattr(model_config, "use_flash_attn"):
            model_config.use_flash_attn = False
        dtype = torch.float16 if _is_cuda(self.device) and self.config.reranker_use_fp16 else torch.float32
        self.model = _from_pretrained_retry_without_attn(
            AutoModelForSequenceClassification,
            self.model_name,
            **_with_attention_kwargs({
                "config": model_config,
                "torch_dtype": dtype,
                "trust_remote_code": True,
                **_hf_common_kwargs(self.model_name),
            }),
        )
        self.model.to(self.device)
        self.model.eval()

    def score(self, query: str, passages: List[str]) -> List[float]:
        if not passages:
            return []
        sentence_pairs = [[query, doc] for doc in passages]
        scores = self.model.compute_score(sentence_pairs, max_length=self.max_length)
        scores = _as_float_list(scores)
        if self.normalize:
            scores = _sigmoid_scores(scores)
        return _validate_scores(scores, passages)


class MixedbreadReranker(BaseRerankerBackend):
    reranker_type = "mixedbread"

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        if os.getenv("RAG_MIXEDBREAD_USE_CROSS_ENCODER", "").strip().lower() in ("1", "true", "yes"):
            raise RuntimeError(
                "Do not use sentence_transformers backend for mxbai-rerank-v2: score.weight may be randomly initialized. "
                "Use reranker_type=mixedbread with mxbai-rerank instead."
            )
        self.backend = MixedbreadRerankerBackend(
            self.model_name,
            max_length=self.max_length,
            batch_size=self.batch_size,
            device=self.device,
            instruction=self.config.reranker_instruction,
        )

    def score(self, query: str, passages: List[str]) -> List[float]:
        scores = self.backend.score(query, passages)
        if self.normalize:
            scores = _sigmoid_scores(scores)
        return _validate_scores(scores, passages)


class MixedbreadRerankerBackend:
    def __init__(
        self,
        model_name: str,
        max_length: int,
        batch_size: int,
        device: str,
        instruction: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.device = device
        self.instruction = instruction
        try:
            from mxbai_rerank import MxbaiRerankV2
        except ImportError as imp_exc:
            raise RuntimeError("Install mixedbread reranker backend: uv pip install -U mxbai-rerank") from imp_exc
        dtype = "float16" if _is_cuda(device) else "auto"
        try:
            self.model = MxbaiRerankV2(
                model_name,
                device=device,
                torch_dtype=dtype,
                max_length=self.max_length,
                tokenizer_kwargs=_hf_common_kwargs(model_name),
                disable_transformers_warnings=True,
                **_hf_common_kwargs(model_name),
            )
        except TypeError:
            self.model = MxbaiRerankV2(
                model_name,
                device=device,
                max_length=self.max_length,
                tokenizer_kwargs=_hf_common_kwargs(model_name),
                **_hf_common_kwargs(model_name),
            )

    def score(self, query: str, passages: List[str]) -> List[float]:
        if not passages:
            return []
        kwargs = {
            "query": query,
            "documents": passages,
            "top_k": len(passages),
            "batch_size": self.batch_size,
            "sort": False,
            "return_documents": False,
            "show_progress": False,
        }
        if self.instruction:
            kwargs["instruction"] = self.instruction
        try:
            results = self.model.rank(**kwargs)
        except TypeError:
            kwargs.pop("instruction", None)
            results = self.model.rank(**kwargs)

        if isinstance(results, (list, tuple)) and len(results) == len(passages):
            simple_scores = []
            for item in results:
                if isinstance(item, (float, int)):
                    simple_scores.append(float(item))
            if len(simple_scores) == len(passages):
                return simple_scores

        # mxbai-rerank 0.1.x returns RankResult(index, score, document).
        # With sort=False, results are currently in input order, but we still
        # restore by index so score() never depends on that ordering.
        scores = [0.0] * len(passages)
        unresolved_by_text: dict[str, List[int]] = {}
        for i, passage in enumerate(passages):
            unresolved_by_text.setdefault(passage, []).append(i)
        seen = 0
        for rank_item in results or []:
            idx = rank_item.get("index") if isinstance(rank_item, dict) else getattr(rank_item, "index", None)
            score = rank_item.get("score") if isinstance(rank_item, dict) else getattr(rank_item, "score", None)
            document = rank_item.get("document") if isinstance(rank_item, dict) else getattr(rank_item, "document", None)
            if idx is None and document is not None:
                candidates = unresolved_by_text.get(document) or []
                idx = candidates.pop(0) if candidates else None
            if idx is not None and score is not None:
                idx = int(idx)
                if 0 <= idx < len(scores):
                    scores[idx] = float(score)
                    seen += 1
        if seen != len(passages):
            raise RuntimeError(f"mxbai-rerank returned scores for {seen}/{len(passages)} passages")
        return _validate_scores(scores, passages)


class TransformersSequenceClassificationReranker(BaseRerankerBackend):
    reranker_type = "transformers_sequence_classification"

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install transformers>=4.51.0 for rerankers") from exc
        self.torch = torch
        if (
            self.model_name == _ALIBABA_GTE_RERANKER_BASE
            and _is_cuda(self.device)
            and os.getenv("RAG_ALIBABA_GTE_ALLOW_CUDA", "").strip().lower() not in ("1", "true", "yes")
        ):
            logger.warning("Alibaba GTE reranker is forced to CPU to avoid CUDA device-side assert in this stack.")
            self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, **_hf_common_kwargs(self.model_name))
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = getattr(self.tokenizer, "eos_token", None) or getattr(self.tokenizer, "unk_token", None)
        dtype = torch.float16 if _is_cuda(self.device) and config.reranker_use_fp16 else torch.float32
        self.model = _from_pretrained_retry_without_attn(
            AutoModelForSequenceClassification,
            self.model_name,
            **_with_attention_kwargs({
                "trust_remote_code": True,
                "torch_dtype": dtype,
                **_hf_common_kwargs(self.model_name),
            }),
        )
        self.model.to(self.device)
        if getattr(self.tokenizer, "pad_token_id", None) is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

    def score(self, query: str, passages: List[str]) -> List[float]:
        if not passages:
            return []
        out: List[float] = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start:start + self.batch_size]
            inputs = self.tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with self.torch.no_grad():
                logits = self.model(**inputs).logits
            if len(logits.shape) == 2 and logits.shape[1] == 1:
                batch_scores = logits[:, 0]
            elif len(logits.shape) == 2 and logits.shape[1] >= 2:
                batch_scores = logits[:, 1]
            else:
                batch_scores = logits.squeeze()
            out.extend(_as_float_list(batch_scores))
        if self.normalize:
            out = _sigmoid_scores(out)
        return _validate_scores(out, passages)
