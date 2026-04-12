import copy
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _chunk_base_id(chunk: dict[str, Any]) -> str:
    meta = chunk.get("metadata") or {}
    if meta.get("id") is not None:
        return str(meta["id"])
    if meta.get("chunk_id") is not None:
        return f"log:{meta['chunk_id']}"
    return f"hash:{hash(chunk.get('content', ''))}"


def _token_windows(
    token_ids: list[int], max_chunk: int, overlap: int
) -> list[list[int]]:
    if len(token_ids) <= max_chunk:
        return [token_ids]
    stride = max_chunk - overlap
    if stride <= 0:
        stride = max_chunk
    windows: list[list[int]] = []
    start = 0
    while start < len(token_ids):
        end = min(start + max_chunk, len(token_ids))
        windows.append(token_ids[start:end])
        if end >= len(token_ids):
            break
        start += stride
    return windows


def _char_windows(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    stride = max_chars - overlap_chars
    if stride <= 0:
        stride = max_chars
    parts: list[str] = []
    start = 0
    while start < len(text):
        parts.append(text[start : start + max_chars])
        if start + max_chars >= len(text):
            break
        start += stride
    return parts


def _cap_single_with_tokenizer(
    chunk: dict[str, Any],
    tokenizer: Any,
    max_chunk: int,
    overlap: int,
) -> list[dict[str, Any]]:
    content = chunk.get("content") or ""
    ids = tokenizer.encode(content, add_special_tokens=False)
    windows = _token_windows(ids, max_chunk, overlap)
    if len(windows) == 1 and len(ids) <= max_chunk:
        return [chunk]

    base_id = _chunk_base_id(chunk)
    meta_template = copy.deepcopy(chunk.get("metadata") or {})
    out: list[dict[str, Any]] = []
    total = len(windows)
    for i, win_ids in enumerate(windows):
        text = tokenizer.decode(win_ids, skip_special_tokens=True)
        new_meta = copy.deepcopy(meta_template)
        new_meta["token_cap_parent_id"] = base_id
        new_meta["token_subchunk_index"] = i
        new_meta["token_subchunk_total"] = total
        new_meta["id"] = f"{base_id}#t{i}"
        out.append(
            {
                "type": chunk.get("type"),
                "content": text,
                "metadata": new_meta,
            }
        )
    return out


def _cap_single_char_fallback(
    chunk: dict[str, Any],
    max_chunk: int,
    overlap: int,
    approx_chars_per_token: int,
) -> list[dict[str, Any]]:
    content = chunk.get("content") or ""
    max_chars = max_chunk * approx_chars_per_token
    overlap_chars = overlap * approx_chars_per_token
    parts = _char_windows(content, max_chars, overlap_chars)
    if len(parts) == 1:
        return [chunk]

    base_id = _chunk_base_id(chunk)
    meta_template = copy.deepcopy(chunk.get("metadata") or {})
    out: list[dict[str, Any]] = []
    total = len(parts)
    for i, text in enumerate(parts):
        new_meta = copy.deepcopy(meta_template)
        new_meta["token_cap_parent_id"] = base_id
        new_meta["token_subchunk_index"] = i
        new_meta["token_subchunk_total"] = total
        new_meta["id"] = f"{base_id}#c{i}"
        out.append(
            {
                "type": chunk.get("type"),
                "content": text,
                "metadata": new_meta,
            }
        )
    return out


def cap_chunks_by_tokens(
    chunks: list[dict[str, Any]],
    *,
    max_context_tokens: int | None = None,
    safety_tokens: int | None = None,
    overlap_tokens: int | None = None,
    tokenizer_model_id: str | None = None,
    approx_chars_per_token: int = 4,
) -> list[dict[str, Any]]:
    """
    Stage-2 cap: split any chunk whose content exceeds the embedding context
    into overlapping windows. Chunk shape stays {type, content, metadata}.

    Config via kwargs or env:
    - EMBED_MAX_CONTEXT_TOKENS (default 512)
    - EMBED_SAFETY_TOKENS (default 32) reserved below the hard context limit
    - EMBED_OVERLAP_TOKENS (default 64)
    - EMBED_TOKENIZER_MODEL: HuggingFace id for AutoTokenizer; empty unset uses char fallback
    """
    mct = max_context_tokens if max_context_tokens is not None else int(
        os.environ.get("EMBED_MAX_CONTEXT_TOKENS", "512")
    )
    st = safety_tokens if safety_tokens is not None else int(
        os.environ.get("EMBED_SAFETY_TOKENS", "32")
    )
    ot = overlap_tokens if overlap_tokens is not None else int(
        os.environ.get("EMBED_OVERLAP_TOKENS", "64")
    )
    tmid = (
        tokenizer_model_id
        if tokenizer_model_id is not None
        else os.environ.get("EMBED_TOKENIZER_MODEL", "mixedbread-ai/mxbai-embed-large-v1")
    )
    tmid = tmid.strip() if isinstance(tmid, str) else tmid
    if tmid == "":
        tmid = None

    max_chunk = max(1, mct - st)
    overlap = min(max(0, ot), max_chunk - 1) if max_chunk > 1 else 0

    tokenizer = None
    if tmid:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tmid, trust_remote_code=True)
        except Exception as e:
            logger.warning(
                "Could not load tokenizer %r (%s); using character fallback",
                tmid,
                e,
            )

    result: list[dict[str, Any]] = []
    for chunk in chunks:
        if tokenizer is not None:
            result.extend(
                _cap_single_with_tokenizer(chunk, tokenizer, max_chunk, overlap)
            )
        else:
            result.extend(
                _cap_single_char_fallback(
                    chunk, max_chunk, overlap, approx_chars_per_token
                )
            )
    return result
