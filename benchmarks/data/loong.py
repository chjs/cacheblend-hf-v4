"""Loong dataset loader for the cache-then-reverse experiment protocol.

Source: `framolfese/Loong` (HF, English subset of the Tencent Loong benchmark).
Default split: `paper` (academic-paper QA, multi-doc).

The protocol asks for ~11 chunks per example. We filter for examples whose
`doc` list has ≥ 11 entries (43 in paper split, all 11-doc as of 2026-05).

Each example exposes:
    id:        unique str
    question:  the QA question
    answer:    gold answer (str; some answers contain commas/aliases via heuristic)
    docs_text: list[str] of the 11 (truncated to first 11) document texts
    instruction: per-example QA instruction (we usually replace with protocol prefix)
    length:    original Loong token length (for bucket analysis)

For our Loong-on-Llama setup, we keep docs full-length (Llama-3.1-8B handles
128k native). For Mistral 32k usage, the caller is responsible for length
filtering and/or truncation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class LoongExample:
    id: str
    question: str
    answer: str
    docs_text: list[str] = field(default_factory=list)
    instruction: str = ""
    length: int = 0
    type: str = ""


def load_loong(
    split: str = "paper",
    min_docs: int = 11,
    max_docs: int = 11,
    max_length_tokens: int | None = None,
    n: int | None = None,
) -> list[LoongExample]:
    """Load Loong examples filtered by doc count and (optional) length budget.

    Args:
        split: 'paper' (default) or 'financial'.
        min_docs / max_docs: filter examples to have between these many docs.
            Set both to 11 for the protocol's default.
        max_length_tokens: if set, skip examples whose Loong-reported `length` > this.
        n: if set, return only first n examples after filtering.

    Returns:
        list of LoongExample, in source order (deterministic for reproducibility).
    """
    from datasets import load_dataset
    ds = load_dataset("framolfese/Loong", split=split)
    out: list[LoongExample] = []
    for ex in ds:
        if len(ex["doc"]) < min_docs or len(ex["doc"]) > max_docs:
            continue
        if max_length_tokens is not None and ex["length"] > max_length_tokens:
            continue
        out.append(LoongExample(
            id=ex["id"],
            question=ex["question"],
            answer=ex["answer"],
            docs_text=_split_docs_text(ex["docs"], ex["doc"]),
            instruction=ex["instruction"],
            length=ex["length"],
            type=ex["type"],
        ))
        if n is not None and len(out) >= n:
            break
    return out


def _split_docs_text(concat_text: str, doc_names: list[str]) -> list[str]:
    """Split the concatenated `docs` field into per-document text using the doc
    name list as markers.

    The Loong dataset's `docs` field is the full concatenated text of all docs.
    Doc names typically appear as section headers ("Table of Contents" / paper
    title etc). We use a simple heuristic: split at each doc name occurrence;
    fall back to equal-length split if heuristic fails.
    """
    # Try splitting at doc name boundaries. Loong typically separates docs by
    # consecutive blank lines (≥3 newlines) or by a markdown-style header.
    # Heuristic: split by '\n\n\n' (most common Loong pattern); if doesn't give
    # the right count, fall back to equal-length char split.
    candidates = concat_text.split("\n\n\n")
    if len(candidates) == len(doc_names):
        return [c.strip() for c in candidates]
    # Fallback A: try splitting at first occurrence of each doc name.
    parts: list[str] = []
    text = concat_text
    for i, name in enumerate(doc_names):
        if i == 0:
            continue
        idx = text.find(name, len(parts[-1]) if parts else 0)
        if idx < 0:
            # name not found verbatim — fall through to fallback B
            parts = []
            break
        parts.append(text[:idx])
        text = text[idx:]
    if parts:
        parts.append(text)
        if len(parts) == len(doc_names):
            return [p.strip() for p in parts]
    # Fallback B: equal-length char split.
    n = len(doc_names)
    L = len(concat_text)
    span = L // n
    return [concat_text[i * span: (i + 1) * span if i < n - 1 else L].strip() for i in range(n)]
