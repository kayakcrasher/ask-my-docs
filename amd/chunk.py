"""Text chunking — split documents into retrievable pieces.

Strategy:
  1. Split into paragraphs (blank-line boundaries).
  2. Merge small paragraphs until they reach the target size.
  3. Split oversized paragraphs at sentence boundaries.
  4. Add overlap between consecutive chunks so information
     at boundaries isn't lost.

Sizes are in characters, roughly 4 chars per token. We don't
tokenize because the LLM tokenizers we'd use are heavy; a good
approximation is fine for retrieval.
"""
from __future__ import annotations

import re
from typing import Iterable

DEFAULT_TARGET = 800     # ~200 tokens — comfortable chunk size
DEFAULT_MAX = 1600       # ~400 tokens — hard ceiling before we split
DEFAULT_OVERLAP = 100    # ~25 tokens — context carried between chunks

# Split on blank lines, optionally followed by a heading line.
_PARA_RE = re.compile(r"\n\s*\n")
# Sentence boundary: . ! ? followed by whitespace, not inside "3.5" or "Dr."
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARA_RE.split(text) if p.strip()]


def _split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(paragraph) if s.strip()]


def _merge_small(paragraphs: list[str], target: int) -> list[str]:
    """Glue consecutive small paragraphs together until they hit `target`."""
    merged: list[str] = []
    buf = ""
    for p in paragraphs:
        if not buf:
            buf = p
        elif len(buf) + len(p) + 2 <= target:
            buf = buf + "\n\n" + p
        else:
            merged.append(buf)
            buf = p
    if buf:
        merged.append(buf)
    return merged


def _split_large(paragraph: str, max_size: int) -> list[str]:
    """If a paragraph exceeds max_size, break it at sentence boundaries."""
    if len(paragraph) <= max_size:
        return [paragraph]
    sentences = _split_sentences(paragraph)
    pieces: list[str] = []
    buf = ""
    for s in sentences:
        if not buf:
            buf = s
        elif len(buf) + len(s) + 1 <= max_size:
            buf = buf + " " + s
        else:
            pieces.append(buf)
            buf = s
    if buf:
        pieces.append(buf)
    return pieces


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Prepend the tail of the previous chunk to the current one."""
    if overlap <= 0 or len(chunks) < 2:
        return chunks
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        tail = prev[-overlap:] if len(prev) > overlap else prev
        # avoid cutting mid-word: back up to the first whitespace
        if tail and not tail[0].isspace():
            ws = tail.find(" ")
            if ws != -1:
                tail = tail[ws + 1:]
        out.append((tail + "\n\n" + chunks[i]).strip() if tail else chunks[i])
    return out


def chunk_text(
    text: str,
    target: int = DEFAULT_TARGET,
    max_size: int = DEFAULT_MAX,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Split `text` into a list of overlapping chunks."""
    if not text or not text.strip():
        return []

    paragraphs = _split_paragraphs(text)

    # Step 1: split any paragraph that's over the hard cap.
    expanded: list[str] = []
    for p in paragraphs:
        expanded.extend(_split_large(p, max_size))

    # Step 2: merge tiny paragraphs up to the target size.
    merged = _merge_small(expanded, target)

    # Step 3: any merged chunk that's still over max gets split again.
    final: list[str] = []
    for c in merged:
        final.extend(_split_large(c, max_size))

    # Step 4: overlap.
    return _add_overlap(final, overlap)


def chunk_file(
    path,
    target: int = DEFAULT_TARGET,
    max_size: int = DEFAULT_MAX,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Read a file and chunk it. Returns [] for unreadable/binary files."""
    from pathlib import Path

    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    return chunk_text(text, target=target, max_size=max_size, overlap=overlap)


def iter_text_files(root, extensions: Iterable[str] = (".md", ".txt", ".rst")):
    """Yield every file under `root` whose suffix is in `extensions`."""
    from pathlib import Path

    root = Path(root)
    exts = tuple(e.lower() for e in extensions)
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p
