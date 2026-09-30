"""Ask — the RAG loop: question → expand → retrieve → generate → answer.

Steps:
  1. Expand the question (LLM rewrites it, expanding abbreviations).
  2. Embed the expanded question (same model as ingestion).
  3. Search the store for top-k similar chunks.
  4. Build a prompt: system rules + chunks + question.
  5. Call Groq (with retries).
  6. Return answer + the chunks it saw.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from amd.embed import embed_query
from amd.store import Store

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_TOP_K = 5

SYSTEM_PROMPT = """You are Ask My Docs, a helpful assistant that answers questions
using ONLY the provided context chunks.

Rules:
- If the answer is in the context, answer using it. Cite sources as [1], [2], etc.
- If the answer is NOT in the context, say exactly: "I don't have information about that in the indexed documents."
- Do not invent facts. Do not use outside knowledge.
- Be concise. Two to four sentences unless the user asks for more.
"""

EXPAND_PROMPT = """You expand search queries to improve retrieval.

Given a user's question, rewrite it to include:
- Expansions of any abbreviations (e.g., "RAG" -> "RAG (retrieval-augmented generation)")
- Synonyms of key terms
- Related concepts that would help find relevant passages

Return ONLY the rewritten query. No explanation, no quotes, no prefix.
If the question is already clear, return it unchanged.
"""


class AskError(RuntimeError):
    pass


@dataclass
class Answer:
    text: str
    sources: list[dict]


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] (from {c['source']}, score={c['score']:.3f})\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def _call_groq(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 500,
    max_retries: int = 4,
) -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise AskError("GROQ_API_KEY is not set. Did you fill in .env?")

    delay = 3.0
    last_error = ""

    for attempt in range(max_retries):
        r = httpx.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=60.0,
        )

        if r.status_code == 200:
            data = r.json()
            try:
                content = data["choices"][0]["message"].get("content") or ""
            except (KeyError, IndexError) as e:
                raise AskError(f"unexpected Groq response: {data}") from e
            if not content.strip():
                raise AskError(
                    "Groq returned empty content (max_tokens may be too low "
                    "for a reasoning model)"
                )
            return content

        if r.status_code == 429:
            last_error = r.text[:200]
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
            continue

        raise AskError(f"Groq {r.status_code}: {r.text[:300]}")

    raise AskError(
        f"Groq rate limit not cleared after {max_retries} retries: {last_error}"
    )


def _expand_query(question: str) -> str:
    """Rewrite a question to improve retrieval. Falls back to original on error."""
    try:
        expanded = _call_groq(
            [
                {"role": "system", "content": EXPAND_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            max_tokens=500,
        ).strip()
    except AskError:
        return question
    if not expanded or len(expanded) > len(question) * 5:
        return question
    return expanded


def ask(
    question: str,
    db_path: str | Path = "amd.db",
    top_k: int = DEFAULT_TOP_K,
) -> Answer:
    """Run the full RAG loop and return the answer plus sources."""
    if not question or not question.strip():
        raise AskError("empty question")

    # 1. Expand the question, then embed.
    expanded = _expand_query(question)
    q_vec = embed_query(expanded)

    # 2. Search.
    store = Store(db_path)
    try:
        chunks = store.search(q_vec, top_k=top_k)
    finally:
        store.close()

    if not chunks:
        return Answer(
            text="I don't have information about that in the indexed documents.",
            sources=[],
        )

    # 3. Build the prompt.
    context = _build_context(chunks)
    user_msg = f"Context:\n\n{context}\n\nQuestion: {question}"

    # 4. Call the LLM.
    answer_text = _call_groq([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ])

    # 5. Return.
    return Answer(text=answer_text.strip(), sources=chunks)


def ask_main(argv: list[str] | None = None) -> int:
    """CLI: python -m amd.ask 'your question here'"""
    import sys
    from rich.console import Console

    console = Console()
    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        console.print('[yellow]usage:[/yellow] python -m amd.ask "your question"')
        return 1

    question = " ".join(args)
    try:
        result = ask(question)
    except Exception as e:
        console.print(f"[red]error:[/red] {e}")
        return 1

    console.print()
    console.print(f"[bold]Q:[/bold] {question}")
    console.print()
    console.print(f"[bold green]A:[/bold green] {result.text}")
    if result.sources:
        console.print()
        console.print("[dim]Sources:[/dim]")
        for i, s in enumerate(result.sources, 1):
            console.print(
                f"  [dim][{i}] {s['source']} (score {s['score']:.3f})[/dim]"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(ask_main())
