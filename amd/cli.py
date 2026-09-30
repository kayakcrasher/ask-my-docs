"""Command-line interface — the single entry point.

Usage:
    python -m amd ingest [docs_dir]
    python -m amd ask "your question"
    python -m amd stats
    python -m amd list
    python -m amd reset
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

# Load .env from the project root (one level above amd/).
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from amd.ingest import ingest_path
from amd.store import Store

console = Console()
DEFAULT_DB = ROOT / "amd.db"
DEFAULT_DOCS = ROOT / "docs"

USAGE = """[bold]ask-my-docs[/bold] — RAG from scratch

[bold cyan]Commands:[/bold cyan]
  [green]ingest[/green] [dim]\\[docs_dir][/dim]     Index documents (default: docs/)
  [green]ask[/green] [dim]"question"[/dim]        Ask a question, get an answer
  [green]stats[/green]                    Show index size and sources
  [green]list[/green]                     List indexed sources
  [green]reset[/green]                    Delete the entire index

[dim]Examples:[/dim]
  python -m amd ingest
  python -m amd ingest my_notes/
  python -m amd ask "how does chunking work?"
"""


def cmd_ingest(args: list[str]) -> int:
    docs_dir = args[0] if args else str(DEFAULT_DOCS)
    return ingest_path(docs_dir, DEFAULT_DB) and 0 or 0


def cmd_ask(args: list[str]) -> int:
    if not args:
        console.print('[yellow]usage:[/yellow] python -m amd ask "your question"')
        return 1
    from amd.ask import ask

    question = " ".join(args)
    try:
        result = ask(question, db_path=DEFAULT_DB)
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
            console.print(f"  [dim][{i}] {s['source']} (score {s['score']:.3f})[/dim]")
    console.print()
    return 0


def cmd_stats(args: list[str]) -> int:
    if not DEFAULT_DB.exists():
        console.print("[yellow]no index yet. Run: python -m amd ingest[/yellow]")
        return 1
    store = Store(DEFAULT_DB)
    try:
        stats = store.stats()
        console.print()
        console.print(f"[bold]Index:[/bold] {DEFAULT_DB}")
        console.print(f"  files:  {stats['sources']}")
        console.print(f"  chunks: {stats['chunks']}")
        console.print()
    finally:
        store.close()
    return 0


def cmd_list(args: list[str]) -> int:
    if not DEFAULT_DB.exists():
        console.print("[yellow]no index yet. Run: python -m amd ingest[/yellow]")
        return 1
    store = Store(DEFAULT_DB)
    try:
        sources = store.sources()
        if not sources:
            console.print("[dim]index is empty[/dim]")
            return 0
        console.print()
        for name, count in sources:
            console.print(f"  [cyan]{name}[/cyan]  [dim]({count} chunk{'s' if count != 1 else ''})[/dim]")
        console.print()
    finally:
        store.close()
    return 0


def cmd_reset(args: list[str]) -> int:
    if not DEFAULT_DB.exists():
        console.print("[dim]nothing to reset[/dim]")
        return 0
    confirm = input("Delete the entire index? [y/N] ").strip().lower()
    if confirm != "y":
        console.print("[dim]cancelled[/dim]")
        return 0
    DEFAULT_DB.unlink()
    console.print("[green]index deleted[/green]")
    return 0


COMMANDS = {
    "ingest": cmd_ingest,
    "ask": cmd_ask,
    "stats": cmd_stats,
    "list": cmd_list,
    "reset": cmd_reset,
    "help": lambda _: (console.print(USAGE), 0)[1],
}


def main(argv: list[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        console.print(USAGE)
        return 0
    cmd, rest = args[0], args[1:]
    fn = COMMANDS.get(cmd)
    if not fn:
        console.print(f"[red]unknown command:[/red] {cmd}")
        console.print(USAGE)
        return 1
    return fn(rest)


if __name__ == "__main__":
    raise SystemExit(main())
