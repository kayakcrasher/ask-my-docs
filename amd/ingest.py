"""Ingest — walk a folder, chunk, embed, store.

The pipeline:
    docs/*.md
      → chunk_text()      (split into pieces)
      → embed_texts()     (each piece → 384-dim vector)
      → Store.add()       (save to SQLite)

Idempotent: re-ingesting the same file replaces its chunks.
Safe to re-run after editing a doc.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (one level above the amd/ package).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from amd.chunk import chunk_file, iter_text_files
from amd.embed import embed_texts
from amd.store import Store

console = Console()


def ingest_path(
    root: str | Path,
    db_path: str | Path = "amd.db",
    extensions: tuple[str, ...] = (".md", ".txt", ".rst"),
) -> dict:
    """Index every text file under `root`. Returns a summary dict."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"docs folder not found: {root}")

    files = list(iter_text_files(root, extensions))
    if not files:
        console.print(f"[yellow]no files found in {root}[/yellow]")
        return {"files": 0, "chunks": 0, "skipped": 0}

    store = Store(db_path)
    total_chunks = 0
    skipped = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("ingesting…", total=len(files))
        for path in files:
            rel = str(path.relative_to(root))
            progress.update(task, description=f"ingesting {rel}")

            chunks = chunk_file(path)
            if not chunks:
                skipped += 1
                progress.advance(task)
                continue

            # Fresh start for this file: drop old chunks, then add new ones.
            store.delete_source(rel)

            vectors = embed_texts(chunks)
            store.add(rel, chunks, vectors)

            total_chunks += len(chunks)
            progress.advance(task)

    store.close()
    summary = {"files": len(files), "chunks": total_chunks, "skipped": skipped}
    console.print(
        f"[green]indexed[/green] {summary['files']} file(s) · "
        f"{summary['chunks']} chunk(s) · skipped {summary['skipped']}"
    )
    return summary


def ingest_main(argv: list[str] | None = None) -> int:
    """Simple CLI entry: python -m amd.ingest [docs_dir]"""
    import sys

    args = list(argv if argv is not None else sys.argv[1:])
    root = args[0] if args else "docs"
    try:
        ingest_path(root)
        return 0
    except Exception as e:
        console.print(f"[red]error:[/red] {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(ingest_main())
