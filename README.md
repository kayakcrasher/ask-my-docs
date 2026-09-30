# Ask My Docs

A small RAG (retrieval-augmented generation) tool built from scratch in Python.

Point it at a folder of text files, ask questions in plain English, get answers
grounded in your documents with citations back to the source.

## What it does

1. **Ingest** — walks a folder, splits each file into overlapping chunks, embeds
   each chunk via Hugging Face, stores the vectors in SQLite.
2. **Ask** — embeds your question, finds the top-k most similar chunks by cosine
   similarity, sends them to Groq's LLM with a strict prompt, returns an answer
   with sources cited.

Every piece is written and explained. No LangChain. No external vector database.
No framework hiding the work.

## Install

```bash
git clone https://github.com/kayakcrasher/ask-my-docs.git
cd ask-my-docs

python -m venv --system-site-packages .venv
source .venv/bin/activate

# On Termux (Android):
pkg install python-numpy
pip install httpx python-dotenv rich

# On a laptop:
pip install numpy httpx python-dotenv rich
## Roadmap

- [ ] Evaluation — measure retrieval and answer quality on a labeled set
- [ ] Hybrid search — combine BM25 keyword search with vector search
- [ ] Reranking — a second model re-scores the top-N before generation
- [ ] Multi-format — PDF, HTML, DOCX loaders
- [ ] Web UI — one-page HTML + FastAPI, ask in the browser

## Status

Working end to end. Answers are grounded, cited, and correct on the sample
docs. Evaluated on a small hand-labeled set next.

## License

MIT
