# Welcome to Ask My Docs

Ask My Docs is a small RAG tool built in Python. RAG stands for
**Retrieval-Augmented Generation** — a technique that combines a search step
(retrieval) with a language model's generation step, so answers are grounded in
your own documents instead of the model's training data.

Ask My Docs reads text files from a folder, splits them into chunks, embeds
each chunk, stores the embeddings in SQLite, and lets you ask questions.

## What it does

Given a folder of documents, Ask My Docs builds a searchable index. When you
ask a question, it finds the most relevant chunks and passes them to an LLM,
which writes an answer grounded in the source material.

## Why build it

To learn how retrieval-augmented generation works from the ground up. Not with
a framework, but by writing every piece: chunking, embedding, cosine
similarity, prompting, and evaluation.

## The stack

- Python 3.11+
- httpx for HTTP calls
- numpy for cosine similarity
- SQLite for vector storage
- Hugging Face for embeddings
- Groq for language generation

## How to use it

Run the ingest command first to build the index. Then ask questions from the
command line. Answers include citations back to the source document.

## Why SQLite instead of a vector database

For a personal project with thousands of chunks, SQLite plus numpy is faster
to set up and just as fast to query. Vector databases pay off at a hundred
thousand chunks or more. Below that, the overhead isn't worth it.
