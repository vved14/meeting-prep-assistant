# Meeting Prep Assistant

A local RAG pipeline that writes pre-meeting briefs. You give it a `meeting_id`; it
treats that meeting as "upcoming", retrieves relevant context from all earlier
meetings, asks a local LLM to produce a structured brief, renders it to Markdown, and
scores the result.

Everything runs locally: Postgres for the meeting data, sentence-transformers for
embeddings, and Qwen via [Ollama](https://ollama.com) for generation. No API keys.

## How it works

```
              ┌─ db.py ──────────────────────────────────────┐
 meeting_id → │ fetch the target meeting + all PAST meetings  │
              └──────────────────────────────────────────────┘
                                  │
        ┌─ chunking.py + embeddings.py + vector_store.py ────────┐
        │ split transcripts into chunks, embed them (cached to   │
        │ disk), index them in FAISS                             │
        └────────────────────────────────────────────────────────┘
                                  │
              ┌─ vector_store.py ────────────────────────────┐
              │ embed the query, pull a candidate pool, and  │
              │ re-rank by cosine * recency decay            │
              └──────────────────────────────────────────────┘
                                  │
              ┌─ prompt.py + llm.py ─────────────────────────┐
              │ ask the LLM for a schema-constrained Brief    │
              │ (JSON), validated into a Pydantic model      │
              └──────────────────────────────────────────────┘
                                  │
              ┌─ prompt.py + validation.py ──────────────────┐
              │ render Markdown in code: drop hallucinated   │
              │ citations, classify attendees, no speculation │
              └──────────────────────────────────────────────┘
                                  │
              ┌─ eval.py + pipeline.py ──────────────────────┐
              │ score the brief (60% guideline / 40% quality) │
              │ and write the .md with the score stamped on  │
              └──────────────────────────────────────────────┘
```

**RAG = retrieval + generation.** Retrieval finds the right context (`db` → `vector_store`);
generation turns it into the brief (`prompt` → `llm`). The brief itself is built
deterministically in Python from a structured model response, so the LLM can't invent
citations, speculate about attendees, or mislabel who's internal/external.

## Files

| File | Role |
|------|------|
| `db.py` | Fetch the target meeting and all earlier meetings from the event log |
| `chunking.py` | Split transcripts into overlapping chunks |
| `embeddings.py` | Embed text with a local sentence-transformers model |
| `embedding_cache.py` | Transparent on-disk cache so repeat runs skip re-embedding |
| `vector_store.py` | Build a FAISS index and search it with recency-weighted re-ranking |
| `prompt.py` | The JSON contract, the prompt, and deterministic Markdown rendering |
| `llm.py` | Call the local model to generate a brief and to judge one |
| `validation.py` | Grounding checks: citation validity, speculation, invented names |
| `eval.py` | Score a brief (deterministic guideline checks + an LLM quality judge) |
| `pipeline.py` | Glue that runs the full pipeline and writes the brief |
| `config.py` | Settings read from the environment |
| `schema.sql` | The `core.events` table, so the seed will load |

## The data model

The backfill is **event-sourced**: one meeting is not one row, but several rows in
`core.events` sharing a `source_external_ref` (`meeting:tldv:<id>`), one per lifecycle
event:

- `meeting.record.requested` — title, attendees, scheduled time
- `meeting.transcript.ready` — transcript text
- `meeting.summary.ready` — summary

`db.py` reassembles a meeting from its event rows into a single `Meeting` object.

## Setup

```bash
# 1. Stand up the database and load the data
createdb lighthouse
psql -d lighthouse -f schema.sql
psql -d lighthouse -f /path/to/tldv-backfill.seed.sql

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Pull the local model (used for generation and the eval judge)
ollama pull qwen3:14b

# 4. Configure
cp .env.example .env     # set DATABASE_URL if it differs from the default
```

The first run also downloads the embedding model (`BAAI/bge-base-en-v1.5`, ~440 MB)
once and caches it.

## Usage

Generate a brief:

```bash
python pipeline.py 6a2f8dd00cab9500136ec383
```

The brief is written to `briefs/<meeting_id>.md`, with an eval score stamped at the
bottom (one judge LLM call per generation):

```
---
eval: 87.5 / 100 (guideline 95.0, quality 75.0)
```

Score a brief on its own and write a full scorecard to `eval/<meeting_id>.json`:

```bash
python eval.py 6a2f8dd00cab9500136ec383
```

## Evaluation

Each brief is scored out of 100:

- **Guideline (60%)** — six deterministic, $0 checks: section completeness, attendee
  coverage, internal/external classification, citation validity, no speculation, no
  invented names.
- **Quality (40%)** — an LLM judge (`qwen3:14b`) grades clarity and usefulness 0–100.

The weights are configurable in `config.py` / `.env`.

## Requirements

- Python 3.10+
- PostgreSQL (with the seed loaded)
- Ollama running locally with `qwen3:14b` pulled
