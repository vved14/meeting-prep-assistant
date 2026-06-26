"""Glue that runs the full pipeline and writes the brief to a file."""
from __future__ import annotations

import argparse
from pathlib import Path

import config, db, chunking, vector_store, llm
from chunking import split_text
from embeddings import get_embedder
from embedding_cache import CachingEmbedder
from vector_store import Chunk, VectorStore
from prompt import Brief, render_markdown
from llm import generate_brief


def build_brief(meeting_id: str):
    """
    Run the full retrieval + generation path for one meeting.

    Returns (meeting, retrieved, brief_struct, markdown):
      - meeting       : the upcoming Meeting
      - retrieved     : list of (Chunk, score) used as context
      - brief_struct  : the validated `Brief` from the model
      - markdown      : the final rendered brief text
    Raises SystemExit with a clear message if the meeting/past data is missing.
    """
    conn = db.get_connection()
    try:
        meeting = db.fetch_meeting(conn, meeting_id)
        if meeting is None:
            raise SystemExit(f"No meeting found for id {meeting_id!r}")

        scheduled = meeting.created_at
        if scheduled is None:
            raise SystemExit(
                f"Meeting {meeting_id!r} has no scheduled time; can't find past meetings."
            )

        past_meetings = db.fetch_past_meetings(conn, scheduled)
        if not past_meetings:
            raise SystemExit(
                "No past meetings with transcripts before this one — nothing to summarize."
            )

        chunks: list[Chunk] = []
        for m in past_meetings:
            for piece in split_text(m.transcript_text or "", config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                chunks.append(
                    Chunk(
                        text=piece,
                        meeting_id=m.id,
                        meeting_title=m.title,
                        # Plain YYYY-MM-DD keeps citations and recency parsing clean.
                        meeting_date=m.created_at.date().isoformat() if m.created_at else "",
                    )
                )

        if not chunks:
            raise SystemExit("Past meetings had no usable transcript text to index.")

        embedder = CachingEmbedder(get_embedder())
        store = VectorStore(embedder)
        store.build(chunks)

        query = f"{meeting.title} {meeting.attendee_names}"
        # as_of applies recency weighting relative to the upcoming meeting's time.
        retrieved = store.search(query, config.TOP_K, as_of=scheduled)

        # The model returns a structured Brief; render it to Markdown in code
        # (deterministic stakeholders, classification, citation dropping).
        brief_struct: Brief = generate_brief(meeting, retrieved)
        markdown = render_markdown(brief_struct, meeting, retrieved)

        return meeting, retrieved, brief_struct, markdown
    finally:
        conn.close()


def run(meeting_id: str, out_path: str | None = None) -> str:
    """
    Build the brief for `meeting_id`, score it, and write the Markdown to a file.

    Every brief is scored and the overall eval is stamped at the bottom as an
    `eval: <number>` line. The judge runs an extra LLM call per generation.
    """
    meeting, retrieved, brief, markdown = build_brief(meeting_id)

    # Lazy import avoids a circular import (eval imports build_brief).
    from eval import score_brief
    result = score_brief(meeting, retrieved, brief, markdown)
    # Stamp the eval score onto the generated file.
    markdown = (
        markdown.rstrip()
        + f"\n\n---\neval: {result['overall_score']} / 100 "
        + f"(guideline {result['guideline_score']}, quality {result['quality_score']})\n"
    )

    if out_path is None:
        out_path = str(Path(config.OUTPUT_DIR) / f"{meeting_id}.md")
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return out_path


def main():
    """Parse a meeting_id argument (and an optional --out), then call run()."""
    parser = argparse.ArgumentParser(
        description="Generate a meeting-prep brief for an upcoming meeting."
    )
    parser.add_argument("meeting_id", help="tl;dv id of the upcoming meeting")
    parser.add_argument(
        "--out",
        dest="out_path",
        default=None,
        help="Where to write the brief (default: <OUTPUT_DIR>/<meeting_id>.md)",
    )
    args = parser.parse_args()

    out_path = run(args.meeting_id, args.out_path)
    print(f"Brief written to {out_path}")


if __name__ == "__main__":
    main()
