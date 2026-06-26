"""The model's JSON contract, its prompt, and rendering the brief to Markdown."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

import config
from validation import is_valid_citation, valid_citation_set


# --- The JSON contract the model must fill -------------------------------------

class Bullet(BaseModel):
    """A single grounded claim plus the meeting it came from (citation required)."""
    claim: str
    source_title: str
    source_date: str


class Stakeholder(BaseModel):
    """
    What the model knows about one attendee. Only `cares_about` (+ its citation)
    is the model's job — the name, and whether they're internal/external, are
    decided in code by render_markdown, not trusted from the model.
    """
    name: str
    cares_about: Optional[str] = None
    source_title: Optional[str] = None
    source_date: Optional[str] = None


class Brief(BaseModel):
    """The whole brief as structured data. `llm.generate_brief` returns one of these."""
    current_state: list[Bullet]
    open_loops: list[Bullet]
    expected_topics: list[str]
    stakeholders: list[Stakeholder]


# --- The standing instructions -------------------------------------------------

SYSTEM_PROMPT = """You are a meeting-prep assistant. Using ONLY the context in the
user message (the upcoming meeting details and the retrieved transcript excerpts),
produce a pre-meeting brief as JSON matching the provided schema.

Rules:
- Use only what is in the context. Never invent facts, names, dates, decisions, or
  job titles. If a section has nothing supported by the context, return an empty
  list for it.
- The excerpts are transcript fragments, speaker-labeled "Name: ...", and may be
  out of order. Base every claim on them.
- For every bullet in `current_state` and `open_loops`, set `source_title` and
  `source_date` to the EXACT title and date of the excerpt it came from (copy them
  verbatim from the excerpt header). Do not merge multiple meetings into one cite.
- `expected_topics` is a list of short predicted discussion points (no citation).
- `stakeholders`: one entry per attendee listed in the upcoming meeting. Set
  `cares_about` to what that person cares about, based ONLY on their own lines in
  the excerpts, with the matching `source_title`/`source_date`. If an attendee
  does not appear in the excerpts, leave `cares_about`, `source_title`, and
  `source_date` null. Do NOT speculate about people who are not in the excerpts.
- Do not add job titles or affiliations; those are handled separately."""


# --- Building the per-request user message ------------------------------------

def build_user_message(meeting, retrieved: list) -> str:
    """
    Assemble the user turn: the upcoming meeting's details and an explicit attendee
    list, then each retrieved excerpt labelled with its source meeting and date so
    the model can copy those into citations.

    `retrieved` is a list of (Chunk, score) tuples from VectorStore.search().
    """
    lines = [
        "# Upcoming meeting",
        f"Title: {meeting.title}",
        f"Scheduled: {meeting.created_at}",
        "Attendees (produce exactly one stakeholder entry per name below):",
    ]

    # List attendees explicitly by name so the model FILLS this fixed list rather
    # than inventing who the stakeholders are.
    if meeting.attendees:
        for a in meeting.attendees:
            name = a.get("name")
            email = a.get("email")
            if name and email:
                lines.append(f"- {name} <{email}>")
            elif email:
                lines.append(f"- <{email}>")
            elif name:
                lines.append(f"- {name}")
    else:
        lines.append("- (none listed)")

    lines += ["", "# Retrieved context from past meetings"]

    if not retrieved:
        lines.append("(No past-meeting context was retrieved.)")
    else:
        for i, (chunk, _score) in enumerate(retrieved, start=1):
            lines.append("")
            lines.append(
                f'[Excerpt {i}] from "{chunk.meeting_title}" ({chunk.meeting_date}):'
            )
            lines.append(chunk.text)

    return "\n".join(lines)


# --- Deterministic rendering (where the bug fixes live) -----------------------

def classify_attendee(email: Optional[str]) -> tuple[str, str]:
    """
    Classify an attendee internal/external in code by email domain rather than
    trusting the LLM. Returns (label, domain). Public so eval.py can reuse the exact
    same classification rule.
    """
    if not email or "@" not in email:
        return "affiliation unknown", "unknown"
    domain = email.split("@")[-1].strip().lower()
    if domain == config.INTERNAL_EMAIL_DOMAIN.lower():
        return "internal", domain
    return "external", domain


def _render_bullets(bullets: list[Bullet], valid: set) -> list[str]:
    """
    Render grounded bullets, DROPPING any whose citation isn't a real retrieved
    meeting. This is what removes hallucinated cites like "(multiple meetings)".
    """
    out = []
    for b in bullets:
        if is_valid_citation(b.source_title, b.source_date, valid):
            out.append(f"- {b.claim.strip()} ({b.source_title}, {b.source_date})")
    return out


def _match_stakeholder(name: str, entries: list[Stakeholder]) -> Optional[Stakeholder]:
    """Find the model's entry for a real attendee by (loose) name match."""
    nl = name.strip().lower()
    for e in entries:
        el = (e.name or "").strip().lower()
        if el and (el == nl or el in nl or nl in el):
            return e
    return None


def render_markdown(brief: Brief, meeting, retrieved: list) -> str:
    """
    Turn the validated `Brief` into the final Markdown.

    Everything deterministic happens here, NOT in the model:
      - Current State / Open Loops: keep only bullets with a valid citation.
      - Stakeholder Context: iterate the REAL attendee list (one bullet each),
        classify internal/external in code, and fall back to "prior positions not
        covered." when the model has nothing grounded.
    """
    valid = valid_citation_set(retrieved)
    parts: list[str] = []

    # ## Current State
    current = _render_bullets(brief.current_state, valid)
    parts.append("## Current State")
    parts.extend(current or ["- Not covered in the available context."])

    # ## Open Loops
    loops = _render_bullets(brief.open_loops, valid)
    parts.append("\n## Open Loops")
    parts.extend(loops or ["- Not covered in the available context."])

    # ## Expected Topics (predictions — no citation required)
    parts.append("\n## Expected Topics")
    topics = [f"- {t.strip()}" for t in brief.expected_topics if t and t.strip()]
    parts.extend(topics or ["- Not covered in the available context."])

    # ## Stakeholder Context — driven by the real attendee list, not the model.
    parts.append("\n## Stakeholder Context")
    if not meeting.attendees:
        parts.append("- (no attendees listed)")
    else:
        for a in meeting.attendees:
            name = a.get("name")
            email = a.get("email")
            # strip() guards against messy source data (e.g. a leading space) so the
            # line — and the eval's coverage match — stay clean.
            display = (name or email or "(unknown)").strip()
            label, domain = classify_attendee(email)

            entry = _match_stakeholder(name or email or "", brief.stakeholders)
            # Only use the model's note if it's grounded in a real citation.
            if (
                entry
                and entry.cares_about
                and is_valid_citation(entry.source_title or "", entry.source_date or "", valid)
            ):
                clause = (
                    f"{entry.cares_about.strip()} "
                    f"({entry.source_title}, {entry.source_date})"
                )
            else:
                clause = "prior positions not covered."

            parts.append(f"- {display} — {label} ({domain}); {clause}")

    return "\n".join(parts) + "\n"
