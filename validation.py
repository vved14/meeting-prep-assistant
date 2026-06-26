"""Shared grounding checks: citation validity, speculation, and invented names."""
from __future__ import annotations

import re

# --- Citations -------------------------------------------------------------

def _norm_title(title: str) -> str:
    """Lower-case + collapse whitespace so trivial formatting differences match."""
    return re.sub(r"\s+", " ", (title or "").strip()).lower()


def _norm_date(date: str) -> str:
    """Reduce any ISO-ish date/timestamp to its YYYY-MM-DD prefix for comparison."""
    return (date or "").strip()[:10]


def valid_citation_set(retrieved: list) -> set[tuple[str, str]]:
    """
    Build the set of citations the model is *allowed* to use: one
    (normalised title, normalised date) per retrieved chunk's source meeting.

    `retrieved` is the list of (Chunk, score) tuples from VectorStore.search.
    """
    valid = set()
    for chunk, _score in retrieved:
        valid.add((_norm_title(chunk.meeting_title), _norm_date(chunk.meeting_date)))
    return valid


def is_valid_citation(title: str, date: str, valid: set[tuple[str, str]]) -> bool:
    """True when (title, date) names a meeting that actually appears in `retrieved`."""
    return (_norm_title(title), _norm_date(date)) in valid


# --- Speculation -----------------------------------------------------------

# Hedging language that has no place in a grounded claim. NOTE: deliberately NOT
# applied to the "Expected Topics" section, whose whole job is to predict.
_SPECULATION_PATTERNS = [
    r"\blikely\b",
    r"\bprobably\b",
    r"\bpresumably\b",
    r"\bperhaps\b",
    r"\bmay\b",
    r"\bmight\b",
    r"\bcould (?:contribute|be|have)\b",
    r"\bassum\w+\b",
    r"\bseems? to\b",
    r"\bappears? to\b",
    r"\bbased on (?:past|prior) roles?\b",
    r"\bexpected to contribute\b",
    r"\bsuggest\w*\b",
]
_SPECULATION_RE = re.compile("|".join(_SPECULATION_PATTERNS), re.IGNORECASE)


def find_speculation(text: str) -> list[str]:
    """Return every hedging phrase found in `text` (empty list = clean/grounded)."""
    return _SPECULATION_RE.findall(text or "")


# --- Invented names --------------------------------------------------------

# A candidate person name: two or more capitalised words in a row (e.g. "Parth
# Basole"). Heuristic — used as a soft signal, not a hard gate.
_NAME_CANDIDATE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")

# Capitalised multi-word phrases that are NOT people, to suppress false positives.
_NON_PERSON_PHRASES = {
    "current state", "open loops", "expected topics", "stakeholder context",
    "lighthouse daily", "lighthouse os", "github details",
}


def speaker_names(retrieved: list) -> set[str]:
    """
    Collect transcript speaker labels ("Name: …") from the retrieved excerpts.
    These are real people the model is allowed to mention.
    """
    names: set[str] = set()
    for chunk, _score in retrieved:
        for line in (chunk.text or "").splitlines():
            m = re.match(r"\s*([A-Z][\w .'-]{1,40}?):", line)
            if m:
                names.add(m.group(1).strip())
    return names


def allowed_names(meeting, retrieved: list) -> set[str]:
    """The full set of real people: meeting attendees ∪ excerpt speakers."""
    allowed = set(speaker_names(retrieved))
    for a in getattr(meeting, "attendees", []) or []:
        name = a.get("name")
        if name:
            allowed.add(name.strip())
    return allowed


def find_invented_names(text: str, allowed: set[str]) -> list[str]:
    """
    Return name-like phrases in `text` that don't match any allowed real person.
    Heuristic (capitalised word pairs); known non-person phrases are ignored.
    """
    allowed_lower = {n.lower() for n in allowed}
    invented = []
    for cand in _NAME_CANDIDATE_RE.findall(text or ""):
        cl = cand.lower()
        if cl in _NON_PERSON_PHRASES:
            continue
        # Accept if the candidate is (or is part of) any allowed name, or vice versa.
        if any(cl in a or a in cl for a in allowed_lower):
            continue
        invented.append(cand)
    return invented
