"""Score a generated brief: deterministic guideline checks + an LLM quality judge.

Usage:  python eval.py <meeting_id>
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import config
from pipeline import build_brief
from prompt import classify_attendee
from llm import judge_quality
from validation import (
    valid_citation_set,
    is_valid_citation,
    find_speculation,
    find_invented_names,
    allowed_names,
)


@dataclass
class Check:
    """One guideline check: a 0-100 score and a human-readable detail line."""
    name: str
    score: float
    detail: str


def _sections(markdown: str) -> dict[str, list[str]]:
    """Parse the rendered Markdown into {section header: [bullet text, ...]}."""
    sections: dict[str, list[str]] = {}
    current = None
    for line in markdown.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
        elif current is not None and line.strip().startswith("- "):
            sections[current].append(line.strip()[2:])
    return sections


def run_guideline_checks(meeting, retrieved, brief, markdown) -> list[Check]:
    """The six deterministic checks. Each returns a 0-100 sub-score."""
    sections = _sections(markdown)
    valid = valid_citation_set(retrieved)
    grounded_bullets = list(brief.current_state) + list(brief.open_loops)
    # Text that MUST be grounded (excludes Expected Topics, which are predictions).
    grounded_text = " ".join(
        [b.claim for b in grounded_bullets]
        + [s.cares_about or "" for s in brief.stakeholders]
    )
    checks: list[Check] = []

    # 1. Section completeness.
    required = ["Current State", "Open Loops", "Expected Topics", "Stakeholder Context"]
    present = [r for r in required if r in sections]
    checks.append(Check(
        "section_completeness",
        100.0 * len(present) / len(required),
        f"{len(present)}/{len(required)} sections present",
    ))

    # 2. Attendee coverage — every attendee appears exactly once as a stakeholder.
    stake_lines = sections.get("Stakeholder Context", [])
    displays = [ln.split(" — ")[0].strip() for ln in stake_lines]
    attendees = meeting.attendees or []
    if attendees:
        covered = sum(
            1 for a in attendees
            if displays.count((a.get("name") or a.get("email") or "(unknown)").strip()) == 1
        )
        score = 100.0 * covered / len(attendees)
        detail = f"{covered}/{len(attendees)} attendees covered exactly once"
    else:
        score, detail = 100.0, "no attendees to cover"
    checks.append(Check("attendee_coverage", score, detail))

    # 3. internal/external classification matches the email domain.
    if attendees:
        correct = 0
        for a in attendees:
            disp = (a.get("name") or a.get("email") or "(unknown)").strip()
            label, domain = classify_attendee(a.get("email"))
            if any(ln.startswith(f"{disp} — {label} ({domain})") for ln in stake_lines):
                correct += 1
        score = 100.0 * correct / len(attendees)
        detail = f"{correct}/{len(attendees)} classified correctly"
    else:
        score, detail = 100.0, "no attendees to classify"
    checks.append(Check("classification_correctness", score, detail))

    # 4. Citation validity — fraction of the MODEL's claims with a real citation.
    if grounded_bullets:
        good = sum(
            1 for b in grounded_bullets
            if is_valid_citation(b.source_title, b.source_date, valid)
        )
        score = 100.0 * good / len(grounded_bullets)
        detail = f"{good}/{len(grounded_bullets)} claims cite a retrieved meeting"
    else:
        score, detail = 100.0, "no claims made"
    checks.append(Check("citation_validity", score, detail))

    # 5. No speculation in grounded text (20-point penalty per hedging phrase).
    spec = find_speculation(grounded_text)
    checks.append(Check(
        "no_speculation",
        max(0.0, 100.0 - 20.0 * len(spec)),
        "clean" if not spec else f"speculative phrases: {spec}",
    ))

    # 6. No invented names (20-point penalty per name not in attendees/excerpts).
    invented = find_invented_names(grounded_text, allowed_names(meeting, retrieved))
    checks.append(Check(
        "no_invented_names",
        max(0.0, 100.0 - 20.0 * len(invented)),
        "clean" if not invented else f"possible invented names: {invented}",
    ))

    return checks


def score_brief(meeting, retrieved, brief, markdown) -> dict:
    """
    Score an already-built brief (no DB/generation). Returns the scorecard dict.

    Kept separate from `evaluate` so `pipeline.run` can score a brief it just
    generated and stamp the result onto the file without regenerating it.
    """
    checks = run_guideline_checks(meeting, retrieved, brief, markdown)
    guideline_score = sum(c.score for c in checks) / len(checks)

    verdict = judge_quality(meeting, markdown)
    quality_score = float(verdict.score)

    overall = (
        config.EVAL_GUIDELINE_WEIGHT * guideline_score
        + config.EVAL_QUALITY_WEIGHT * quality_score
    )

    return {
        "meeting_id": meeting.id,
        "model": config.OLLAMA_MODEL,
        "overall_score": round(overall, 1),
        "guideline_score": round(guideline_score, 1),
        "quality_score": round(quality_score, 1),
        "weights": {
            "guideline": config.EVAL_GUIDELINE_WEIGHT,
            "quality": config.EVAL_QUALITY_WEIGHT,
        },
        "guideline_checks": [asdict(c) for c in checks],
        "quality_justification": verdict.justification,
    }


def evaluate(meeting_id: str) -> dict:
    """Build the brief, score it, write + return the scorecard."""
    meeting, retrieved, brief, markdown = build_brief(meeting_id)
    result = score_brief(meeting, retrieved, brief, markdown)

    out_dir = Path(config.EVAL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{meeting_id}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def _print_scorecard(result: dict) -> None:
    """Pretty-print the scorecard to the terminal."""
    print(f"\n=== Eval scorecard — meeting {result['meeting_id']} ({result['model']}) ===")
    print(f"  Guideline (deterministic, weight {result['weights']['guideline']}):")
    for c in result["guideline_checks"]:
        mark = "✓" if c["score"] >= 99.9 else ("~" if c["score"] >= 50 else "✗")
        print(f"    {mark} {c['name']:<28} {c['score']:5.1f}  {c['detail']}")
    print(f"    → guideline_score: {result['guideline_score']}")
    print(f"  Quality (qwen3 judge, weight {result['weights']['quality']}):")
    print(f"    score: {result['quality_score']}  — {result['quality_justification']}")
    print(f"\n  OVERALL: {result['overall_score']} / 100\n")


def main():
    parser = argparse.ArgumentParser(description="Score a generated meeting brief.")
    parser.add_argument("meeting_id", help="tl;dv id of the meeting to evaluate")
    args = parser.parse_args()
    _print_scorecard(evaluate(args.meeting_id))


if __name__ == "__main__":
    main()
