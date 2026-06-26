"""Call the local Qwen model (via Ollama) to generate a brief and to judge one."""
from __future__ import annotations

import json

import ollama
from pydantic import BaseModel, ValidationError

import config
from prompt import SYSTEM_PROMPT, Brief, build_user_message


def _client() -> "ollama.Client":
    return ollama.Client(host=config.OLLAMA_HOST)


def generate_brief(meeting, retrieved: list) -> Brief:
    """
    Build the user message, send it to the model with `SYSTEM_PROMPT`, and return
    the parsed `Brief`. Ollama is given the Brief JSON schema so output is
    structurally constrained; one repair retry covers a rare invalid-JSON reply.
    """
    user_message = build_user_message(meeting, retrieved)
    client = _client()
    # Pass the Pydantic JSON schema as `format` so the model must emit valid
    # Brief-shaped JSON.
    schema = Brief.model_json_schema()

    last_err: Exception | None = None
    for attempt in range(2):  # original try + one repair retry
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        if attempt == 1:
            # Repair turn: nudge the model to emit ONLY valid JSON for the schema.
            messages.append({
                "role": "user",
                "content": "Your previous reply was not valid JSON for the schema. "
                           "Reply again with ONLY the JSON object, nothing else.",
            })

        response = client.chat(
            model=config.OLLAMA_MODEL,
            messages=messages,
            format=schema,
            options={"temperature": config.LLM_TEMPERATURE,
                     "num_predict": config.LLM_MAX_TOKENS},
        )
        content = response["message"]["content"]
        try:
            return Brief.model_validate_json(content)
        except (ValidationError, json.JSONDecodeError) as e:
            last_err = e  # try once more, then give up

    raise ValueError(f"Model did not return valid Brief JSON: {last_err}")


# --- Quality judge (the 40% half of the eval) ---------------------------------

class _QualityVerdict(BaseModel):
    """Shape the judge must return: a 0-100 score plus a one-line justification."""
    score: int
    justification: str


_JUDGE_SYSTEM_PROMPT = """You are a strict reviewer of meeting-prep briefs. Score the
brief from 0 to 100 on how clear, well-organised, and useful it is to someone about
to walk into the meeting. Reward concrete, specific, well-cited points; penalise
vagueness, padding, and anything unsupported. Return JSON with `score` (integer
0-100) and a one-sentence `justification`."""


def judge_quality(meeting, brief_markdown: str) -> _QualityVerdict:
    """
    Ask the model to grade a finished brief 0-100. Used by eval.py for the quality
    portion of the score. Returns a `_QualityVerdict`.
    """
    client = _client()
    user = (
        f"Upcoming meeting: {meeting.title}\n\n"
        f"Brief to score:\n\n{brief_markdown}"
    )
    response = client.chat(
        model=config.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        format=_QualityVerdict.model_json_schema(),
        options={"temperature": config.LLM_TEMPERATURE},
    )
    verdict = _QualityVerdict.model_validate_json(response["message"]["content"])
    # Clamp defensively in case the model strays out of range.
    verdict.score = max(0, min(100, verdict.score))
    return verdict
