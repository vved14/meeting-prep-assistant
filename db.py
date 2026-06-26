"""Fetch the target meeting and all earlier meetings from the event log."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import json

import psycopg2
from psycopg2.extras import RealDictCursor

import config


@dataclass
class Meeting:
    """
    The clean shape the rest of the pipeline expects, regardless of how messy the
    underlying events are. Reassembled from several event rows per meeting.
    """
    id: str
    title: str
    created_at: Optional[datetime]
    attendees: list[dict] = field(default_factory=list)
    transcript_text: Optional[str] = None
    summary_markdown: Optional[str] = None

    @property
    def attendee_names(self) -> str:
        names = [a.get("name") or a.get("email") or "" for a in self.attendees]
        return ", ".join(n for n in names if n)


def get_connection():
    """Open and return a psycopg2 connection using config.DATABASE_URL."""
    return psycopg2.connect(config.DATABASE_URL)


def fetch_meeting(conn, meeting_id: str) -> Optional[Meeting]:
    """Return the single meeting whose tl;dv id == meeting_id, or None."""
    sql = """
        SELECT
            split_part(source_external_ref, ':', 3) AS id,
            MAX(payload->>'meeting_title')
                FILTER (WHERE kind = 'meeting.record.requested') AS title,
            MAX((payload ->>'scheduled_start')::timestamptz)
                FILTER (WHERE kind = 'meeting.record.requested') AS scheduled_start,
            MAX(payload->>'attendees')
                FILTER (WHERE kind = 'meeting.record.requested') AS attendees,
            MAX(payload->>'transcript_text')
                FILTER (WHERE kind = 'meeting.transcript.ready') AS transcript_text,
            MAX(payload->>'summary_markdown')
                FILTER (WHERE kind = 'meeting.summary.ready') AS summary_markdown
        FROM core.events
        WHERE source_external_ref = %s
        GROUP BY source_external_ref; 
    """
    ref = f"meeting:tldv:{meeting_id}"

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (ref,))
        row = cur.fetchone()

    if row is None:
        return None

    raw_attendees = row["attendees"] or []
    if isinstance(raw_attendees, str):
        raw_attendees = json.loads(raw_attendees)

    attendees = [
        {"name": a.get("display_name"), "email": a.get("email")}
        for a in raw_attendees
    ]

    return Meeting(
        id=row["id"],
        title=row["title"],
        created_at=row["scheduled_start"],
        attendees=attendees,
        transcript_text=row["transcript_text"],
        summary_markdown=row["summary_markdown"],
    )


def fetch_past_meetings(conn, before: datetime) -> list[Meeting]:
    """Return every meeting that happened strictly before `before` and has a
    transcript, oldest first."""
    sql = """
        SELECT
            split_part(source_external_ref, ':', 3) AS id,
            MAX(payload->>'meeting_title')
                FILTER (WHERE kind = 'meeting.record.requested') AS title,
            MAX((payload ->>'scheduled_start')::timestamptz)
                FILTER (WHERE kind = 'meeting.record.requested') AS scheduled_start,
            MAX(payload->>'attendees')
                FILTER (WHERE kind = 'meeting.record.requested') AS attendees,
            MAX(payload->>'transcript_text')
                FILTER (WHERE kind = 'meeting.transcript.ready') AS transcript_text,
            MAX(payload->>'summary_markdown')
                FILTER (WHERE kind = 'meeting.summary.ready') AS summary_markdown
        FROM core.events
        GROUP BY source_external_ref
        HAVING MAX((payload->>'scheduled_start')::timestamptz)
                    FILTER (WHERE kind = 'meeting.record.requested') < %s
            AND MAX(payload->>'transcript_text')
                  FILTER (WHERE kind = 'meeting.transcript.ready') IS NOT NULL
        ORDER BY scheduled_start ASC;
    """
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, (before,))
        rows = cur.fetchall()

    past_meetings = []
    for row in rows:
        raw_attendees = row["attendees"] or []
        if isinstance(raw_attendees, str):
            raw_attendees = json.loads(raw_attendees)

        attendees = [
            {"name": a.get("display_name"), "email": a.get("email")}
            for a in raw_attendees
        ]

        past_meetings.append(
            Meeting(
                id=row["id"],
                title=row["title"],
                created_at=row["scheduled_start"],
                attendees=attendees,
                transcript_text=row["transcript_text"],
                summary_markdown=row["summary_markdown"],
            )
        )

    return past_meetings

