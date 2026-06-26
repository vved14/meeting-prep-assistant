"""Split a long transcript into smaller, overlapping chunks of text."""
from __future__ import annotations

import config

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split `text` into chunks of roughly `chunk_size` characters, carrying
    `overlap` characters between consecutive chunks.
    """
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return splitter.split_text(text)
