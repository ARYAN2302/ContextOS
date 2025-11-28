"""Text processing utilities."""

import re


def clean_text(text: str) -> str:
    """Clean and normalize text input."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def estimate_tokens(text: str) -> int:
    """Estimate token count (approx 4 chars per token)."""
    return len(text) // 4
