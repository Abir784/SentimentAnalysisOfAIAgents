from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


ARTIFACT_PATTERNS = {
    "contains_continue_reading": re.compile(r"Continue Reading", re.IGNORECASE),
    "contains_more_from": re.compile(r"More from m/", re.IGNORECASE),
    "contains_http_link": re.compile(r"https?://|\[http", re.IGNORECASE),
    "contains_ui_vote_glyph": re.compile(r"\u25b2|\u25bc|▲|▼"),
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = (len(sorted_values) - 1) * q
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return float(sorted_values[int(idx)])
    return float(sorted_values[low] * (high - idx) + sorted_values[high] * (idx - low))


def _top_counter(counter: Counter, n: int = 10) -> List[Tuple[Any, int]]:
    return [(k, int(v)) for k, v in counter.most_common(n)]


def build_eda_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    row_count = len(rows)
    if row_count == 0:
        return {
            "row_count": 0,
            "unique_posts": 0,
            "unique_authors": 0,
            "duplicate_rows_by_platform_post_comment": 0,
            "missing_rate": {},
            "level_counts": {},
            "verified_counts": {},
            "upvotes_stats": {},
            "text_length_chars_stats": {},
            "text_length_words_stats": {},
            "top_posts_by_comment_count": [],
            "top_authors_by_comment_count": [],
            "artifact_counts": {},
            "artifact_rates": {},
        }

    fields = [
        "platform",
        "post_id",
        "thread_id",
        "comment_id",
        "parent_id",
        "level",
        "author_id",
        "relative_time",
        "is_verified",
        "upvotes",
        "text",
        "source_url",
        "fetched_at",
    ]

    missing_rate: Dict[str, float] = {}
    for field in fields:
        missing_count = sum(1 for r in rows if r.get(field) in (None, ""))
        missing_rate[field] = round(missing_count / row_count, 4)

    key3 = [
        (
            r.get("platform"),
            r.get("post_id"),
            r.get("comment_id"),
        )
        for r in rows
    ]
    duplicate_rows = len(key3) - len(set(key3))

    post_counts = Counter(r.get("post_id") for r in rows)
    author_counts = Counter(r.get("author_id") for r in rows)
    level_counts = Counter(r.get("level") for r in rows)
    verified_counts = Counter(bool(r.get("is_verified")) for r in rows)

    upvotes: List[float] = []
    text_lengths_chars: List[int] = []
    text_lengths_words: List[int] = []
    artifact_counts = {key: 0 for key in ARTIFACT_PATTERNS}

    for row in rows:
        upvote = row.get("upvotes")
        if isinstance(upvote, (int, float)):
            upvotes.append(float(upvote))

        text = str(row.get("text", "")).strip()
        text_lengths_chars.append(len(text))
        text_lengths_words.append(len(text.split()))

        for key, pattern in ARTIFACT_PATTERNS.items():
            if pattern.search(text):
                artifact_counts[key] += 1

    upvotes_stats = {
        "count": len(upvotes),
        "mean": round(statistics.fmean(upvotes), 4) if upvotes else 0.0,
        "median": _quantile(upvotes, 0.5) if upvotes else 0.0,
        "p90": _quantile(upvotes, 0.9) if upvotes else 0.0,
        "max": max(upvotes) if upvotes else 0.0,
    }

    text_length_chars_stats = {
        "mean": round(statistics.fmean(text_lengths_chars), 4),
        "median": _quantile(text_lengths_chars, 0.5),
        "p90": _quantile(text_lengths_chars, 0.9),
        "max": max(text_lengths_chars),
        "empty_text_rows": sum(1 for value in text_lengths_chars if value == 0),
    }

    text_length_words_stats = {
        "mean": round(statistics.fmean(text_lengths_words), 4),
        "median": _quantile(text_lengths_words, 0.5),
        "p90": _quantile(text_lengths_words, 0.9),
        "max": max(text_lengths_words),
    }

    artifact_rates = {
        key: round(value / row_count, 4) for key, value in artifact_counts.items()
    }

    summary = {
        "row_count": row_count,
        "unique_posts": len(post_counts),
        "unique_authors": len(author_counts),
        "duplicate_rows_by_platform_post_comment": int(duplicate_rows),
        "missing_rate": missing_rate,
        "level_counts": {str(k): int(v) for k, v in level_counts.items()},
        "verified_counts": {str(k): int(v) for k, v in verified_counts.items()},
        "upvotes_stats": upvotes_stats,
        "text_length_chars_stats": text_length_chars_stats,
        "text_length_words_stats": text_length_words_stats,
        "top_posts_by_comment_count": _top_counter(post_counts, n=10),
        "top_authors_by_comment_count": _top_counter(author_counts, n=10),
        "artifact_counts": artifact_counts,
        "artifact_rates": artifact_rates,
    }

    return summary
