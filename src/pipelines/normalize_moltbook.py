from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


def normalize_moltbook_record(raw_record: Dict[str, Any]) -> Dict[str, Any]:
    payload = raw_record.get("source_payload", {})

    post_id = _pick(payload, ["post_id", "id", "uuid"])
    thread_id = _pick(payload, ["thread_id", "conversation_id", "root_id", "id"])
    parent_id = _pick(payload, ["parent_id", "reply_to", "in_reply_to"])
    author_id = _pick(payload, ["agent_id", "author_id", "user_id", "author"])
    author_model = _pick(payload, ["agent_model", "model", "llm", "backend_model"])
    topic = _pick(payload, ["topic", "submolt", "category", "channel"])
    text = _pick(payload, ["text", "content", "body", "message"]) or ""

    upvotes = _to_int(_pick(payload, ["upvotes", "likes", "score"]))
    reply_count = _to_int(_pick(payload, ["reply_count", "replies"]))
    timestamp = _parse_timestamp(
        _pick(payload, ["timestamp", "created_at", "createdAt", "time"])
    )

    normalized = {
        "platform": "moltbook",
        "post_id": _safe_str(post_id),
        "thread_id": _safe_str(thread_id),
        "parent_id": _safe_str(parent_id),
        "author_id": _safe_str(author_id),
        "author_type": "agent",
        "author_model": _safe_str(author_model),
        "topic": _safe_str(topic),
        "text": text.strip(),
        "timestamp_utc": timestamp,
        "upvotes": upvotes,
        "reply_count": reply_count,
        "source_page": raw_record.get("source_page"),
        "fetched_at": raw_record.get("fetched_at"),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }

    return normalized


def normalize_batch(raw_records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for raw in raw_records:
        normalized.append(normalize_moltbook_record(raw))
    return normalized


def _pick(payload: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value: Any) -> Optional[str]:
    if value is None:
        return None

    if isinstance(value, (int, float)):
        # Supports unix seconds and milliseconds.
        epoch = float(value)
        if epoch > 1e12:
            epoch = epoch / 1000.0
        return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()

    if isinstance(value, str):
        v = value.strip()
        if not v:
            return None
        if v.endswith("Z"):
            v = v[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(v)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except ValueError:
            return None

    return None
