from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional

import requests


@dataclass
class MoltBookClientConfig:
    base_url: str
    posts_endpoint: str = "/posts"
    api_key: Optional[str] = None
    timeout_seconds: int = 30
    page_size: int = 100
    max_pages: int = 10


class MoltBookClient:
    def __init__(self, config: MoltBookClientConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        if config.api_key:
            self.session.headers.update({"Authorization": f"Bearer {config.api_key}"})

    def fetch_posts(
        self,
        since_iso: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Yield raw records from the MoltBook posts endpoint.

        The response parser is intentionally flexible because API schemas often evolve.
        """
        page = 1
        while page <= self.config.max_pages:
            params: Dict[str, Any] = {
                "page": page,
                "limit": self.config.page_size,
            }
            if since_iso:
                params["since"] = since_iso
            if topic:
                params["topic"] = topic

            url = f"{self.config.base_url.rstrip('/')}{self.config.posts_endpoint}"
            response = self.session.get(url, params=params, timeout=self.config.timeout_seconds)
            response.raise_for_status()
            payload = response.json()

            items = _extract_items(payload)
            if not items:
                break

            fetched_at = datetime.now(timezone.utc).isoformat()
            for item in items:
                yield {
                    "platform": "moltbook",
                    "fetched_at": fetched_at,
                    "source_payload": item,
                    "source_page": page,
                }

            page += 1


def _extract_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]

    if isinstance(payload, dict):
        for key in ("items", "posts", "data", "results"):
            if key in payload and isinstance(payload[key], list):
                return [p for p in payload[key] if isinstance(p, dict)]

    return []
