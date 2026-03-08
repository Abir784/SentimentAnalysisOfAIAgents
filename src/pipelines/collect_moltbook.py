from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.collectors.moltbook_client import MoltBookClient, MoltBookClientConfig
from src.collectors.moltbook_scraper import MoltBookScraper, MoltBookScraperConfig
from src.pipelines.normalize_moltbook import normalize_batch


def run_collection(config_path: Path) -> Dict[str, Any]:
    config = _load_config(config_path)
    return run_collection_from_config(config)


def run_collection_from_config(config: Dict[str, Any]) -> Dict[str, Any]:

    output_dir_raw = Path(config["output"]["raw_dir"])
    output_dir_staged = Path(config["output"]["staged_dir"])
    output_dir_raw.mkdir(parents=True, exist_ok=True)
    output_dir_staged.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = output_dir_raw / f"moltbook_raw_{run_id}.jsonl"
    staged_path = output_dir_staged / f"moltbook_normalized_{run_id}.jsonl"
    comments_path = output_dir_staged / f"moltbook_comments_with_levels_{run_id}.jsonl"

    collector_cfg = config["collector"]
    source_type = collector_cfg.get("source_type", "api").lower()
    raw_records, collect_meta = _collect_raw_records(collector_cfg, source_type, output_dir_raw)
    normalized_records = normalize_batch(raw_records)
    comment_rows = _extract_comment_rows(raw_records)

    if raw_records:
        _write_jsonl(raw_path, raw_records)
        _write_jsonl(staged_path, normalized_records)
        _write_jsonl(comments_path, comment_rows)
    else:
        raw_path = None
        staged_path = None
        comments_path = None

    result = {
        "run_id": run_id,
        "raw_path": str(raw_path) if raw_path else "",
        "staged_path": str(staged_path) if staged_path else "",
        "comments_path": str(comments_path) if comments_path else "",
        "raw_count": len(raw_records),
        "normalized_count": len(normalized_records),
        "comments_extracted": len(comment_rows),
        "source_type": source_type,
        "requested_urls": collect_meta.get("requested_urls", 0),
        "scraped_urls": collect_meta.get("scraped_urls", 0),
        "skipped_existing_urls": collect_meta.get("skipped_existing_urls", 0),
    }

    # Report cumulative rows/files so each run shows data stored till now.
    result.update(_build_storage_totals(output_dir_raw, output_dir_staged))
    return result


def _collect_raw_records(
    collector_cfg: Dict[str, Any],
    source_type: str,
    output_dir_raw: Path,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if source_type == "url":
        scraper = MoltBookScraper(
            MoltBookScraperConfig(
                timeout_seconds=int(collector_cfg.get("timeout_seconds", 30)),
                user_agent=collector_cfg.get(
                    "user_agent",
                    "Mozilla/5.0 (compatible; SentimentAnalysisBot/1.0)",
                ),
                use_reader_fallback=bool(collector_cfg.get("use_reader_fallback", True)),
            )
        )
        urls = _dedupe_urls(_get_urls(collector_cfg))
        requested_urls = len(urls)

        skip_existing_urls = bool(collector_cfg.get("skip_existing_urls", True))
        if skip_existing_urls:
            existing_urls = _load_existing_urls(output_dir_raw)
            urls = [u for u in urls if u not in existing_urls]

        records = [scraper.scrape_post(url) for url in urls]
        return records, {
            "requested_urls": requested_urls,
            "scraped_urls": len(urls),
            "skipped_existing_urls": requested_urls - len(urls),
        }

    client_cfg = MoltBookClientConfig(
        base_url=collector_cfg["base_url"],
        posts_endpoint=collector_cfg.get("posts_endpoint", "/posts"),
        api_key=collector_cfg.get("api_key"),
        timeout_seconds=int(collector_cfg.get("timeout_seconds", 30)),
        page_size=int(collector_cfg.get("page_size", 100)),
        max_pages=int(collector_cfg.get("max_pages", 10)),
    )
    client = MoltBookClient(client_cfg)

    since_iso: Optional[str] = collector_cfg.get("since_iso")
    topic: Optional[str] = collector_cfg.get("topic")
    records = list(client.fetch_posts(since_iso=since_iso, topic=topic))
    return records, {}


def _get_urls(collector_cfg: Dict[str, Any]) -> List[str]:
    urls = collector_cfg.get("urls")
    if isinstance(urls, list) and urls:
        return [str(u) for u in urls if str(u).strip()]

    single_url = collector_cfg.get("url")
    if isinstance(single_url, str) and single_url.strip():
        return [single_url.strip()]

    raise ValueError("When source_type is 'url', provide collector.url or collector.urls.")


def _dedupe_urls(urls: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for url in urls:
        cleaned = str(url).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _load_existing_urls(raw_dir: Path) -> Set[str]:
    existing_urls: Set[str] = set()
    if not raw_dir.exists():
        return existing_urls

    for path in raw_dir.glob("moltbook_raw_*.jsonl"):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                payload = row.get("source_payload", {})
                if not isinstance(payload, dict):
                    continue
                url = payload.get("url")
                if isinstance(url, str) and url.strip():
                    existing_urls.add(url.strip())

    return existing_urls


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_storage_totals(output_dir_raw: Path, output_dir_staged: Path) -> Dict[str, int]:
    raw_files = list(output_dir_raw.glob("moltbook_raw_*.jsonl"))
    normalized_files = list(output_dir_staged.glob("moltbook_normalized_*.jsonl"))
    comments_files = list(output_dir_staged.glob("moltbook_comments_with_levels_*.jsonl"))

    return {
        "stored_raw_files_total": len(raw_files),
        "stored_raw_rows_total": _count_jsonl_rows(raw_files),
        "stored_normalized_files_total": len(normalized_files),
        "stored_normalized_rows_total": _count_jsonl_rows(normalized_files),
        "stored_comments_files_total": len(comments_files),
        "stored_comments_rows_total": _count_jsonl_rows(comments_files),
    }


def _count_jsonl_rows(paths: List[Path]) -> int:
    count = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    return count


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _extract_comment_rows(raw_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in raw_records:
        payload = raw.get("source_payload", {})
        comments = payload.get("comments", [])
        if not isinstance(comments, list):
            continue

        post_id = payload.get("post_id")
        source_url = payload.get("url")
        fetched_at = raw.get("fetched_at")

        for item in comments:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue

            rows.append(
                {
                    "platform": "moltbook",
                    "post_id": post_id,
                    "thread_id": post_id,
                    "comment_id": item.get("comment_id"),
                    "parent_id": item.get("parent_id") or post_id,
                    "level": int(item.get("level", 0)),
                    "author_id": item.get("author_id"),
                    "relative_time": item.get("relative_time"),
                    "is_verified": bool(item.get("is_verified", False)),
                    "upvotes": item.get("upvotes"),
                    "text": text,
                    "source_url": source_url,
                    "fetched_at": fetched_at,
                }
            )

    return rows
