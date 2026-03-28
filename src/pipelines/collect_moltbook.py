from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

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
    
    # Raw files remain timestamped for audit trail
    raw_path = output_dir_raw / f"moltbook_raw_{run_id}.jsonl"
    
    # Comments appended to consolidated file
    comments_path = output_dir_staged / "moltbook_comments_all.jsonl"

    collector_cfg = config["collector"]
    source_type = "url"
    raw_records, collect_meta = _collect_raw_records(collector_cfg, output_dir_raw, output_dir_staged)
    normalized_records = normalize_batch(raw_records)
    comment_rows = _extract_comment_rows(raw_records)

    if raw_records:
        # Write timestamped raw file (original behavior)
        _write_jsonl(raw_path, raw_records)
        # Append comments to consolidated file
        _append_jsonl(comments_path, comment_rows)
    else:
        raw_path = None
        comments_path = None

    result = {
        "run_id": run_id,
        "raw_path": str(raw_path) if raw_path else "",
        "comments_path": str(comments_path) if comments_path else "",
        "raw_count": len(raw_records),
        "comments_extracted": len(comment_rows),
        "source_type": source_type,
        "requested_urls": collect_meta.get("requested_urls", 0),
        "scraped_urls": collect_meta.get("scraped_urls", 0),
        "skipped_existing_urls": collect_meta.get("skipped_existing_urls", 0),
    }

    # Report storage totals
    result.update(_build_storage_totals(output_dir_raw, comments_path))
    return result


def _collect_raw_records(
    collector_cfg: Dict[str, Any],
    output_dir_raw: Path,
    output_dir_staged: Path,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
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
    input_urls = _dedupe_urls(_get_urls(collector_cfg))
    requested_urls = len(input_urls)

    processed_urls_path = _resolve_processed_urls_path(collector_cfg, output_dir_staged)
    processed_urls = _load_processed_urls_registry(processed_urls_path)

    urls = list(input_urls)

    skip_existing_urls = bool(collector_cfg.get("skip_existing_urls", True))
    if skip_existing_urls:
        existing_urls = _load_existing_urls(output_dir_raw)
        seen_urls = existing_urls.union(processed_urls)
        urls = [u for u in urls if u not in seen_urls]

    records = [scraper.scrape_post(url) for url in urls]

    # Persist successfully processed links so repeated entries in moltbook_urls.txt
    # are not scraped again in subsequent runs.
    if input_urls:
        processed_urls.update(input_urls)
        _save_processed_urls_registry(processed_urls_path, processed_urls)

    return records, {
        "requested_urls": requested_urls,
        "scraped_urls": len(urls),
        "skipped_existing_urls": requested_urls - len(urls),
    }


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
        cleaned = _normalize_url(url)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _normalize_url(url: str) -> str:
    cleaned = str(url).strip()
    if not cleaned:
        return ""
    return cleaned.rstrip("/")


def _load_existing_urls(raw_dir: Path) -> Set[str]:
    existing_urls: Set[str] = set()
    if not raw_dir.exists():
        return existing_urls

    # Read from timestamped raw files
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
                normalized = _normalize_url(str(url)) if isinstance(url, str) else ""
                if normalized:
                    existing_urls.add(normalized)

    return existing_urls


def _resolve_processed_urls_path(collector_cfg: Dict[str, Any], output_dir_staged: Path) -> Path:
    configured = str(collector_cfg.get("processed_urls_path", "")).strip()
    if configured:
        return Path(configured)
    return output_dir_staged / ".processed_moltbook_urls.json"


def _load_processed_urls_registry(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()

    urls = payload.get("processed_urls", [])
    if not isinstance(urls, list):
        return set()
    return {u for u in (_normalize_url(str(x)) for x in urls) if u}


def _save_processed_urls_registry(path: Path, urls: Set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "processed_urls": sorted(urls),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_storage_totals(output_dir_raw: Path, comments_path: Optional[Path]) -> Dict[str, int]:
    """Report storage totals for timestamped raw files and consolidated comments file."""
    raw_files = list(output_dir_raw.glob("moltbook_raw_*.jsonl"))
    
    comments_rows = 0
    if comments_path is not None and comments_path.exists():
        comments_rows = _count_jsonl_rows([comments_path])

    return {
        "stored_raw_files_total": len(raw_files),
        "stored_raw_rows_total": _count_jsonl_rows(raw_files),
        "total_comments_rows": comments_rows,
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


def _append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Append rows to a JSONL file, creating it if it doesn't exist."""
    with path.open("a", encoding="utf-8") as f:
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
