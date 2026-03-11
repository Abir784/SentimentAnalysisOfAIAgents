"""Process raw JSONL files into staged comments, tracking which files have been processed."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _extract_comment_rows(raw_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract comment rows from raw post records."""
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


def main() -> None:
    raw_dir = Path("data/raw")
    staged_dir = Path("data/staged")
    staged_dir.mkdir(parents=True, exist_ok=True)
    
    # Consolidated output file
    comments_path = staged_dir / "moltbook_comments_all.jsonl"
    
    # Tracking file to remember which raw files have been processed
    tracking_path = staged_dir / ".processed_raw_files.json"
    
    # Load previously processed files
    processed_files = _load_processed_files(tracking_path)
    
    # Find all raw JSONL files
    raw_files = sorted([f for f in raw_dir.glob("moltbook_raw_*.jsonl") 
                        if f.name not in processed_files])
    
    if not raw_files:
        print("No new raw files to process.")
        return
    
    print(f"Found {len(raw_files)} new raw files to process")
    
    # Process each raw file
    total_comments_extracted = 0
    for raw_file in raw_files:
        print(f"\nProcessing: {raw_file.name}")
        
        # Read raw records from file
        raw_records = _read_jsonl(raw_file)
        print(f"  Read {len(raw_records)} raw records")
        
        # Extract comments
        comment_rows = _extract_comment_rows(raw_records)
        print(f"  Extracted {len(comment_rows)} comments")
        
        # Append to consolidated comments file
        if comment_rows:
            _append_jsonl(comments_path, comment_rows)
            total_comments_extracted += len(comment_rows)
        
        # Mark this file as processed
        processed_files.add(raw_file.name)
    
    # Save updated tracking file
    _save_processed_files(tracking_path, processed_files)
    
    # Print summary
    print("\n" + "="*60)
    print(f"Processing complete")
    print(f"Files processed: {len(raw_files)}")
    print(f"Total comments extracted: {total_comments_extracted}")
    print(f"Output: {comments_path}")
    print(f"Tracking file: {tracking_path}")
    
    # Show totals
    total_comments = sum(1 for _ in open(comments_path)) if comments_path.exists() else 0
    print(f"Total rows in {comments_path.name}: {total_comments}")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL file and return list of dictionaries."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _append_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Append rows to a JSONL file, creating it if it doesn't exist."""
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _load_processed_files(tracking_path: Path) -> Set[str]:
    """Load the set of already-processed raw files."""
    if not tracking_path.exists():
        return set()
    
    try:
        data = json.loads(tracking_path.read_text(encoding="utf-8"))
        return set(data.get("processed_files", []))
    except (json.JSONDecodeError, IOError):
        return set()


def _save_processed_files(tracking_path: Path, processed_files: Set[str]) -> None:
    """Save the set of processed raw files to tracking file."""
    data = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "processed_files": sorted(list(processed_files))
    }
    tracking_path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


if __name__ == "__main__":
    main()
