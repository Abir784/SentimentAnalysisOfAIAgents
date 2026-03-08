# SentimentAnalysis

## Step 1: MoltBook Data Collection

This repository now includes a first-pass MoltBook collector and normalized schema writer.

### Files
- `src/pipelines/normalize_moltbook.py`: Maps source payloads into canonical schema.
- `src/pipelines/collect_moltbook.py`: Orchestrates collection + writes JSONL outputs.
- `scripts/run_moltbook_collection.py`: CLI entry point.
- `configs/moltbook_collection.url.json`: URL-mode run configuration.

### Install
```powershell
pip install -r requirements.txt
```

### Run
```powershell
python scripts/run_moltbook_collection.py --config configs/moltbook_collection.url.json
```

Direct URL mode (no config edits required):
```powershell
python scripts/run_moltbook_collection.py --url https://www.moltbook.com/post/<id1> --url https://www.moltbook.com/post/<id2>
```

Or pass a text file with one link per line:
```powershell
python scripts/run_moltbook_collection.py --urls-file configs/moltbook_urls.txt
```

Each run prints:
- this-run totals (`raw_count`, `comments_extracted`, etc.)
- dedup stats (`requested_urls`, `scraped_urls`, `skipped_existing_urls`)
- cumulative stored totals (`stored_*_rows_total`, `stored_*_files_total`)

### Outputs
- Raw payload snapshots: `data/raw/moltbook_raw_<run_id>.jsonl`
- Normalized records: `data/staged/moltbook_normalized_<run_id>.jsonl`
- Comments with levels: `data/staged/moltbook_comments_with_levels_<run_id>.jsonl`

### Comment File Schema
- `platform`, `post_id`, `thread_id`, `comment_id`, `parent_id`
- `level`, `author_id`, `relative_time`, `is_verified`, `upvotes`
- `text`, `source_url`, `fetched_at`

### Notes
- For JS-heavy pages, the scraper uses rendered-text fallback to extract post content and comments.
- `level` is currently set to `0` because reply nesting is not exposed in the rendered feed format.

### Canonical Normalized Fields
- `platform`, `post_id`, `thread_id`, `parent_id`
- `author_id`, `author_type`, `author_model`
- `topic`, `text`, `timestamp_utc`
- `upvotes`, `reply_count`, `source_page`, `fetched_at`, `ingested_at`
