from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipelines.collect_moltbook import run_collection, run_collection_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MoltBook collection and normalization.")
    parser.add_argument(
        "--config",
        default="configs/moltbook_collection.url.json",
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        help="MoltBook post URL. Repeat --url for multiple links.",
    )
    parser.add_argument(
        "--urls-file",
        default="",
        help="Optional text file with one URL per line.",
    )
    args = parser.parse_args()

    cli_urls = _read_cli_urls(args.url, args.urls_file)
    if cli_urls:
        config = _load_config(Path(args.config))
        config.setdefault("collector", {})
        config["collector"]["source_type"] = "url"
        config["collector"]["urls"] = cli_urls
        config["collector"]["skip_existing_urls"] = True
        result = run_collection_from_config(config)
    else:
        result = run_collection(Path(args.config))

    print("Run complete")
    for key, value in result.items():
        print(f"{key}: {value}")


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_cli_urls(url_args: List[str], urls_file: str) -> List[str]:
    urls: List[str] = []

    for item in url_args:
        cleaned = str(item).strip()
        if cleaned:
            urls.append(cleaned)

    if urls_file:
        path = Path(urls_file)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                cleaned = line.strip()
                if cleaned and not cleaned.startswith("#"):
                    urls.append(cleaned)

    seen = set()
    deduped: List[str] = []
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


if __name__ == "__main__":
    main()
