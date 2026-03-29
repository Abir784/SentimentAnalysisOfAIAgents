#!/usr/bin/env python
"""
Script to clean up old data files, keeping only the latest version of each.

This script removes older versions of data files from:
- data/modeling/ (predictions and summaries)
- data/polarity/ (polarity JSONL and summaries)  
- data/preprocessed/ (preprocessed JSONL and training CSV)
- data/eda/ (EDA summaries only)

Modeling graphs are intentionally preserved for every run.

Raw and staged folders are NOT cleaned up (per requirements).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.file_management import cleanup_old_files


def main() -> None:
    """Clean up all data folders, keeping only latest files."""
    
    data_root = Path("data")
    
    # Define cleanup rules: (folder, pattern, keep_count)
    cleanup_rules = [
        ("modeling", "moltbook_model_predictions_*.csv", 1),
        ("modeling", "moltbook_model_summary_*.json", 1),
        ("polarity", "moltbook_comments_polarity_*.jsonl", 1),
        ("polarity", "moltbook_polarity_summary_*.json", 1),
        ("preprocessed", "moltbook_comments_preprocessed_*.jsonl", 1),
        ("preprocessed", "moltbook_training_ready_*.csv", 1),
        ("eda", "moltbook_eda_summary_*.json", 1),
    ]
    
    print("Cleaning up old data files...")
    print("-" * 60)
    
    total_deleted = 0
    
    for folder_name, pattern, keep_count in cleanup_rules:
        folder = data_root / folder_name
        
        if not folder.exists():
            continue
        
        deleted_files = cleanup_old_files(folder, pattern, keep_latest=keep_count)
        
        if deleted_files:
            print(f"\n{folder_name}/ - Pattern: {pattern}")
            for f in deleted_files:
                print(f"  DELETED: {f.name}")
            total_deleted += len(deleted_files)
        else:
            # Only print if folder has files
            files = list(folder.glob(pattern))
            if files:
                print(f"\n{folder_name}/ - Pattern: {pattern}")
                print(f"  Kept: {files[0].name if files else '(none)'}")
    
    print("\n" + "-" * 60)
    print(f"Total files deleted: {total_deleted}")
    print("\nCleanup complete!")
    print("\nNote: raw/ and staged/ folders were NOT modified per configuration.")


if __name__ == "__main__":
    main()
