#!/usr/bin/env python
"""
Script to clean up old data files, keeping only the latest version of each.

This script removes older versions of data files from active rule-based folders:
- data/preprocessed_rule_based/
- data/eda_rule_based/
- data/features_rule_based/
- data/rule_based/
- data/eda/ (RQ1 interaction artifacts)

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
        ("preprocessed_rule_based", "moltbook_preprocessed_rule_based_*.csv", 1),
        ("preprocessed_rule_based", "moltbook_preprocessed_rule_based_*.jsonl", 1),
        ("preprocessed_rule_based", "moltbook_preprocessed_rule_based_summary_*.json", 1),
        ("eda_rule_based", "moltbook_eda_rule_based_summary_*.json", 1),
        ("features_rule_based", "moltbook_features_rule_based_*.csv", 1),
        ("features_rule_based", "moltbook_features_rule_based_summary_*.json", 1),
        ("rule_based", "moltbook_rule_based_comments_*.csv", 1),
        ("rule_based", "moltbook_rule_based_summary_*.json", 1),
        ("rule_based", "moltbook_rule_based_label_share_*.png", 1),
        ("rule_based", "moltbook_rule_based_score_distribution_*.png", 1),
        ("eda", "moltbook_interaction_network_summary_*.json", 1),
        ("eda", "moltbook_interaction_network_nodes_*.csv", 1),
        ("eda", "moltbook_interaction_network_edges_*.csv", 1),
        ("eda", "moltbook_interaction_network_thread_stats_*.csv", 1),
        ("eda", "moltbook_interaction_network_topology_*.png", 1),
        ("eda", "moltbook_interaction_network_distributions_*.png", 1),
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
