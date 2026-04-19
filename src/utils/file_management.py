"""Utility functions for managing data files and keeping only latest versions."""

import re
from pathlib import Path
from typing import Dict, List, Optional


def extract_timestamp(filename: str) -> Optional[str]:
    """Extract timestamp from filename pattern like 'prefix_YYYYMMDDTHHMMSSZ.ext'."""
    match = re.search(r'(\d{8}T\d{6}Z)', filename)
    return match.group(1) if match else None


def cleanup_old_files(
    folder: Path,
    pattern: str,
    keep_latest: int = 1,
    dry_run: bool = False
) -> List[Path]:
    """
    Keep only the latest N files matching pattern, delete others.
    
    Args:
        folder: Directory to clean up
        pattern: Glob pattern (e.g., 'moltbook_training_ready_*.csv')
        keep_latest: Number of latest files to keep (default 1)
        dry_run: If True, don't actually delete, just return what would be deleted
    
    Returns:
        List of deleted file paths
    """
    if not folder.exists():
        return []
    
    files = sorted(folder.glob(pattern))
    if len(files) <= keep_latest:
        return []
    
    # Sort by timestamp if available, otherwise by modification time
    def get_sort_key(f):
        timestamp = extract_timestamp(f.name)
        if timestamp:
            return timestamp
        return str(f.stat().st_mtime)
    
    files_sorted = sorted(files, key=get_sort_key)
    files_to_delete = files_sorted[:-keep_latest]
    
    deleted = []
    for f in files_to_delete:
        if not dry_run:
            f.unlink()
        deleted.append(f)
    
    return deleted


def get_latest_file(folder: Path, pattern: str) -> Optional[Path]:
    """Get the most recent file matching pattern in folder."""
    if not folder.exists():
        return None
    
    files = sorted(folder.glob(pattern))
    if not files:
        return None
    
    # Sort by timestamp if available, otherwise by modification time
    def get_sort_key(f):
        timestamp = extract_timestamp(f.name)
        if timestamp:
            return timestamp
        return str(f.stat().st_mtime)
    
    return max(files, key=get_sort_key) if files else None


def cleanup_data_folders() -> Dict[str, List[Path]]:
    """
    Clean up old files from data folders, keeping only the latest version of each.
    
    Returns:
        Dictionary mapping folder names to lists of deleted files
    """
    data_root = Path("data")
    deleted = {}
    
    # Define cleanup rules: folder -> (pattern, keep_count)
    cleanup_rules = {
        "preprocessed_rule_based": [
            ("moltbook_preprocessed_rule_based_*.csv", 1),
            ("moltbook_preprocessed_rule_based_*.jsonl", 1),
            ("moltbook_preprocessed_rule_based_summary_*.json", 1),
        ],
        "eda_rule_based": [
            ("moltbook_eda_rule_based_summary_*.json", 1),
        ],
        "features_rule_based": [
            ("moltbook_features_rule_based_*.csv", 1),
            ("moltbook_features_rule_based_summary_*.json", 1),
        ],
        "rule_based": [
            ("moltbook_rule_based_comments_*.csv", 1),
            ("moltbook_rule_based_summary_*.json", 1),
            ("moltbook_rule_based_label_share_*.png", 1),
            ("moltbook_rule_based_score_distribution_*.png", 1),
        ],
        "eda": [
            ("moltbook_interaction_network_summary_*.json", 1),
            ("moltbook_interaction_network_nodes_*.csv", 1),
            ("moltbook_interaction_network_edges_*.csv", 1),
            ("moltbook_interaction_network_thread_stats_*.csv", 1),
            ("moltbook_interaction_network_topology_*.png", 1),
            ("moltbook_interaction_network_distributions_*.png", 1),
        ],
    }
    
    for folder_name, patterns in cleanup_rules.items():
        folder = data_root / folder_name
        deleted[folder_name] = []
        
        for pattern, keep_count in patterns:
            deleted_files = cleanup_old_files(folder, pattern, keep_latest=keep_count)
            deleted[folder_name].extend(deleted_files)
    
    return deleted
