"""Check required data inputs for coping-dynamics-sequencing figures.

Prints resolved paths from src.config and whether they exist. If any are missing,
searches under the freezing directory root for likely matches.
"""
from pathlib import Path

from src import config


def _print_item(name, path):
    exists = Path(path).exists()
    print(f"{name}: {path} (exists={exists})")
    return exists


def _search_under(root: Path, pattern: str):
    if root is None or not root.exists():
        return []
    return list(root.rglob(pattern))


def main():
    print("Resolved paths from config:")
    all_ok = True
    all_ok &= _print_item("FREEZING_DIR", config.FREEZING_DIR)
    all_ok &= _print_item("DATA_DIR", config.DATA_DIR)
    all_ok &= _print_item("PROCESSED_DIR", config.PROCESSED_DIR)
    all_ok &= _print_item("METADATA_DIR", config.METADATA_DIR)
    all_ok &= _print_item("INDEX_CSV", config.INDEX_CSV)
    all_ok &= _print_item("RESULTS_RAW_PKL", config.RESULTS_RAW_PKL)
    all_ok &= _print_item("RESULTS_CLUSTERS_PKL", config.RESULTS_CLUSTERS_PKL)
    _print_item("POSE_FEATURES_PARQUET (optional)", config.POSE_FEATURES_PARQUET)

    if all_ok:
        print("\nAll required inputs exist.")
        return

    print("\nSearching under freezing directory root for missing items...")
    root = config.FREEZING_DIR.parent if config.FREEZING_DIR.exists() else None
    if root is None:
        print("Freezing directory root not available for search.")
        return

    patterns = {
        "RESULTS_RAW_PKL": "new_results.pkl",
        "RESULTS_CLUSTERS_PKL": "new_results_clusters.pkl",
        # Pose features are optional for Figure 9
        "INDEX_CSV": "index.csv",
    }

    for key, pattern in patterns.items():
        hits = _search_under(root, pattern)
        if hits:
            print(f"{key}: found {len(hits)} match(es)")
            for p in hits:
                print(f"  - {p}")
        else:
            print(f"{key}: no matches under {root}")


if __name__ == "__main__":
    main()

