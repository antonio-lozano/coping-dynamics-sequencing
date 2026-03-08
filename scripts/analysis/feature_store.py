"""Reusable parquet-backed feature store with manifest tracking.

This module is intentionally scripts-level (not under src/) to avoid
changing core library APIs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


MANIFEST_COLUMNS = [
    "recording",
    "variant",
    "source_path",
    "source_mtime",
    "n_frames",
    "preprocess_config_hash",
    "feature_config_hash",
    "table_path_raw",
    "table_path_extended",
    "table_path_model_aligned",
    "ref_dist_px",
    "finite_ratio",
    "x_range_px",
    "y_range_px",
    "source_tag",
    "updated_at_utc",
]


@dataclass(frozen=True)
class FeatureStoreMatchKey:
    recording: str
    variant: str
    source_path: str
    source_mtime: float
    preprocess_config_hash: str
    feature_config_hash: str


def stable_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def ensure_store_dir(base_dir: Path, dataset_id: str) -> Path:
    store_dir = Path(base_dir) / str(dataset_id)
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir


def manifest_path(store_dir: Path) -> Path:
    return Path(store_dir) / "manifest.parquet"


def load_manifest(store_dir: Path) -> pd.DataFrame:
    path = manifest_path(store_dir)
    if not path.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    df = pd.read_parquet(path)
    for col in MANIFEST_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[MANIFEST_COLUMNS].copy()


def save_manifest(store_dir: Path, manifest_df: pd.DataFrame) -> None:
    out = manifest_df.copy()
    for col in MANIFEST_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[MANIFEST_COLUMNS]
    path = manifest_path(store_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def _entry_matches(df: pd.DataFrame, key: FeatureStoreMatchKey) -> pd.Series:
    return (
        (df["recording"].astype(str) == str(key.recording))
        & (df["variant"].astype(str) == str(key.variant))
        & (df["source_path"].astype(str) == str(key.source_path))
        & (pd.to_numeric(df["source_mtime"], errors="coerce") == float(key.source_mtime))
        & (df["preprocess_config_hash"].astype(str) == str(key.preprocess_config_hash))
        & (df["feature_config_hash"].astype(str) == str(key.feature_config_hash))
    )


def find_manifest_entry(manifest_df: pd.DataFrame, key: FeatureStoreMatchKey) -> dict[str, Any] | None:
    if manifest_df.empty:
        return None
    mask = _entry_matches(manifest_df, key)
    if not bool(mask.any()):
        return None
    row = manifest_df.loc[mask].iloc[-1]
    return row.to_dict()


def upsert_manifest_entry(store_dir: Path, entry: dict[str, Any]) -> pd.DataFrame:
    manifest_df = load_manifest(store_dir)
    for col in MANIFEST_COLUMNS:
        if col not in entry:
            entry[col] = pd.NA
    row_df = pd.DataFrame([{k: entry.get(k, pd.NA) for k in MANIFEST_COLUMNS}])
    key = FeatureStoreMatchKey(
        recording=str(entry["recording"]),
        variant=str(entry["variant"]),
        source_path=str(entry["source_path"]),
        source_mtime=float(entry["source_mtime"]),
        preprocess_config_hash=str(entry["preprocess_config_hash"]),
        feature_config_hash=str(entry["feature_config_hash"]),
    )
    if not manifest_df.empty:
        manifest_df = manifest_df.loc[~_entry_matches(manifest_df, key)].copy()
        out_df = pd.concat([manifest_df, row_df], ignore_index=True)
    else:
        out_df = row_df.copy()
    save_manifest(store_dir, out_df)
    return out_df


def _safe_token(value: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value)
    return out.strip("._") or "item"


def write_feature_table(
    store_dir: Path,
    df: pd.DataFrame,
    *,
    recording: str,
    variant: str,
    kind: str,
) -> str:
    rec = _safe_token(recording)
    var = _safe_token(variant)
    knd = _safe_token(kind)
    rel = Path("tables") / var / f"{rec}__{knd}.parquet"
    abs_path = Path(store_dir) / rel
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(abs_path, index=False)
    return str(rel.as_posix())


def read_feature_table(store_dir: Path, rel_path: str) -> pd.DataFrame:
    abs_path = Path(store_dir) / rel_path
    return pd.read_parquet(abs_path)
