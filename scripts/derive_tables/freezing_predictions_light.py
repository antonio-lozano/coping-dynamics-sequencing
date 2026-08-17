"""Build compact raw-level freezing prediction companion files.

The per-animal prediction CSVs in data/raw/freezing_predictions are preserved as
the primary raw inputs. This script adds a single gzip-compressed long table and
a small file-level index for review and audit.

Run: python scripts/derive_tables/freezing_predictions_light.py
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[2]
RAW_DIR = REPO / "data" / "raw"
PRED_DIR = RAW_DIR / "freezing_predictions"
GROUPS_CSV = RAW_DIR / "animal_groups.csv"
OUT_LIGHT = RAW_DIR / "freezing_predictions_light.csv.gz"
OUT_INDEX = RAW_DIR / "freezing_predictions_index.csv"
FREEZING_COL = "Freezing_Jen_0-125_threshold"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(name)).strip("_")


def short_id(name: str) -> str:
    norm = normalize(name)
    match = re.search(r"Animal[_ ]?(\d+(?:[_-]\d+)?)", norm, flags=re.IGNORECASE)
    if match:
        return match.group(1).replace("_", ".").replace("-", ".")
    parts = norm.split("_")
    label = "_".join(parts[-2:]) if len(parts) >= 2 else norm
    return label.replace("_", ".").replace("-", ".")


def parse_base(path: Path) -> str:
    return path.name.replace("_freezing_predictions_only.csv", "")


def load_group_map() -> dict[str, str]:
    groups = pd.read_csv(GROUPS_CSV)
    out = {}
    for row in groups.itertuples(index=False):
        name = str(row.name)
        group = str(row.group)
        out[name] = group
        out[normalize(name)] = group
        out[short_id(name)] = group
    return out


def main() -> None:
    if not PRED_DIR.is_dir():
        raise FileNotFoundError(f"Missing raw freezing prediction directory: {PRED_DIR}")

    group_map = load_group_map()
    frames = []
    index_rows = []
    for path in sorted(PRED_DIR.glob("*_freezing_predictions_only.csv")):
        base = parse_base(path)
        animal_id = short_id(base)
        group = group_map.get(base) or group_map.get(normalize(base)) or group_map.get(animal_id) or "Unknown"
        df = pd.read_csv(path)
        if FREEZING_COL not in df.columns:
            raise ValueError(f"Missing {FREEZING_COL!r} in {path}")
        frame = df["Unnamed: 0"] if "Unnamed: 0" in df.columns else pd.Series(range(len(df)))
        freezing = pd.to_numeric(df[FREEZING_COL], errors="coerce").fillna(0).astype("int8")
        light = pd.DataFrame(
            {
                "animal": base,
                "animal_id": animal_id,
                "group": group,
                "frame": frame.astype("int32"),
                "freezing": freezing,
            }
        )
        frames.append(light)
        index_rows.append(
            {
                "file": path.name,
                "animal": base,
                "animal_id": animal_id,
                "group": group,
                "frames": int(len(light)),
                "freezing_frames": int(freezing.sum()),
                "freezing_fraction": float(freezing.mean()),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    if not frames:
        raise FileNotFoundError(f"No freezing prediction CSVs found in {PRED_DIR}")

    # mtime=0 keeps the gzip header out of the bytes, so rebuilding unchanged
    # data is a no-op instead of a new hash in MANIFEST.csv every run.
    pd.concat(frames, ignore_index=True).to_csv(
        OUT_LIGHT, index=False, compression={"method": "gzip", "mtime": 0}
    )
    pd.DataFrame(index_rows).to_csv(OUT_INDEX, index=False)
    print(f"Saved {OUT_LIGHT.relative_to(REPO)}")
    print(f"Saved {OUT_INDEX.relative_to(REPO)} ({len(index_rows)} files)")


if __name__ == "__main__":
    main()
