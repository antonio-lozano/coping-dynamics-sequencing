"""Read-only frame-keyed comparison of an external MoSeq CSV and bundled derivative.

This records differences, not accuracy, regeneration, or scientific acceptance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["name", "frame_index"]
KINEMATICS = ["centroid_x", "centroid_y", "heading", "angular_velocity", "velocity_px_s"]
REQUIRED = KEYS + ["group", "onset", "syllable"] + KINEMATICS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_frames(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={column: str for column in KEYS + ["group", "onset", "syllable"]},
        keep_default_na=False,
    )
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    if (frame[KEYS] == "").any().any():
        raise ValueError(f"Empty frame key in {path}")
    # CSV boolean capitalization is serialization, not a changed event label.
    frame["onset"] = frame["onset"].str.lower()
    if not frame["onset"].isin(["true", "false"]).all():
        raise ValueError(f"Non-boolean onset label in {path}")
    for column in KINEMATICS:
        frame[column] = pd.to_numeric(
            frame[column].mask(frame[column].eq(""), np.nan), errors="raise"
        )
    return frame


def key_summary(index: pd.MultiIndex) -> dict:
    return {
        "count": len(index),
        "examples": [dict(zip(KEYS, values)) for values in list(index[:10])],
    }


def audit(source_path: Path, bundled_path: Path) -> dict:
    paths = {"external": source_path.resolve(), "bundled": bundled_path.resolve()}
    before = {label: sha256(path) for label, path in paths.items()}
    external, bundled = (read_frames(paths[label]) for label in ("external", "bundled"))
    result = {
        "schema_version": 1,
        "claim": "Observed input differences only; not accuracy, regeneration, or scientific acceptance.",
        "key_policy": "Exact string name and frame_index; no recording ID normalization; onset capitalization ignored.",
        "inputs": {
            label: {
                "path": str(paths[label]),
                "sha256": before[label],
                "rows": len(frame),
                "recordings": frame["name"].nunique(),
            }
            for label, frame in (("external", external), ("bundled", bundled))
        },
    }
    ext_index = pd.MultiIndex.from_frame(external[KEYS])
    bun_index = pd.MultiIndex.from_frame(bundled[KEYS])
    result["duplicate_keys"] = {
        label: key_summary(index[index.duplicated(keep=False)].unique())
        for label, index in (("external", ext_index), ("bundled", bun_index))
    }
    result["external_keys_not_bundled"] = key_summary(ext_index.difference(bun_index))
    result["bundled_keys_not_external"] = key_summary(bun_index.difference(ext_index))
    result["external_recordings_not_bundled"] = sorted(set(external["name"]) - set(bundled["name"]))
    result["bundled_recordings_not_external"] = sorted(set(bundled["name"]) - set(external["name"]))
    if any(item["count"] for item in result["duplicate_keys"].values()):
        result["comparison_status"] = "BLOCKED_DUPLICATE_KEYS"
    else:
        external = external.set_index(KEYS)
        bundled = bundled.set_index(KEYS)
        common = ext_index.intersection(bun_index, sort=False)
        a, b = external.loc[common], bundled.loc[common]
        result["matched_frames"] = len(common)
        result["label_differences"] = {
            column: int((a[column] != b[column]).sum()) for column in ("group", "onset", "syllable")
        }
        changed = a["syllable"] != b["syllable"]
        pairs = pd.DataFrame(
            {
                "external": a.loc[changed, "syllable"].to_numpy(),
                "bundled": b.loc[changed, "syllable"].to_numpy(),
            }
        )
        result["syllable_remaps"] = [
            {"external": old, "bundled": new, "frames": int(count)}
            for (old, new), count in pairs.groupby(["external", "bundled"]).size().items()
        ]
        result["kinematics"] = {}
        for column in KINEMATICS:
            left, right = a[column].to_numpy(dtype=float), b[column].to_numpy(dtype=float)
            finite = np.isfinite(left) & np.isfinite(right)
            equal = (left == right) | (np.isnan(left) & np.isnan(right))
            result["kinematics"][column] = {
                "different_frames_exact": int((~equal).sum()),
                "nonfinite_mismatch_frames": int((~equal & ~finite).sum()),
                "finite_pairs": int(finite.sum()),
                "max_absolute_difference_finite": float(
                    np.max(np.abs(left[finite] - right[finite]))
                )
                if finite.any()
                else None,
            }
        result["comparison_status"] = "AUDIT_COMPLETE_NOT_SCIENTIFIC_ACCEPTANCE"
    after = {label: sha256(path) for label, path in paths.items()}
    result["input_hashes_unchanged"] = before == after
    if before != after:
        raise RuntimeError("Input changed during audit; no stable evidence can be reported")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external", required=True, type=Path)
    parser.add_argument("--bundled", required=True, type=Path)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New JSON file; existing files are never overwritten",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new evidence path")
    result = audit(args.external, args.bundled)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"{result['comparison_status']}: {args.output}")
    if result["comparison_status"] == "BLOCKED_DUPLICATE_KEYS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
