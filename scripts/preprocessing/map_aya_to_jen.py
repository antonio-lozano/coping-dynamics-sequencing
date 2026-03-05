#!/usr/bin/env python
"""Convert raw AYA DLC tracking outputs into JEN-style DLC bodypart schema."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Ensure repo root is importable when script is run directly.
import sys

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.config import DATA_DIR, DLC_DIR  # noqa: E402


JEN_BODY_ORDER = [
    "nose",
    "H1R",
    "H2R",
    "H1L",
    "H2L",
    "B1R",
    "B2R",
    "B3R",
    "B1L",
    "B2L",
    "B3L",
    "tail",
    "S2",
    "S1",
]

# Safe default mapping JEN <- AYA.
# Ambiguous duplicates are set to None and become NaN in output.
DEFAULT_JEN_TO_AYA_MAP: dict[str, str | None] = {
    "nose": "nose",
    "H1R": "right_ear",
    "H2R": None,
    "H1L": "left_ear",
    "H2L": None,
    "B1R": None,
    "B2R": "right_lateral",
    "B3R": None,
    "B1L": None,
    "B2L": "left_lateral",
    "B3L": None,
    "tail": "tail_base",
    "S2": None,
    "S1": "Centroid",
}

EXPECTED_COORDS = ["x", "y", "likelihood"]
SCORER_EXCLUDE = {"scorer", "unnamed: 0", ""}
BP_EXCLUDE = {"bodyparts", "unnamed: 0", ""}
COORD_EXCLUDE = {"coords", "unnamed: 0", ""}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map raw AYA DLC bodyparts into JEN-style schema and emit "
            "mapped CSV plus fps_manifest template."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing AYA .h5/.csv tracking files. Defaults to env/config fallbacks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for mapped files. Defaults to data/to_predict/<dataset_id>.",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="Dataset name used for default output dir under data/to_predict.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*_filtered.h5",
        help="Input glob pattern (default: *_filtered.h5).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of files to convert.",
    )
    parser.add_argument(
        "--mapping-overrides",
        type=str,
        default=None,
        help="Override mapping with comma-separated jen=aya pairs (aya can be 'nan').",
    )
    parser.add_argument(
        "--target-scorer",
        type=str,
        default=None,
        help="Output scorer name. If omitted, infer from JEN CSV header when possible.",
    )
    parser.add_argument(
        "--write-h5",
        action="store_true",
        help="Also write mapped .h5 alongside CSV.",
    )
    parser.add_argument(
        "--fps-default",
        type=float,
        default=None,
        help="Optional default FPS value used to prefill fps_manifest.csv.",
    )
    parser.add_argument(
        "--write-fps-manifest",
        dest="write_fps_manifest",
        action="store_true",
        help="Write fps_manifest.csv template (default: on).",
    )
    parser.add_argument(
        "--no-write-fps-manifest",
        dest="write_fps_manifest",
        action="store_false",
        help="Disable fps_manifest.csv writing.",
    )
    parser.set_defaults(write_fps_manifest=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected files and mapping summary without writing outputs.",
    )
    return parser.parse_args()


def _normalize_tokens(values: list[Any], excluded: set[str]) -> list[str]:
    out = []
    for value in values:
        text = str(value).strip()
        if text.lower() in excluded:
            continue
        out.append(text)
    return out


def _resolve_default_input_dir() -> Path | None:
    candidates: list[Path] = []
    env_candidates = [
        os.getenv("COPING_DYNAMICS_AYA_RAW_DIR"),
        os.getenv("AYA_BEHAVIOR_ENCODED_DIR"),
    ]
    for raw in env_candidates:
        if raw:
            candidates.append(Path(raw))
    candidates.extend(
        [
            Path(r"D:\NINDATA\Aya_behavior\encoded-20260131T123011Z-3-001\encoded"),
            DATA_DIR / "aya_behavior" / "encoded",
            DATA_DIR / "aya_raw",
            repo_root / "data" / "aya_raw",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()

    dataset_id = args.dataset_id or "aya_to_jen_mapped"
    return (repo_root / "data" / "to_predict" / dataset_id).resolve()


def parse_mapping_overrides(raw_value: str | None) -> dict[str, str | None]:
    if not raw_value:
        return {}
    overrides: dict[str, str | None] = {}
    for item in raw_value.split(","):
        pair = item.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise SystemExit(f"Invalid mapping override '{pair}'. Use jen=aya.")
        jen_bp, aya_bp = pair.split("=", 1)
        jen_bp = jen_bp.strip()
        aya_bp = aya_bp.strip()
        if jen_bp not in JEN_BODY_ORDER:
            raise SystemExit(f"Invalid JEN bodypart '{jen_bp}' in override '{pair}'.")
        if aya_bp.lower() in {"nan", "none", "null", "<nan>", ""}:
            overrides[jen_bp] = None
        else:
            overrides[jen_bp] = aya_bp
    return overrides


def infer_target_scorer() -> str | None:
    if not DLC_DIR.exists():
        return None
    csv_files = sorted(DLC_DIR.glob("*.csv"))
    if not csv_files:
        return None
    try:
        header_df = pd.read_csv(csv_files[0], header=[0, 1, 2], nrows=0)
    except Exception:
        return None

    if not isinstance(header_df.columns, pd.MultiIndex):
        return None
    scorers = _normalize_tokens(
        header_df.columns.get_level_values(0).unique().tolist(),
        SCORER_EXCLUDE,
    )
    return scorers[0] if scorers else None


def list_aya_files(input_dir: Path, pattern: str, limit: int | None) -> list[Path]:
    if not input_dir.exists():
        raise SystemExit(f"AYA input directory not found: {input_dir}")

    files = sorted(input_dir.glob(pattern))
    if not files:
        files = sorted(input_dir.glob("*.h5"))
    if not files:
        files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No input files matched in {input_dir}.")

    if limit is not None:
        if limit < 1:
            raise SystemExit("--limit must be >= 1")
        files = files[:limit]
    return files


def load_tracking_df(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".h5":
        try:
            return pd.read_hdf(path)
        except ImportError as exc:
            raise SystemExit(
                "Reading .h5 requires PyTables. Install it with: pip install tables"
            ) from exc
    if suffix == ".csv":
        return pd.read_csv(path, header=[0, 1, 2], index_col=0)
    raise SystemExit(f"Unsupported input file type: {path.suffix}")


def first_scorer(df: pd.DataFrame) -> str:
    if not isinstance(df.columns, pd.MultiIndex):
        raise SystemExit("Expected DLC MultiIndex columns (scorer/bodyparts/coords).")
    scorers = _normalize_tokens(
        df.columns.get_level_values(0).unique().tolist(),
        SCORER_EXCLUDE,
    )
    if not scorers:
        raise SystemExit("No scorer found in DLC columns.")
    return scorers[0]


def build_renamed_df(
    source_df: pd.DataFrame,
    source_scorer: str,
    target_scorer: str,
    jen_to_aya_map: dict[str, str | None],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not isinstance(source_df.columns, pd.MultiIndex):
        raise SystemExit("Input dataframe must have DLC MultiIndex columns.")

    available_bodyparts = _normalize_tokens(
        source_df.columns.get_level_values(1).unique().tolist(),
        BP_EXCLUDE,
    )
    available_bodyparts_set = set(available_bodyparts)
    available_coords = _normalize_tokens(
        source_df.columns.get_level_values(2).unique().tolist(),
        COORD_EXCLUDE,
    )
    coords = [coord for coord in EXPECTED_COORDS if coord in set(available_coords)]
    if not coords:
        raise SystemExit("No expected coords found in source file (x/y/likelihood).")

    out_columns: dict[tuple[str, str, str], pd.Series] = {}
    missing_targets: list[str] = []
    filled_targets: list[str] = []

    for jen_bp in JEN_BODY_ORDER:
        aya_bp = jen_to_aya_map.get(jen_bp)
        has_bp = aya_bp is not None and aya_bp in available_bodyparts_set
        if has_bp:
            filled_targets.append(jen_bp)
        else:
            missing_targets.append(jen_bp)

        for coord in coords:
            column_key = (target_scorer, jen_bp, coord)
            if has_bp and (source_scorer, aya_bp, coord) in source_df.columns:
                out_columns[column_key] = source_df[(source_scorer, aya_bp, coord)].copy()
            else:
                out_columns[column_key] = pd.Series(np.nan, index=source_df.index, dtype=float)

    output_df = pd.DataFrame(out_columns, index=source_df.index)
    output_df.columns = pd.MultiIndex.from_tuples(
        output_df.columns,
        names=["scorer", "bodyparts", "coords"],
    )
    details = {
        "available_bodyparts": sorted(available_bodyparts),
        "available_coords": coords,
        "missing_target_bodyparts": sorted(missing_targets),
        "filled_target_bodyparts": sorted(filled_targets),
    }
    return output_df, details


def build_fps_manifest_rows(
    outputs: list[dict[str, Any]],
    fps_default: float | None,
) -> pd.DataFrame:
    rows = []
    for item in outputs:
        rows.append(
            {
                "recording": item["recording"],
                "fps": fps_default if fps_default is not None else np.nan,
                "source_file": item["source_file"],
            }
        )
    return pd.DataFrame(rows).sort_values("recording").reset_index(drop=True)


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.resolve() if args.input_dir is not None else None
    if input_dir is None:
        input_dir = _resolve_default_input_dir()
        if input_dir is None:
            raise SystemExit(
                "Could not infer AYA input directory. Pass --input-dir or set COPING_DYNAMICS_AYA_RAW_DIR."
            )
    output_dir = _resolve_output_dir(args)

    jen_to_aya_map = dict(DEFAULT_JEN_TO_AYA_MAP)
    overrides = parse_mapping_overrides(args.mapping_overrides)
    jen_to_aya_map.update(overrides)

    source_files = list_aya_files(input_dir, args.pattern, args.limit)
    target_scorer = args.target_scorer or infer_target_scorer()

    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Files selected: {len(source_files)}")
    for file_path in source_files:
        print(f"  - {file_path.name}")

    print("\nBodypart mapping (JEN <- AYA):")
    for jen_bp in JEN_BODY_ORDER:
        mapped = jen_to_aya_map.get(jen_bp)
        print(f"  - {jen_bp} <- {mapped if mapped is not None else 'NaN'}")

    if args.dry_run:
        print("\nDry run only. No files written.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    converted_outputs: list[dict[str, Any]] = []

    for source_path in source_files:
        source_df = load_tracking_df(source_path)
        source_scorer = first_scorer(source_df)
        scorer_out = target_scorer or source_scorer

        converted_df, details = build_renamed_df(
            source_df=source_df,
            source_scorer=source_scorer,
            target_scorer=scorer_out,
            jen_to_aya_map=jen_to_aya_map,
        )

        out_csv = output_dir / f"{source_path.stem}_matched_to_jen.csv"
        converted_df.to_csv(out_csv)
        print(f"Wrote CSV: {out_csv}")

        out_h5 = None
        if args.write_h5:
            try:
                out_h5 = output_dir / f"{source_path.stem}_matched_to_jen.h5"
                converted_df.to_hdf(out_h5, key="df_with_missing", mode="w")
                print(f"Wrote H5: {out_h5}")
            except ImportError as exc:
                raise SystemExit(
                    "Writing .h5 requires PyTables. Install it with: pip install tables"
                ) from exc

        converted_outputs.append(
            {
                "source_file": source_path.name,
                "source_scorer": source_scorer,
                "output_scorer": scorer_out,
                "recording": out_csv.stem,
                "output_csv": out_csv.name,
                "output_h5": out_h5.name if out_h5 else None,
                **details,
            }
        )

    if args.write_fps_manifest:
        fps_manifest = build_fps_manifest_rows(converted_outputs, args.fps_default)
        fps_path = output_dir / "fps_manifest.csv"
        fps_manifest.to_csv(fps_path, index=False)
        print(f"Wrote FPS manifest template: {fps_path}")

    report = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "pattern": args.pattern,
        "limit": args.limit,
        "source_file_count": len(source_files),
        "converted_file_count": len(converted_outputs),
        "target_scorer": target_scorer,
        "mapping": {k: v for k, v in jen_to_aya_map.items()},
        "mapping_overrides": overrides,
        "files": converted_outputs,
    }
    report_path = output_dir / "mapping_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote mapping report: {report_path}")


if __name__ == "__main__":
    main()
