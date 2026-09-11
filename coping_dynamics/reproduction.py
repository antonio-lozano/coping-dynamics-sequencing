"""Isolated, frozen-reference reproduction; never updates the reference manifest.

Run from a source checkout with ``python -m coping_dynamics.reproduction --output DIR``.
DIR must not exist and must be outside the checkout. A successful execution is not
necessarily scientific agreement: all differences are retained in evidence.json.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Identity columns must compare as strings (150.10 and 150.1 are not one animal).
IDENTIFIERS = {
    "animal",
    "animal_id",
    "name",
    "id",
    "litter",
    "group",
    "condition",
    "experiment",
    "cohort",
    "parameter",
    "metric",
    "behavior",
    "cluster",
    "syllable",
}
ARTIFACT_DIRS = (
    "data/processed",
    "figure_source_data",
    "statistics",
    "figures",
    "report",
    "classifier",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    if path.is_symlink():
        # Bind the alias itself; target files have their own inventory hashes.
        h.update(b"symlink\0" + os.fsencode(os.readlink(path)))
        return h.hexdigest()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def changed_sources(source: Path, source_hashes: dict[str, str]) -> list[str]:
    """Identify changed or missing inventoried files and aliases."""
    return [
        n
        for n, h in source_hashes.items()
        if not ((source / n).is_file() or (source / n).is_symlink()) or digest(source / n) != h
    ]


def compare_csv(reference: Path, candidate: Path, *, rtol=1e-7, atol=1e-10) -> dict:
    """Compare row/column order and text exactly, permitting explicit numeric roundoff.

    Missing values must remain missing. Identifiers never receive numeric tolerance.
    Tolerances are numerical, not a claim that a changed analysis is scientifically valid.
    """
    with (
        reference.open(newline="", encoding="utf-8-sig") as a,
        candidate.open(newline="", encoding="utf-8-sig") as b,
    ):
        left, right = list(csv.reader(a)), list(csv.reader(b))
    if not left or not right or left[0] != right[0] or len(left) != len(right):
        return {"status": "different", "reason": "header or row count differs"}
    differences = []
    for row_number, (row_a, row_b) in enumerate(zip(left[1:], right[1:]), 2):
        if len(row_a) != len(left[0]) or len(row_b) != len(left[0]):
            return {"status": "different", "reason": f"column count differs at row {row_number}"}
        for column, va, vb in zip(left[0], row_a, row_b):
            if va == vb:
                continue
            equal = False
            if column.lower() not in IDENTIFIERS:
                try:
                    fa, fb = float(va), float(vb)
                    equal = math.isclose(fa, fb, rel_tol=rtol, abs_tol=atol)
                except ValueError:
                    pass
            if not equal:
                differences.append(
                    {"row": row_number, "column": column, "reference": va, "candidate": vb}
                )
                if len(differences) >= 20:
                    return {"status": "different", "examples": differences, "truncated": True}
    return (
        {"status": "different", "examples": differences}
        if differences
        else {"status": "numerically_equal"}
    )


def copy_checkout(source: Path, destination: Path, names: list[str]) -> None:
    """Make a byte-copy, never hardlinks, in a new directory outside the source."""
    source, destination = source.resolve(), destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Output must be outside the source checkout")
    if destination.exists():
        raise FileExistsError(destination)
    # Validate the whole path set before writing anything.
    for name in names:
        rel = Path(name)
        path = source / rel
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Unsafe checkout path: {name}")
        if path.is_symlink():
            link = Path(os.readlink(path))
            resolved = path.resolve()
            if link.is_absolute() or source not in resolved.parents or not resolved.exists():
                raise ValueError(f"Unsafe checkout alias: {name}")
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
    destination.mkdir(parents=True)
    for name in names:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        original = source / name
        if original.is_symlink():
            target.symlink_to(os.readlink(original), target_is_directory=original.is_dir())
        else:
            shutil.copy2(original, target)


def compare_raw_inputs(
    source: Path, candidate: Path, names: list[str]
) -> tuple[list[str], list[str]]:
    """Require primary inputs unchanged; name known derived-companion serialization.

    The compact freezing gzip is generated by the pipeline even though inherited
    layout places it in data/raw. Only LF/CRLF differences in its decompressed
    payload are permitted; any changed value remains an input difference.
    """
    changed, serialization = [], []
    for name in names:
        if not name.startswith("data/raw/"):
            continue
        a, b = source / name, candidate / name
        if b.is_file() and digest(a) == digest(b):
            continue
        if name == "data/raw/freezing_predictions_light.csv.gz" and b.is_file():
            try:
                left = gzip.decompress(a.read_bytes()).replace(b"\r\n", b"\n")
                right = gzip.decompress(b.read_bytes()).replace(b"\r\n", b"\n")
                if left == right:
                    serialization.append(name)
                    continue
            except (OSError, EOFError):
                pass
        changed.append(name)
    return changed, serialization


def summarize_run(exit_codes: list[int], comparisons: list[dict], changed_inputs: list[str]) -> str:
    """A zero process exit cannot turn changed science or missing output into PASS."""
    if (
        any(exit_codes)
        or changed_inputs
        or any(x["status"] in {"different", "missing"} for x in comparisons)
    ):
        return "FAILED"
    if any(x["status"] == "byte_different" for x in comparisons):
        return "REVIEW_REQUIRED"
    return "PASS"


def compare_artifacts(source: Path, candidate: Path, names: list[str]) -> list[dict]:
    results = []
    for name in names:
        if not any(name.startswith(prefix + "/") for prefix in ARTIFACT_DIRS):
            continue
        a, b = source / name, candidate / name
        kind = (
            "csv"
            if a.suffix == ".csv"
            else "figure"
            if name.startswith("figures/")
            else "binary_or_text"
        )
        row = {"path": name, "kind": kind, "reference_sha256": digest(a)}
        if not b.is_file():
            row["status"] = "missing"
        else:
            row["candidate_sha256"] = digest(b)
            if row["reference_sha256"] == row["candidate_sha256"]:
                row["status"] = "byte_equal"
            elif kind == "csv":
                row.update(compare_csv(a, b))
            else:
                row["status"] = "byte_different"
        results.append(row)
    return results


def comparison_summary(comparisons: list[dict], observations: list[dict]) -> dict:
    """Separate artifact agreement from observed writes, never infer a model refit."""
    writes = {row["path"]: row["write_observed"] for row in observations}
    counts = {key: {} for key in ("write_observed", "no_write_observed", "unknown")}
    for row in comparisons:
        written = writes.get(row["path"])
        key = (
            "write_observed"
            if written is True
            else "no_write_observed"
            if written is False
            else "unknown"
        )
        bucket = counts[key]
        bucket[row["status"]] = bucket.get(row["status"], 0) + 1
    return counts


def observe_outputs(checkout: Path, before: dict[str, int]) -> list[dict]:
    """Record write observation separately from numerical/byte agreement.

    An mtime change is evidence of a write, not proof of a model refit. Explicit
    --recompute commands and their logs supply the latter execution evidence.
    """
    return [
        {
            "path": name,
            "status": "present" if (checkout / name).is_file() else "missing",
            "write_observed": (checkout / name).is_file()
            and (checkout / name).stat().st_mtime_ns != stamp,
        }
        for name, stamp in before.items()
    ]


def failed_models(checkout: Path) -> list[dict]:
    """Report model failure rows even when their generator exits successfully."""
    failures = []
    for path in sorted((checkout / "statistics").glob("*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for number, row in enumerate(csv.DictReader(handle), 2):
                if (
                    row.get("parameter") == "model_failed"
                    or row.get("converged", "").lower() == "false"
                ):
                    failures.append(
                        {"path": path.relative_to(checkout).as_posix(), "line": number, "row": row}
                    )
    return failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New evidence directory outside the source checkout",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Partial tables/reports run; not full reproduction",
    )
    args = parser.parse_args(argv)
    source = Path(__file__).resolve().parents[1]
    if not (source / "scripts/run_all.py").is_file():
        parser.error("Reproduction requires the source checkout, not only an installed wheel")
    output = args.output.resolve()
    if output == source or source in output.parents or output.exists():
        parser.error("Output must be a new directory outside the source checkout")
    try:
        names = (
            subprocess.check_output(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=source
            )
            .decode()
            .split("\0")
        )
        names = sorted(set(n for n in names if n))
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip()
    except subprocess.CalledProcessError:
        parser.error("Run from a Git source checkout")
    # Freeze hashes of actual working files: revision alone does not identify uncommitted edits.
    source_hashes = {n: digest(source / n) for n in names}
    copy_checkout(source, output / "checkout", names)
    checkout = output / "checkout"
    output_stamps = {
        n: (checkout / n).stat().st_mtime_ns
        for n in names
        if any(n.startswith(prefix + "/") for prefix in ARTIFACT_DIRS)
    }
    evidence = {
        "revision": revision,
        "source_hashes": source_hashes,
        "python": sys.version,
        "platform": platform.platform(),
        "scope": "tables_reports" if args.skip_figures else "data_derived_pipeline",
        "rtol": 1e-7,
        "atol": 1e-10,
        "steps": [],
        "limitations": [
            "Archived schematics and classifier results are not re-estimated.",
            "Agreement with bundled artifacts is not agreement with the original manuscript.",
            "CSV row and column order must agree; other changed artifacts require review.",
        ],
    }
    env = {k: v for k, v in os.environ.items() if not k.startswith("COPING_DYNAMICS_")}
    env.update(
        COPING_DYNAMICS_ROOT=str(checkout),
        PYTHONPATH=str(checkout),
        MPLBACKEND="Agg",
        BATCH_MODE="1",
        SOURCE_DATE_EPOCH="0",
        PYTHONUNBUFFERED="1",
    )
    # Load only the task declaration, not a mutable manifest or precomputed verdict.
    import importlib.util

    spec = importlib.util.spec_from_file_location("pipeline_tasks", checkout / "scripts/run_all.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    skipped = {"scripts/update_manifest.py", "scripts/check_reproducibility.py"}
    if args.skip_figures:
        skipped.add("scripts/run_all_figures.py")
    tasks = [("verify frozen source artifacts", ["scripts/check_reproducibility.py"])]
    tasks += [
        (label, [*cmd, "--recompute"] if cmd[0] == "scripts/run_all_figures.py" else cmd)
        for label, cmd in module.TASKS
        if cmd[0] not in skipped
    ]
    for index, (label, command) in enumerate(tasks):
        log = output / f"{index:02d}.log"
        start = time.monotonic()
        print(f"[{index + 1}/{len(tasks)}] {label}", flush=True)
        with log.open("w") as handle:
            result = subprocess.run(
                [sys.executable, *command],
                cwd=checkout,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        evidence["steps"].append(
            {
                "label": label,
                "command": [sys.executable, *command],
                "exit_code": result.returncode,
                "seconds": time.monotonic() - start,
                "log": log.name,
            }
        )
        (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
        if result.returncode:
            break
    evidence["comparisons"] = compare_artifacts(source, checkout, names)
    evidence["output_write_observations"] = observe_outputs(checkout, output_stamps)
    evidence["comparison_summary"] = comparison_summary(
        evidence["comparisons"], evidence["output_write_observations"]
    )
    # Require a write for the primary generated products. Other preserved
    # artifacts are explicitly listed, never credited as regenerated models.
    required = {
        f"figures/{stem}{suffix}"
        for stem in [
            "figure2",
            "figure3",
            "figure4",
            "figure5",
            "figure6",
            "figure7",
            "supplementary_figure1",
            "supplementary_figure3",
            "supplementary_figure4",
        ]
        for suffix in [".pdf", ".svg", ".png"]
    }
    required |= {"report/raw_data.xlsx", "report/statistical_report.xlsx"}
    if args.skip_figures:
        required = {n for n in required if not n.startswith("figures/")}
    evidence["missing_regeneration"] = sorted(
        required
        - {row["path"] for row in evidence["output_write_observations"] if row["write_observed"]}
    )

    evidence["source_changed"] = changed_sources(source, source_hashes)
    evidence["raw_input_changed"], evidence["derived_companion_serialization"] = compare_raw_inputs(
        source, checkout, names
    )
    evidence["status"] = summarize_run(
        [s["exit_code"] for s in evidence["steps"]],
        evidence["comparisons"],
        evidence["source_changed"] + evidence["raw_input_changed"],
    )
    evidence["failed_models"] = failed_models(checkout)
    if evidence["failed_models"] or evidence["missing_regeneration"]:
        evidence["status"] = "FAILED"
    if args.skip_figures and evidence["status"] == "PASS":
        evidence["status"] = "PARTIAL"
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"{evidence['status']}: {output / 'evidence.json'}")
    return 0 if evidence["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
