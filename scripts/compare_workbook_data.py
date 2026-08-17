"""Compare workbook data exactly, including formulas and Python value types."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import openpyxl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="Absolute tolerance for numeric cells (default: exact equality).",
    )
    return parser.parse_args()


def same_value(expected: object, actual: object, atol: float) -> bool:
    if atol and not isinstance(expected, bool) and not isinstance(actual, bool):
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return abs(float(expected) - float(actual)) <= atol
    if isinstance(expected, float) and isinstance(actual, float):
        if math.isnan(expected) and math.isnan(actual):
            return True
        return abs(expected - actual) <= atol
    return type(expected) is type(actual) and expected == actual


def populated_cells(worksheet) -> dict[str, tuple[str, object]]:
    return {
        cell.coordinate: (cell.data_type, cell.value)
        for cell in worksheet._cells.values()
        if cell.value is not None
    }


def main() -> int:
    args = parse_args()
    expected = openpyxl.load_workbook(args.expected, data_only=False)
    actual = openpyxl.load_workbook(args.actual, data_only=False)
    failures = 0

    if expected.sheetnames != actual.sheetnames:
        print(
            f"Sheet order mismatch:\n  expected={expected.sheetnames}\n  actual={actual.sheetnames}"
        )
        failures += 1

    for title in expected.sheetnames:
        if title not in actual.sheetnames:
            continue
        left = populated_cells(expected[title])
        right = populated_cells(actual[title])
        coordinates = sorted(set(left) | set(right))
        differences = []
        for coordinate in coordinates:
            expected_cell = left.get(coordinate)
            actual_cell = right.get(coordinate)
            if expected_cell is None or actual_cell is None:
                differences.append((coordinate, expected_cell, actual_cell))
                continue
            expected_type, expected_value = expected_cell
            actual_type, actual_value = actual_cell
            if expected_type != actual_type or not same_value(
                expected_value, actual_value, args.atol
            ):
                differences.append((coordinate, expected_cell, actual_cell))
        failures += len(differences)
        print(f"{title}: {len(differences)} differences across {len(coordinates)} populated cells")
        for coordinate, expected_cell, actual_cell in differences[:5]:
            print(f"  {coordinate}: expected={expected_cell!r} actual={actual_cell!r}")

    if failures:
        print(f"FAIL: {failures} total differences")
        return 1
    if args.atol:
        print(f"PASS: workbook data match within atol={args.atol:g} (100.00%)")
    else:
        print("PASS: exact workbook data match (100.00%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
