#!/usr/bin/env python
"""Extract the manuscript Results section and mark regenerated-statistic differences.

Revision-bound: the paragraph markers and before/after statistic strings below
are tied to the manuscript revision current in August 2026 (the one audited by
``statistics/manuscript_consistency_audit.csv``). They match on exact text, so
the next manuscript edit silently stops matching; re-derive the strings from
the audit CSV before trusting the output against a newer draft. Not part of
the reproducibility rebuild.
"""

from __future__ import annotations

import argparse
import csv
import html
import zipfile
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "statistics" / "manuscript_consistency_audit.csv"
MARKDOWN_OUT = ROOT / "report" / "results_section_with_differences.md"
HTML_OUT = ROOT / "report" / "results_section_with_differences.html"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

PARAGRAPH_CLAIMS = {
    "Syllables with mixed behaviors": [
        "Sniff ELS vs Control frequency",
    ],
    "ELS may also alter the pattern": [
        "Simpson diversity ELS vs Control",
        "Cumulative usage index ELS vs Control",
        "Freezing bout duration ELS vs Control",
        "Sniffing bout duration ELS vs Control",
        "Turn bout duration ELS vs Control",
    ],
    "We next focused on the transitions": [
        "Recurrence ELS vs Control",
        "Markov entropy ELS vs Control",
    ],
    "To further explore individual differences": [
        "Freeze resilient vs vulnerable ELS frequency",
        "Freeze time resilient vs vulnerable ELS",
    ],
    "Representative ethograms and barcodes": [
        "Simpson vulnerable ELS vs Control",
        "Freezing bout resilient vs vulnerable ELS",
        "Lempel-Ziv vulnerable ELS vs Control",
        "Recurrence vulnerable ELS vs Control",
        "Recurrence resilient vs vulnerable ELS",
        "Determinism vulnerable ELS vs Control",
        "Markov entropy vulnerable ELS vs Control",
    ],
}

STATUS_LABELS = {
    "numeric_mismatch_same_conclusion": "same significance conclusion",
    "minor_mismatch_same_interpretation": "near-threshold; same non-significant interpretation",
    "major_mismatch": "major mismatch",
}


def extract_results(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    paragraphs = []
    collecting = False
    for paragraph in root.iter(W + "p"):
        text = "".join(node.text or "" for node in paragraph.iter(W + "t")).strip()
        if text == "Results":
            collecting = True
        if collecting and text:
            if text == "Discussion":
                break
            paragraphs.append(text)
    if not paragraphs or paragraphs[0] != "Results":
        raise ValueError("Could not locate the Results section in the manuscript")
    return paragraphs


def load_differences() -> list[dict[str, str]]:
    with AUDIT.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row["status"] != "match"]


def changed_fields(row: dict[str, str]) -> list[tuple[str, str, str]]:
    fields = []
    for label, manuscript_key, repo_key in [
        ("beta", "manuscript_beta", "repo_beta"),
        ("SE", "manuscript_se", "repo_se"),
        ("z", "manuscript_z", "repo_z"),
        ("p", "manuscript_p", "repo_p"),
    ]:
        old = row.get(manuscript_key, "")
        new = row.get(repo_key, "")
        if old and new and old != new:
            fields.append((label, old, new))
    return fields


def markdown_difference(row: dict[str, str]) -> str:
    changes = "; ".join(
        f"{label}: ~~`{old}`~~ -> **`{new}`**" for label, old, new in changed_fields(row)
    )
    note = f" {row['note']}" if row.get("note") else ""
    return (
        f"> - **{row['section']}, {row['claim']}** ({STATUS_LABELS[row['status']]}): "
        f"{changes}.{note}"
    )


def html_difference(row: dict[str, str]) -> str:
    changes = "; ".join(
        f"{html.escape(label)}: <del>{html.escape(old)}</del> &rarr; <strong>{html.escape(new)}</strong>"
        for label, old, new in changed_fields(row)
    )
    note = f" {html.escape(row['note'])}" if row.get("note") else ""
    css_class = " major" if row["status"] == "major_mismatch" else ""
    return (
        f'<li class="difference{css_class}"><b>{html.escape(row["section"])}, '
        f"{html.escape(row['claim'])}</b> ({html.escape(STATUS_LABELS[row['status']])}): "
        f"{changes}.{note}</li>"
    )


def paragraph_rows(text: str, differences: list[dict[str, str]]) -> list[dict[str, str]]:
    claims = []
    for prefix, claim_names in PARAGRAPH_CLAIMS.items():
        if text.startswith(prefix):
            claims = claim_names
            break
    by_claim = {row["claim"]: row for row in differences}
    return [by_claim[claim] for claim in claims]


def is_subheading(text: str) -> bool:
    return text != "Results" and len(text) < 100 and not text.endswith(".")


def render_markdown(paragraphs: list[str], differences: list[dict[str, str]]) -> str:
    lines = [
        "# Results Section With Marked Statistical Differences",
        "",
        "Original manuscript values are struck through; regenerated report values are bold. "
        "Only audited differences are marked. Unmarked statistics matched at manuscript rounding.",
        "",
    ]
    mapped = set()
    for text in paragraphs:
        if text == "Results":
            lines.extend(["## Results", ""])
        elif is_subheading(text):
            lines.extend([f"### {text}", ""])
        else:
            lines.extend([text, ""])
        rows = paragraph_rows(text, differences)
        if rows:
            lines.extend(
                [
                    "> **AUDIT DIFFERENCES FOR THIS PARAGRAPH**",
                    ">",
                    *[markdown_difference(row) for row in rows],
                    "",
                ]
            )
            mapped.update(row["claim"] for row in rows)

    lines.extend(["## Complete Difference Register", ""])
    lines.append("| Figure | Result | Marked differences | Interpretation |")
    lines.append("|---|---|---|---|")
    for row in differences:
        changes = "; ".join(f"{label}: {old} -> {new}" for label, old, new in changed_fields(row))
        lines.append(
            f"| {row['section']} | {row['claim']} | {changes} | {STATUS_LABELS[row['status']]} |"
        )
    unmapped = {row["claim"] for row in differences}.difference(mapped)
    if unmapped:
        raise AssertionError(f"Unmapped audited differences: {sorted(unmapped)}")
    return "\n".join(lines) + "\n"


def render_html(paragraphs: list[str], differences: list[dict[str, str]]) -> str:
    body = [
        "<h1>Results Section With Marked Statistical Differences</h1>",
        "<p class=legend>Original manuscript values are struck through; regenerated report values are bold. "
        "Only audited differences are marked. Unmarked statistics matched at manuscript rounding.</p>",
    ]
    mapped = set()
    for text in paragraphs:
        if text == "Results":
            body.append("<h2>Results</h2>")
        elif is_subheading(text):
            body.append(f"<h3>{html.escape(text)}</h3>")
        else:
            body.append(f"<p>{html.escape(text)}</p>")
        rows = paragraph_rows(text, differences)
        if rows:
            body.append("<aside><b>AUDIT DIFFERENCES FOR THIS PARAGRAPH</b><ul>")
            body.extend(html_difference(row) for row in rows)
            body.append("</ul></aside>")
            mapped.update(row["claim"] for row in rows)

    body.append("<h2>Complete Difference Register</h2>")
    body.append(
        "<table><thead><tr><th>Figure</th><th>Result</th><th>Marked differences</th><th>Interpretation</th></tr></thead><tbody>"
    )
    for row in differences:
        changes = "; ".join(
            f"{html.escape(label)}: <del>{html.escape(old)}</del> &rarr; <strong>{html.escape(new)}</strong>"
            for label, old, new in changed_fields(row)
        )
        body.append(
            f"<tr><td>{html.escape(row['section'])}</td><td>{html.escape(row['claim'])}</td>"
            f"<td>{changes}</td><td>{html.escape(STATUS_LABELS[row['status']])}</td></tr>"
        )
    body.append("</tbody></table>")
    unmapped = {row["claim"] for row in differences}.difference(mapped)
    if unmapped:
        raise AssertionError(f"Unmapped audited differences: {sorted(unmapped)}")

    css = """
body { font-family: Arial, sans-serif; color: #2f2f2f; max-width: 980px; margin: 40px auto; line-height: 1.55; }
h1, h2, h3 { color: #24364b; }
h3 { margin-top: 2rem; }
.legend { padding: 12px 14px; background: #eef3f8; border-left: 4px solid #6c8ebf; }
aside { margin: 12px 0 24px; padding: 12px 16px; background: #fff8d9; border-left: 4px solid #d5a900; }
.difference { margin: 7px 0; }
.major { color: #9c1c1c; }
del { color: #a33; }
strong { color: #125b88; }
table { border-collapse: collapse; width: 100%; font-size: 0.92rem; }
th, td { border: 1px solid #c8c8c8; padding: 8px; text-align: left; vertical-align: top; }
th { background: #e8edf2; }
"""
    return (
        "<!doctype html><html><head><meta charset=utf-8><title>Results differences</title><style>"
        + css
        + "</style></head><body>"
        + "\n".join(body)
        + "</body></html>\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, default=MARKDOWN_OUT)
    parser.add_argument("--html-output", type=Path, default=HTML_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paragraphs = extract_results(args.manuscript)
    differences = load_differences()
    markdown = render_markdown(paragraphs, differences)
    html_text = render_html(paragraphs, differences)
    for path, content in [(args.markdown_output, markdown), (args.html_output, html_text)]:
        output = path if path.is_absolute() else ROOT / path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
        print(f"Saved: {output.relative_to(ROOT)}")
    print(f"Marked {len(differences)} audited differences")


if __name__ == "__main__":
    main()
