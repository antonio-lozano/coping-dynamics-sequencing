#!/usr/bin/env python
"""Create a clean Word Results section using the regenerated statistics."""
from __future__ import annotations

import argparse
import copy
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from prepare_results_differences import extract_results


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "report" / "results_section_revised_statistics.docx"
HIGHLIGHT_OUTPUT = ROOT / "report" / "results_section_revised_statistics_highlighted.docx"
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
XML = "{http://www.w3.org/XML/1998/namespace}"


PARAGRAPH_REPLACEMENTS: dict[str, list[tuple[str, str]]] = {
    "Syllables with mixed behaviors": [
        (
            "beta = 0.621, SE = 0.231, z = 2.694, p = 0.007, BH_FDR p = 0.025",
            "beta = 0.623, SE = 0.231, z = 2.696, p = 0.007, BH_FDR p = 0.025",
        ),
    ],
    "ELS may also alter the pattern": [
        (
            "beta = -0.013, SE = 0.007, z = -1.915, p = 0.056",
            "beta = -0.013, SE = 0.007, z = -1.886, p = 0.059",
        ),
        (
            "beta = 0.064, SE = 0.028, z = 2.250, p = 0.025",
            "beta = 0.064, SE = 0.028, z = 2.316, p = 0.021",
        ),
        (
            "beta = -0.094, SE = 0.033, z = -2.824, p = 0.005, BH_FDR p = 0.016",
            "beta = -0.094, SE = 0.038, z = -2.446, p = 0.014, BH_FDR p = 0.034",
        ),
        (
            "beta = 0.321, SE = 0.126, z = 2.540, p = 0.011, BH_FDR p = 0.026",
            "beta = 0.321, SE = 0.127, z = 2.519, p = 0.012, BH_FDR p = 0.034",
        ),
        (
            "beta = 0.164, SE = 0.058, z = 2.824, p = 0.005, BH_FDR p = 0.016",
            "beta = 0.164, SE = 0.061, z = 2.693, p = 0.007, BH_FDR p = 0.034",
        ),
        ("showed prolonged bout durations. showed prolonged bout durations.", "showed prolonged bout durations."),
    ],
    "To further explore individual differences": [
        (
            "beta = 0.409, SE = 0.118, z = 3.476, p < 0.001, BH_FDR = 0.002",
            "beta = 0.409, SE = 0.118, z = 3.475, p < 0.001, BH_FDR p = 0.002",
        ),
        (
            "beta = 1.865, SE = 0.249, z = 7.484, p < 0.001, BH_FDR p < 0.001",
            "beta = 1.865, SE = 0.241, z = 7.747, p < 0.001, BH_FDR p < 0.001",
        ),
        ("and were statistically comparable to controls. ). Likewise", "and were statistically comparable to controls. Likewise"),
    ],
    "Representative ethograms and barcodes": [
        (
            "beta = -0.022, SE = 0.008, z = -2.835, p = 0.005",
            "beta = -0.022, SE = 0.008, z = -2.818, p = 0.005",
        ),
        (
            "beta = 0.263, SE = 0.089, z = 2.969, p = 0.003; BH_FDR p = 0.021",
            "beta = 0.263, SE = 0.089, z = 2.972, p = 0.003; BH_FDR p = 0.021",
        ),
        (
            "beta = -8.965, SE = 2.658, z = -3.37; p < 0.001",
            "beta = -8.965, SE = 4.417, z = -2.030; p = 0.042",
        ),
        (
            "beta = 0.026, SE = 0.008, z = 2.798, p = 0.005",
            "beta = 0.026, SE = 0.009, z = 3.025, p = 0.003",
        ),
        (
            "beta = 0.010, SE = 0.003, z = 2.860, p = 0.004",
            "beta = 0.010, SE = 0.003, z = 2.799, p = 0.005",
        ),
        (
            "beta = -0.044, SE = 0.022, z = -1.99, p = 0.046",
            "beta = -0.044, SE = 0.021, z = -2.100, p = 0.036",
        ),
        (
            "beta = -0.029, SE = 0.011, z = -2.727, p = 0.006",
            "beta = -0.033, SE = 0.012, z = -2.755, p = 0.006",
        ),
    ],
}


TRANSITION_PARAGRAPH = (
    "We next focused on the transitions between behavioral states. The transition chordplots per group and "
    "the differences between groups are shown in Fig. 4R and S, respectively. While Lempel-Ziv complexity "
    "was unchanged (Fig. 4T), ELS animals showed a trend toward increased recurrence rate "
    "(beta = 0.013, SE = 0.007, z = 1.890, p = 0.059; Fig. 4U) as well as significantly increased determinism "
    "(beta = 0.016, SE = 0.007, z = 2.167, p = 0.030; Fig. 4V). Markov entropy, quantified from the transition "
    "probabilities between behavioral states using a first-order Markov chain, was unchanged "
    "(beta = -0.041, SE = 0.025, z = -1.618, p = 0.106; Fig. 4W). Together, these findings indicate that even "
    "though the overall variety of behavioral sequences and transition entropy were similar, ELS animals "
    "engaged in more repetitive and predictable behavioral patterns. Their behavior therefore appeared more "
    "rigid and less flexible during fear conditioning, suggesting a shift toward stereotyped patterns that may "
    "reflect reduced adaptability when faced with a secondary stressor or challenge."
)


def ascii_stat_symbols(text: str) -> str:
    """Use ASCII while editing, then restore the manuscript's statistical symbols."""
    return text.replace("beta", "\N{GREEK SMALL LETTER BETA}")


def difference_segments() -> list[str]:
    """Return the 17 regenerated statistic strings that differ from the manuscript."""
    segments = [
        ascii_stat_symbols(new)
        for replacements in PARAGRAPH_REPLACEMENTS.values()
        for _old, new in replacements
        if "beta =" in new
    ]
    segments.extend(
        [
            ascii_stat_symbols("beta = 0.013, SE = 0.007, z = 1.890, p = 0.059"),
            ascii_stat_symbols("beta = -0.041, SE = 0.025, z = -1.618, p = 0.106"),
        ]
    )
    if len(segments) != 17:
        raise AssertionError(f"Expected 17 highlighted differences, found {len(segments)}")
    return segments


def revise_paragraphs(paragraphs: list[str]) -> list[str]:
    revised = []
    applied = 0
    expected = sum(len(items) for items in PARAGRAPH_REPLACEMENTS.values())
    for original in paragraphs:
        text = (
            original.replace("\u00a0", " ")
            .replace("\N{MINUS SIGN}", "-")
            .replace("\N{GREEK SMALL LETTER BETA}", "beta")
        )
        if text.startswith("We next focused on the transitions"):
            text = TRANSITION_PARAGRAPH
            applied += 2  # Recurrence and Markov-entropy audit differences.
        for prefix, replacements in PARAGRAPH_REPLACEMENTS.items():
            if not text.startswith(prefix):
                continue
            for old, new in replacements:
                if old not in text:
                    raise AssertionError(f"Expected manuscript text not found: {old}")
                text = text.replace(old, new, 1)
                applied += 1
        revised.append(ascii_stat_symbols(text))
    if applied != expected + 2:
        raise AssertionError(f"Applied {applied} revisions; expected {expected + 2}")
    return revised


def paragraph_text(paragraph: ElementTree.Element) -> str:
    return "".join(node.text or "" for node in paragraph.iter(W + "t")).strip()


def set_paragraph_text(paragraph: ElementTree.Element, text: str) -> None:
    first_run = paragraph.find(W + "r")
    run_properties = None if first_run is None else first_run.find(W + "rPr")
    for child in list(paragraph):
        if child.tag != W + "pPr":
            paragraph.remove(child)
    run = ElementTree.SubElement(paragraph, W + "r")
    if run_properties is not None:
        run.append(copy.deepcopy(run_properties))
    text_node = ElementTree.SubElement(run, W + "t")
    text_node.set(XML + "space", "preserve")
    text_node.text = text


def set_paragraph_highlights(
    paragraph: ElementTree.Element,
    text: str,
    segments: list[str],
) -> int:
    first_run = paragraph.find(W + "r")
    base_properties = None if first_run is None else first_run.find(W + "rPr")
    for child in list(paragraph):
        if child.tag != W + "pPr":
            paragraph.remove(child)

    spans = []
    for segment in segments:
        start = text.find(segment)
        if start >= 0:
            spans.append((start, start + len(segment)))
    spans.sort()

    cursor = 0
    pieces: list[tuple[str, bool]] = []
    for start, end in spans:
        if start < cursor:
            raise AssertionError("Overlapping highlighted statistic strings")
        if start > cursor:
            pieces.append((text[cursor:start], False))
        pieces.append((text[start:end], True))
        cursor = end
    if cursor < len(text):
        pieces.append((text[cursor:], False))

    for piece, highlighted in pieces or [(text, False)]:
        run = ElementTree.SubElement(paragraph, W + "r")
        if base_properties is not None:
            properties = copy.deepcopy(base_properties)
            for prior in list(properties.findall(W + "highlight")):
                properties.remove(prior)
        else:
            properties = ElementTree.Element(W + "rPr")
        if highlighted:
            marker = ElementTree.SubElement(properties, W + "highlight")
            marker.set(W + "val", "yellow")
        if len(properties):
            run.append(properties)
        text_node = ElementTree.SubElement(run, W + "t")
        text_node.set(XML + "space", "preserve")
        text_node.text = piece
    return len(spans)


def create_results_docx(
    manuscript: Path,
    output: Path,
    revised: list[str],
    highlighted_segments: list[str] | None = None,
) -> int:
    with zipfile.ZipFile(manuscript) as source:
        document_root = ElementTree.fromstring(source.read("word/document.xml"))
        body = document_root.find(W + "body")
        if body is None:
            raise ValueError("The manuscript has no Word document body")

        result_paragraphs = []
        collecting = False
        for child in list(body):
            if child.tag != W + "p":
                continue
            text = paragraph_text(child)
            if text == "Results":
                collecting = True
            if collecting and text == "Discussion":
                break
            if collecting and text:
                result_paragraphs.append(copy.deepcopy(child))

        if len(result_paragraphs) != len(revised):
            raise AssertionError(
                f"Found {len(result_paragraphs)} source paragraphs for {len(revised)} revised paragraphs"
            )

        section_properties = body.find(W + "sectPr")
        for child in list(body):
            body.remove(child)
        highlighted = 0
        for paragraph, text in zip(result_paragraphs, revised):
            if highlighted_segments is None:
                set_paragraph_text(paragraph, text)
            else:
                highlighted += set_paragraph_highlights(paragraph, text, highlighted_segments)
            body.append(paragraph)
        if section_properties is not None:
            body.append(copy.deepcopy(section_properties))

        document_xml = ElementTree.tostring(document_root, encoding="utf-8", xml_declaration=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp.docx")
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as target:
            for item in source.infolist():
                content = document_xml if item.filename == "word/document.xml" else source.read(item.filename)
                target.writestr(item, content)
        temporary.replace(output)
    return highlighted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--highlight-output", type=Path, default=HIGHLIGHT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manuscript = args.manuscript.resolve()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    highlight_output = args.highlight_output if args.highlight_output.is_absolute() else ROOT / args.highlight_output
    original = extract_results(manuscript)
    revised = revise_paragraphs(original)
    create_results_docx(manuscript, output.resolve(), revised)
    highlighted = create_results_docx(
        manuscript,
        highlight_output.resolve(),
        revised,
        highlighted_segments=difference_segments(),
    )
    if highlighted != 17:
        raise AssertionError(f"Highlighted {highlighted} differences; expected 17")
    print(f"Saved: {output.resolve().relative_to(ROOT)}")
    print(f"Saved: {highlight_output.resolve().relative_to(ROOT)}")
    print(f"Paragraphs: {len(revised)}")
    print(f"Highlighted differences: {highlighted}")


if __name__ == "__main__":
    main()
