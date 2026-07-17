# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jeniffer Sanguino Gomez and Antonio Lozano
"""Import manuscript Figure 7 from the assembled source PDF.

Figure 7 is maintained as an assembled PDF rather than regenerated from tabular
source data in this repository. The source PDF is tracked in ``dataset/source/``
and this script writes only the canonical manuscript outputs in ``figures/``.

Override the source PDF with:
FIGURE7_SOURCE_PDF="path/to/Figure7.pdf" python scripts/generate_figures/figure_7_import_pdf.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "figures"

BUNDLED_SOURCE = REPO_ROOT / "dataset" / "source" / "figure7.pdf"
PANEL_A_UNASSIGNED_RECT = (19.5, 123.0, 46.5, 131.5)
PANEL_A_UNASSIGNED_ORIGIN = (20.582500457763672, 128.6944580078125)
PANEL_A_UNASSIGNED_SIZE = 5.0
PANEL_A_UNASSIGNED_COLOR = (77 / 255, 77 / 255, 78 / 255)
PANEL_C_LABEL_RECT = (18.0, 173.5, 28.0, 188.5)
PANEL_C_LABEL_ORIGIN = (20.138200759887695, 183.87310791015625)
PANEL_C_LABEL_SIZE = 8.568400382995605
PANEL_LABEL_COLOR = (77 / 255, 77 / 255, 77 / 255)
PANEL_C_BODY_RECT = (0.0, 175.0, 210.0, 342.0)
PANEL_C_BODY_DEST = (-14.0, 174.0, 209.0, 338.0)
CONFUSION_LABELS = ["Freeze", "Sniff", "Groom", "Turn", "Locomotion", "Climb", "Jump", "Unassigned"]
CONFUSION_MATRIX = [
    [0.8, 0.004, 0.0, 0.021, 0.044, 0.041, 0.044, 0.041],
    [0.002, 0.86, 0.0, 0.0, 0.002, 0.022, 0.069, 0.044],
    [0.069, 0.007, 0.45, 0.069, 0.038, 0.0, 0.031, 0.34],
    [0.038, 0.0, 0.0, 0.71, 0.16, 0.034, 0.026, 0.038],
    [0.078, 0.001, 0.0, 0.06, 0.61, 0.043, 0.088, 0.11],
    [0.036, 0.037, 0.0, 0.001, 0.045, 0.68, 0.13, 0.068],
    [0.048, 0.049, 0.0, 0.008, 0.1, 0.16, 0.47, 0.16],
    [0.076, 0.085, 0.0, 0.025, 0.19, 0.096, 0.23, 0.3],
]


def resolve_source_pdf() -> Path:
    """Return the first available Figure 7 source PDF."""
    env_source = os.environ.get("FIGURE7_SOURCE_PDF")
    candidates = [Path(env_source)] if env_source else []
    candidates.append(BUNDLED_SOURCE)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n  - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find Figure 7 source PDF. Set FIGURE7_SOURCE_PDF or place it at one of:\n"
        f"  - {searched}"
    )


def validate_pdf(path: Path) -> None:
    with path.open("rb") as handle:
        header = handle.read(4)
    if header != b"%PDF":
        raise ValueError(f"Figure 7 source is not a PDF: {path}")


def copy_pdf(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    print(f"Saved: {destination}")


def save_pdf_atomic(doc, destination: Path) -> None:
    """Save a PyMuPDF document through a temporary file for Windows-friendly replace."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f"{destination.stem}_",
        suffix=".pdf",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temp_destination = Path(handle.name)

    try:
        doc.save(temp_destination, garbage=4, deflate=True)
        shutil.move(str(temp_destination), destination)
    finally:
        if temp_destination.exists():
            temp_destination.unlink()


def render_confusion_panel_png(destination: Path) -> None:
    """Draw panel C cleanly instead of scaling the assembled PDF fragment."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    data = np.array(CONFUSION_MATRIX, dtype=float)
    cmap = LinearSegmentedColormap.from_list(
        "figure7_confusion",
        ["#ffffff", "#ead6e2", "#b7588f"],
    )
    text_color = "#bb5b8f"
    axis_color = "#4d4d4d"

    fig = plt.figure(figsize=(3.45, 2.40), dpi=500, facecolor="white")
    ax = fig.add_axes([0.235, 0.245, 0.595, 0.660])
    image = ax.imshow(data, cmap=cmap, vmin=0.0, vmax=0.86, interpolation="nearest")

    ax.set_xticks(np.arange(len(CONFUSION_LABELS)))
    ax.set_yticks(np.arange(len(CONFUSION_LABELS)))
    ax.set_xticklabels(CONFUSION_LABELS, fontsize=3.35, color=axis_color)
    ax.set_yticklabels(CONFUSION_LABELS, fontsize=4.5, color=axis_color)
    ax.tick_params(axis="both", length=1.5, width=0.35, colors=axis_color, pad=1.8)
    ax.set_xlabel("Predicted Label", fontsize=6.0, color=axis_color, labelpad=4.5)
    ax.set_ylabel("True Label", fontsize=6.0, color=axis_color, labelpad=6.0)
    ax.set_xlim(-0.5, len(CONFUSION_LABELS) - 0.5)
    ax.set_ylim(len(CONFUSION_LABELS) - 0.5, -0.5)

    for spine in ax.spines.values():
        spine.set_color(axis_color)
        spine.set_linewidth(0.45)

    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            label = f"{value:.3g}"
            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                fontsize=4.8,
                color="white" if value >= 0.45 else text_color,
            )

    cax = fig.add_axes([0.852, 0.245, 0.032, 0.660])
    colorbar = fig.colorbar(image, cax=cax, ticks=np.arange(0.0, 0.81, 0.1))
    colorbar.ax.tick_params(labelsize=5.0, length=1.6, width=0.35, colors=axis_color, pad=2.0)
    colorbar.outline.set_linewidth(0.45)
    colorbar.outline.set_edgecolor(axis_color)

    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=500, facecolor="white")
    plt.close(fig)


def write_normalized_pdf(source: Path, destination: Path) -> None:
    """Copy Figure 7 while normalizing panel A and redrawing panel C."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required to normalize Figure 7 PDF output. "
            "Install dependencies from requirements.txt or environment.yml."
        ) from exc

    doc = fitz.open(source)
    page = doc[0]
    myriad_font = None
    for font in page.get_fonts(full=True):
        xref, _, _, name, *_ = font
        if "MyriadPro-Regular" in name:
            myriad_font = doc.extract_font(xref)[3]
            break
    if myriad_font is None:
        raise ValueError("Could not find embedded MyriadPro-Regular font in Figure 7 PDF.")

    page.add_redact_annot(fitz.Rect(PANEL_A_UNASSIGNED_RECT), fill=(1, 1, 1))
    page.add_redact_annot(fitz.Rect(PANEL_C_LABEL_RECT), fill=(1, 1, 1))
    page.add_redact_annot(fitz.Rect(PANEL_C_BODY_RECT), fill=(1, 1, 1))
    page.apply_redactions()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        panel_c_png = Path(handle.name)
    try:
        render_confusion_panel_png(panel_c_png)
        page.insert_image(fitz.Rect(PANEL_C_BODY_DEST), filename=panel_c_png, overlay=True)
    finally:
        if panel_c_png.exists():
            panel_c_png.unlink()

    page.insert_text(
        PANEL_C_LABEL_ORIGIN,
        "C",
        fontsize=PANEL_C_LABEL_SIZE,
        fontname="hebo",
        color=PANEL_LABEL_COLOR,
        overlay=True,
    )

    page.insert_font(fontname="myriad_regular", fontbuffer=myriad_font)
    page.insert_text(
        PANEL_A_UNASSIGNED_ORIGIN,
        "Unassigned",
        fontsize=PANEL_A_UNASSIGNED_SIZE,
        fontname="myriad_regular",
        color=PANEL_A_UNASSIGNED_COLOR,
        overlay=True,
    )

    save_pdf_atomic(doc, destination)
    doc.close()
    print(f"Saved: {destination}")


def render_png_and_svg(source: Path, png_destination: Path, svg_destination: Path) -> None:
    """Render the first page of the assembled Figure 7 PDF to PNG and SVG."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required to render Figure 7 PNG/SVG outputs. "
            "Install dependencies from requirements.txt or environment.yml."
        ) from exc

    png_destination.parent.mkdir(parents=True, exist_ok=True)
    svg_destination.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(source)
    page = doc[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(4, 4), alpha=False)
    pix.save(png_destination)
    svg_destination.write_text(page.get_svg_image(), encoding="utf-8")
    doc.close()

    print(f"Saved: {png_destination}")
    print(f"Saved: {svg_destination}")


def main() -> None:
    source_pdf = resolve_source_pdf()
    validate_pdf(source_pdf)

    figure_pdf = FIGURES_DIR / "figure7.pdf"
    write_normalized_pdf(source_pdf, figure_pdf)
    render_png_and_svg(figure_pdf, FIGURES_DIR / "figure7.png", FIGURES_DIR / "figure7.svg")

    print(f"Source: {source_pdf}")


if __name__ == "__main__":
    main()
