# SPDX-License-Identifier: MIT
"""Per-animal ethograms and barcodes, in the style of Figures 4 and 6.

The manuscript shows one representative animal per condition. Readers who want
to judge how representative that choice was need the whole set, so this renders
every mouse and files them by plot type:

    supplementary_media/ethograms/<animal>_ethogram.png/.pdf
    supplementary_media/barcodes/<animal>_barcode.png/.pdf

The drawing routines are imported from figure_6_resilience_diversity rather
than reimplemented, so the panels are identical to the published ones - same
cluster order, colours, event bands, axis treatment and typography - and they
cannot drift apart if that figure is restyled.

    python scripts/generate_per_animal_media.py
    python scripts/generate_per_animal_media.py --animals 2.4 11.4
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "supplementary_media"
ETHOGRAMS = MEDIA / "ethograms"
BARCODES = MEDIA / "barcodes"

# The twelve resilient ELS animals, from the Figure 5 Euclidean classification.
RESILIENT = {"2.4", "15.4", "26.4", "43.3", "43.5", "49.6",
             "123.3", "134.2", "148.4", "150.3", "150.5", "159.4"}


def load_figure6():
    spec = importlib.util.spec_from_file_location(
        "fig6", ROOT / "scripts" / "generate_figures" / "figure_6_resilience_diversity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render(fig6, pred, animal: str, profile: str, kind: str, out: Path) -> None:
    """One panel per file, drawn by the figure's own routine."""
    if kind == "ethogram":
        figure = plt.figure(figsize=(4.6, 2.15), dpi=300, facecolor="white")
        ax = figure.add_axes((0.20, 0.30, 0.74, 0.55))
        fig6.plot_ethogram(ax, pred, animal, "", show_legend=True)
    else:
        figure = plt.figure(figsize=(4.6, 1.30), dpi=300, facecolor="white")
        ax = figure.add_axes((0.06, 0.42, 0.88, 0.34))
        fig6.plot_barcode(ax, pred, animal, "", show_legend=True)
    # add_events draws the tone and shock bands above the axes, so the title
    # needs clearance or it lands on top of them.
    ax.set_title(f"Animal {animal} · {profile}", fontsize=6.2,
                 color=fig6.AXIS, pad=14 if kind == "ethogram" else 10)
    # Built by concatenation, not Path.with_suffix: animal ids contain a dot,
    # so with_suffix would treat "2.4_ethogram" as stem "2" plus a suffix.
    for extension in (".png", ".pdf"):
        figure.savefig(out / f"{animal}_{kind}{extension}",
                       facecolor="white", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animals", nargs="*", default=None,
                        help="animal ids to render; default is all")
    args = parser.parse_args()

    fig6 = load_figure6()
    pred, _seq, _full, meta = fig6.load_sequences()
    info = meta.set_index("Animal").to_dict("index")

    ETHOGRAMS.mkdir(parents=True, exist_ok=True)
    BARCODES.mkdir(parents=True, exist_ok=True)

    animals = args.animals or sorted(
        pred["Animal"].astype(str).unique(), key=lambda a: float(a))
    rows = []
    for animal in animals:
        if animal not in info:
            print(f"  skipped {animal}: not in metadata")
            continue
        group = info[animal]["group"]
        profile = ("Control" if group == "Control"
                   else "ELS resilient" if animal in RESILIENT else "ELS vulnerable")
        render(fig6, pred, animal, profile, "ethogram", ETHOGRAMS)
        render(fig6, pred, animal, profile, "barcode", BARCODES)
        rows.append({"animal": animal, "group": group, "profile": profile,
                     "experiment": info[animal]["Experiment"],
                     "litter": str(animal).split(".")[0],
                     "ethogram": f"ethograms/{animal}_ethogram.png",
                     "barcode": f"barcodes/{animal}_barcode.png"})

    index = pd.DataFrame(rows)
    index.to_csv(MEDIA / "per_animal_index.csv", index=False)
    print(f"rendered {len(index)} animals")
    print(f"  {ETHOGRAMS}")
    print(f"  {BARCODES}")
    print(f"  {MEDIA / 'per_animal_index.csv'}")


if __name__ == "__main__":
    main()
