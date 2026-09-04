"""Render the raster assets the project page uses, into ``docs/web/assets/``.

Everything here is derived from artifacts the rebuild already produces, so the
page can show real data without duplicating any of it:

* ``sequence_wall.png``: every animal's behavioural sequence as one row of
  cluster colours, 82 rows grouped as Control, ELS vulnerable and ELS resilient.
  It uses the same 250 ms bins, cluster mapping and colours as Figure 6, taken
  from that figure's own module, so it cannot drift from the published panels.
* ``paradigm.png``: panel A of Figure 1, the timeline of the paradigm.
* ``figure{3..6}_thumb.png``: the manuscript figures at 1400 px, on white,
  trimmed to their drawn area.

This is an authoring tool, not part of the rebuild: ``docs/`` sits outside the
artifact directories hashed by ``MANIFEST.csv``.

    uv run python scripts/build_page_media.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "web" / "assets"
FIGURES = ROOT / "figures"

# Wall geometry, in pixels: each 250 ms bin is BIN_PX wide, each animal ROW_PX
# tall, with a GROUP_GAP between the three profiles.
BIN_PX = 2
ROW_PX = 9
ROW_GAP = 1
GROUP_GAP = 14
PROFILE_ORDER = ("Control", "ELS vulnerable", "ELS resilient")
THUMB_WIDTH = 1400


def load_figure6():
    spec = importlib.util.spec_from_file_location(
        "fig6", ROOT / "scripts" / "generate_figures" / "figure_6_resilience_diversity.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hex_to_rgba(value: str) -> tuple[int, int, int, int]:
    value = value.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16), 255)


def profile_of(fig6, animal: str, group: str) -> str:
    if group == "Control":
        return "Control"
    return "ELS resilient" if animal in fig6.ELS_RESILIENT else "ELS vulnerable"


def build_sequence_wall(fig6) -> dict[str, int]:
    """One row per animal, ordered by profile, then by animal id."""
    pred, _seq, _full, meta = fig6.load_sequences()
    pred["Animal"] = pred["Animal"].astype(str)
    bins = sorted(pred["time_bin"].unique())
    position = {b: i for i, b in enumerate(bins)}
    colours = {name: hex_to_rgba(value) for name, value in fig6.COLORS.items()}

    rows: dict[str, list[str]] = {p: [] for p in PROFILE_ORDER}
    for _, row in meta.iterrows():
        animal = str(row["Animal"])
        rows[profile_of(fig6, animal, row["group"])].append(animal)
    for animals in rows.values():
        animals.sort(key=float)

    n_animals = sum(len(a) for a in rows.values())
    height = n_animals * (ROW_PX + ROW_GAP) + GROUP_GAP * (len(PROFILE_ORDER) - 1)
    width = len(bins) * BIN_PX
    canvas = np.zeros((height, width, 4), dtype=np.uint8)

    y = 0
    for profile in PROFILE_ORDER:
        for animal in rows[profile]:
            sub = pred[pred["Animal"] == animal]
            for time_bin, cluster in zip(sub["time_bin"], sub["cluster"]):
                if cluster not in colours:
                    continue  # unassigned bins stay transparent
                x = position[time_bin] * BIN_PX
                canvas[y : y + ROW_PX, x : x + BIN_PX] = colours[cluster]
            y += ROW_PX + ROW_GAP
        y += GROUP_GAP

    Image.fromarray(canvas, "RGBA").save(ASSETS / "sequence_wall.png", optimize=True)
    return {
        "animals": n_animals,
        "bins": len(bins),
        "width": width,
        "height": height,
        **{p: len(a) for p, a in rows.items()},
        "first_bin": float(bins[0]),
        "last_bin": float(bins[-1]),
    }


def build_paradigm() -> tuple[int, int]:
    """Panel A of Figure 1, without the panel letter."""
    image = Image.open(FIGURES / "figure1.png").convert("RGB")
    width, height = image.size
    panel = image.crop((int(width * 0.075), 0, width, int(height * 0.245)))
    target = 2000
    panel = panel.resize((target, round(target * panel.height / panel.width)), Image.LANCZOS)
    panel.save(ASSETS / "paradigm.png", optimize=True)
    return panel.size


def build_thumbs() -> dict[str, tuple[int, int]]:
    """The figures at web size, trimmed to the drawn area.

    The PNGs are full A4 pages, and some figures fill less than half of theirs;
    the blank remainder is cropped away so the panels get the pixels.
    """
    sizes = {}
    for number in (3, 4, 5, 6):
        source = Image.open(FIGURES / f"figure{number}.png").convert("RGBA")
        flat = Image.new("RGBA", source.size, (255, 255, 255, 255))
        flat.alpha_composite(source)
        flat = flat.convert("RGB")
        content = ImageChops.difference(flat, Image.new("RGB", flat.size, (255, 255, 255)))
        box = content.getbbox() or (0, 0, flat.width, flat.height)
        pad = round(flat.width * 0.02)
        flat = flat.crop(
            (
                max(box[0] - pad, 0),
                max(box[1] - pad, 0),
                min(box[2] + pad, flat.width),
                min(box[3] + pad, flat.height),
            )
        )
        target = (THUMB_WIDTH, round(THUMB_WIDTH * flat.height / flat.width))
        flat = flat.resize(target, Image.LANCZOS)
        name = f"figure{number}_thumb.png"
        flat.save(ASSETS / name, optimize=True)
        sizes[name] = target
    return sizes


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig6 = load_figure6()
    wall = build_sequence_wall(fig6)
    print("sequence_wall.png", wall)
    print("paradigm.png", build_paradigm())
    for name, size in build_thumbs().items():
        print(name, size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
