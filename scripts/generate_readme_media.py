"""Build the animated media embedded in README.md.

Two GIFs are written to ``docs/media/``:

* ``syllable_atlas.gif`` -- one representative tile per behavioral cluster,
  taken from the same archived clips as Supplementary Video 1 and verified
  against the SHA-256 recorded in its source index.
* ``figure_tour.gif`` -- a slow slideshow over the main data figures, rendered
  from the shipped ``figures/*.png`` plates.

docs/media/ is repository presentation material, not a manuscript artifact, so
it is deliberately outside the MANIFEST.csv artifact directories.

Run with the archived clip directory::

    python scripts/generate_readme_media.py --clip-dir PATH/TO/video_clips
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_supplementary_video_1 import (  # noqa: E402
    COLORS,
    OUTPUT_INDEX,
    font,
    read_video_segment,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]
MEDIA_DIR = ROOT / "docs/media"

BG = "#0C0E10"
INK = "#E8EAED"

# One representative syllable per cluster, in the atlas's cluster order.
STRIP_TILES = [
    ("Freeze", 0),
    ("Sniff", 18),
    ("Groom", 24),
    ("Turn", 3),
    ("Locomotion", 12),
    ("Jump", 23),
]
TILE = 140
TILE_GAP = 6
STRIP_FRAMES = 60  # three 20-frame occurrences, decimated 2x on write
STRIP_STEP = 2
STRIP_MS = 80  # per written frame: 12.5 fps

FIGURES = [
    (2, "Pipeline validation"),
    (4, "Diversity dynamics"),
    (5, "Resilience dynamics"),
    (6, "Resilience and diversity"),
    (7, "Predicting resilience"),
]
TOUR_HEIGHT = 840
TOUR_CAPTION = 44
TOUR_MS = 2800


def build_syllable_strip(clip_dir: Path) -> None:
    recorded: dict[int, str] = {}
    with OUTPUT_INDEX.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            recorded[int(row["syllable"])] = row["source_clip_sha256"]

    label_font = font(15, True)
    width = len(STRIP_TILES) * TILE + (len(STRIP_TILES) - 1) * TILE_GAP
    tiles: list[list[Image.Image]] = []
    for _, syllable in STRIP_TILES:
        clip = clip_dir / f"syllable_{syllable}_clip.mp4"
        if sha256(clip) != recorded.get(syllable):
            raise RuntimeError(f"{clip.name} does not match the shipped source index")
        frames = read_video_segment(clip, STRIP_FRAMES)
        tiles.append([f.resize((TILE, TILE), Image.Resampling.LANCZOS) for f in frames])

    written: list[Image.Image] = []
    for frame_index in range(0, STRIP_FRAMES, STRIP_STEP):
        canvas = Image.new("RGB", (width, TILE), BG)
        draw = ImageDraw.Draw(canvas)
        for index, ((cluster, _), frames) in enumerate(zip(STRIP_TILES, tiles)):
            x0 = index * (TILE + TILE_GAP)
            canvas.paste(frames[frame_index], (x0, 0))
            draw.rectangle((x0, 0, x0 + TILE - 1, 2), fill=COLORS[cluster])
            box = draw.textbbox((0, 0), cluster, font=label_font)
            chip_w = (box[2] - box[0]) + 18
            draw.rounded_rectangle(
                (x0 + 5, TILE - 27, x0 + 5 + chip_w, TILE - 5),
                radius=5,
                fill=(10, 12, 14),
            )
            draw.text((x0 + 14, TILE - 25), cluster, font=label_font, fill=INK)
        written.append(canvas.quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=0))

    out = MEDIA_DIR / "syllable_atlas.gif"
    written[0].save(
        out, save_all=True, append_images=written[1:], duration=STRIP_MS, loop=0, optimize=True
    )
    print(f"wrote {out} ({out.stat().st_size / 1024 / 1024:.1f} MB; {len(written)} frames)")


def build_figure_tour() -> None:
    caption_font = font(22, True)
    frames: list[Image.Image] = []
    plates = []
    for number, title in FIGURES:
        plate = Image.open(ROOT / f"figures/figure{number}.png").convert("RGB")
        scale = (TOUR_HEIGHT - TOUR_CAPTION) / plate.height
        plates.append(
            (
                number,
                title,
                plate.resize(
                    (round(plate.width * scale), TOUR_HEIGHT - TOUR_CAPTION),
                    Image.Resampling.LANCZOS,
                ),
            )
        )
    width = max(plate.width for _, _, plate in plates) + 40

    for number, title, plate in plates:
        canvas = Image.new("RGB", (width, TOUR_HEIGHT), "white")
        canvas.paste(plate, ((width - plate.width) // 2, 0))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, TOUR_HEIGHT - TOUR_CAPTION, width, TOUR_HEIGHT), fill="#14171A")
        caption = f"Figure {number}  ·  {title}"
        box = draw.textbbox((0, 0), caption, font=caption_font)
        draw.text(
            ((width - (box[2] - box[0])) / 2, TOUR_HEIGHT - TOUR_CAPTION + 9),
            caption,
            font=caption_font,
            fill=INK,
        )
        frames.append(canvas.quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=0))

    out = MEDIA_DIR / "figure_tour.gif"
    frames[0].save(
        out, save_all=True, append_images=frames[1:], duration=TOUR_MS, loop=0, optimize=True
    )
    print(f"wrote {out} ({out.stat().st_size / 1024 / 1024:.1f} MB; {len(frames)} frames)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.clip_dir.is_dir():
        raise FileNotFoundError(f"MoSeq clip directory not found: {args.clip_dir}")
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    build_syllable_strip(args.clip_dir)
    build_figure_tour()


if __name__ == "__main__":
    main()
