"""Build Supplementary Video 2, the grid view of the syllable atlas.

All 25 cluster-mapped syllables play simultaneously in a 5 x 5 grid so the
whole behavioral repertoire can be compared at a glance, complementing the
sequential atlas. The sources are the same archived clips as the sequential
video, and every clip is verified against the SHA-256 recorded in
``supplementary_media/Supplementary_Video_source_index.csv`` before use, so
the shipped index stays the single provenance record for both videos.

Run with the archived clip directory::

    python scripts/generate_supplementary_video_2_grid.py --clip-dir PATH/TO/video_clips
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_supplementary_video_1 import (  # noqa: E402
    CLUSTERS,
    COLORS,
    FPS,
    FRAMES_PER_OCCURRENCE,
    FRAMES_PER_SYLLABLE,
    OCCURRENCE_SELECTIONS,
    OUTPUT_INDEX,
    font,
    read_video_segment,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_VIDEO = ROOT / "supplementary_media/Supplementary_Video_2_MoSeq_syllable_grid.mp4"

SIZE = (1920, 1080)
BG = "#0C0E10"
INK = "#E8EAED"
MUTED = "#98A0A8"
RULE = "#262B30"

PANEL_W = 470
CELL = 196
GAP = 8
GRID_COLS = 5
GRID_W = GRID_COLS * CELL + (GRID_COLS - 1) * GAP
GRID_X = PANEL_W + (SIZE[0] - PANEL_W - GRID_W) // 2
GRID_Y = (SIZE[1] - GRID_W) // 2
STRIPE = 4

TITLE_FRAMES = 75
LOOPS = 8  # each tile loops its three 20-frame occurrences eight times

# Row-major display order: identical to the sequential video's cluster order,
# so same-cluster tiles sit adjacent and read as colored bands.
ORDER = [(cluster, syllable) for cluster, syllables in CLUSTERS for syllable in syllables]

F = {
    "eyebrow": font(20, True),
    "title": font(44, True),
    "hero": font(54, True),
    "body": font(21),
    "legend": font(22, True),
    "count": font(20),
    "members": font(18),
    "label": font(18, True),
    "small": font(17),
}


def load_tiles(clip_dir: Path) -> list[list[Image.Image]]:
    recorded: dict[int, str] = {}
    with OUTPUT_INDEX.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            recorded[int(row["syllable"])] = row["source_clip_sha256"]

    tiles: list[list[Image.Image]] = []
    for _, syllable in ORDER:
        clip = clip_dir / f"syllable_{syllable}_clip.mp4"
        if not clip.is_file():
            raise FileNotFoundError(f"Required source clip not found: {clip}")
        if sha256(clip) != recorded.get(syllable):
            raise RuntimeError(f"{clip.name} does not match the shipped source index")
        selections = OCCURRENCE_SELECTIONS.get(syllable)
        if selections:
            frames: list[Image.Image] = []
            for occurrence in selections:
                frames.extend(
                    read_video_segment(
                        clip,
                        FRAMES_PER_OCCURRENCE,
                        start_frame=occurrence * FRAMES_PER_OCCURRENCE,
                    )
                )
        else:
            frames = read_video_segment(clip, FRAMES_PER_SYLLABLE)
        tiles.append([f.resize((CELL, CELL), Image.Resampling.LANCZOS) for f in frames])
    return tiles


def build_background() -> Image.Image:
    image = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(image)
    x = 48
    draw.text((x, 64), "SUPPLEMENTARY VIDEO 2", font=F["eyebrow"], fill=MUTED)
    draw.text((x, 106), "Keypoint-MoSeq", font=F["title"], fill=INK)
    draw.text((x, 158), "syllable atlas", font=F["title"], fill=INK)
    draw.line((x, 236, PANEL_W - 48, 236), fill=RULE, width=2)

    # The three-line description of what the grid shows was dropped; the title
    # above and the labelled tiles beside it already say it. The legend moves
    # up into the space rather than leaving the rule stranded.
    #
    # Each cluster now names its member syllables instead of counting them.
    # "n = 8" said how many tiles carried the colour but not which, so the
    # legend could not be read against the grid; the bracketed list can.
    y = 286
    for cluster, syllables in CLUSTERS:
        draw.rounded_rectangle((x, y + 3, x + 22, y + 25), radius=5, fill=COLORS[cluster])
        draw.text((x + 36, y), cluster, font=F["legend"], fill=INK)
        members = "[" + ", ".join(str(syllable) for syllable in syllables) + "]"
        draw.text((x + 36, y + 26), members, font=F["members"], fill=MUTED)
        y += 64

    y = 920
    for line in (
        "Class 111 (Climb) is derived from arena-floor",
        "overlap rather than fitted by keypoint-MoSeq.",
        "",
        "Sources and SHA-256 provenance:",
        "Supplementary_Video_source_index.csv",
    ):
        draw.text((x, y), line, font=F["small"], fill=MUTED)
        y += 26
    return image


def build_overlay() -> Image.Image:
    overlay = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for index, (cluster, syllable) in enumerate(ORDER):
        col, row = index % GRID_COLS, index // GRID_COLS
        x0 = GRID_X + col * (CELL + GAP)
        y0 = GRID_Y + row * (CELL + GAP)
        x1, y1 = x0 + CELL, y0 + CELL
        draw.rectangle((x0, y0, x1 - 1, y0 + STRIPE - 1), fill=COLORS[cluster])
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline="#2A2F35", width=1)

        label = "111" if syllable == 111 else f"S{syllable}"
        box = draw.textbbox((0, 0), label, font=F["label"])
        width = box[2] - box[0]
        chip_left, chip_top = x0 + 6, y1 - 34
        draw.rounded_rectangle(
            (chip_left, chip_top, chip_left + width + 38, y1 - 6),
            radius=6,
            fill=(10, 12, 14, 215),
        )
        draw.ellipse(
            (chip_left + 9, chip_top + 9, chip_left + 19, chip_top + 19), fill=COLORS[cluster]
        )
        draw.text((chip_left + 26, chip_top + 3), label, font=F["label"], fill=INK)
    return overlay


def title_card() -> Image.Image:
    image = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(image)

    def centered(y: int, text: str, text_font, fill: str) -> None:
        box = draw.textbbox((0, 0), text, font=text_font)
        draw.text(((SIZE[0] - (box[2] - box[0])) / 2, y), text, font=text_font, fill=fill)

    # Number and title only, matching the sequential atlas's card: the line
    # describing the contents was dropped, and the block re-centred so the
    # cluster key does not sit adrift below it.
    centered(448, "SUPPLEMENTARY VIDEO 2", F["eyebrow"], MUTED)
    centered(492, "Keypoint-MoSeq syllable atlas", F["hero"], INK)

    widths = []
    for cluster, _ in CLUSTERS:
        box = draw.textbbox((0, 0), cluster, font=F["count"])
        widths.append(30 + (box[2] - box[0]))
    x = (SIZE[0] - (sum(widths) + 36 * (len(CLUSTERS) - 1))) / 2
    for (cluster, _), width in zip(CLUSTERS, widths):
        draw.rounded_rectangle((x, 614, x + 20, 634), radius=5, fill=COLORS[cluster])
        draw.text((x + 30, 612), cluster, font=F["count"], fill=INK)
        x += width + 36
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_VIDEO)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.clip_dir.is_dir():
        raise FileNotFoundError(f"MoSeq clip directory not found: {args.clip_dir}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    tiles = load_tiles(args.clip_dir)
    background = build_background()
    overlay = build_overlay()

    writer = imageio_ffmpeg.write_frames(
        str(args.output),
        SIZE,
        fps=FPS,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        output_params=["-crf", "20", "-preset", "medium", "-movflags", "+faststart", "-an"],
    )
    writer.send(None)

    title = np.asarray(title_card(), dtype=np.uint8).tobytes()
    for _ in range(TITLE_FRAMES):
        writer.send(title)

    for frame_index in range(LOOPS * FRAMES_PER_SYLLABLE):
        canvas = background.copy()
        for index, frames in enumerate(tiles):
            col, row = index % GRID_COLS, index // GRID_COLS
            canvas.paste(
                frames[frame_index % len(frames)],
                (GRID_X + col * (CELL + GAP), GRID_Y + row * (CELL + GAP)),
            )
        canvas.paste(overlay, (0, 0), overlay)
        writer.send(np.asarray(canvas, dtype=np.uint8).tobytes())
    writer.close()

    metadata = imageio_ffmpeg.read_frames(str(args.output), pix_fmt="rgb24")
    info = next(metadata)
    metadata.close()
    size_mb = args.output.stat().st_size / (1024 * 1024)
    if info.get("codec") != "h264" or info.get("pix_fmt", "").split("(")[0] != "yuv420p":
        raise RuntimeError(f"Unexpected output encoding: {info}")
    if size_mb > 30:
        raise RuntimeError(f"Output exceeds Nature's 30 MB per-file limit: {size_mb:.1f} MB")
    print(
        f"wrote {args.output} ({size_mb:.1f} MB; {info['duration']:.1f} s; {info['size']}; H.264)"
    )


if __name__ == "__main__":
    main()
