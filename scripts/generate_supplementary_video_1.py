"""Build Supplementary Video 1 from the archived keypoint-MoSeq clips.

The source working tree contains pose-overlaid representative occurrences for
syllables 0--34 and the geometry-derived climbing label 111, plus an animated
atlas of the corresponding canonical skeleton trajectories. This script turns
those files into one concise, journal-ready H.264 video and records source-file
hashes for provenance.

Run with explicit source paths on another computer::

    python scripts/generate_supplementary_video_1.py \
        --clip-dir PATH/TO/video_clips \
        --skeleton-gif PATH/TO/skeleton_trajectories.gif
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Iterable

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "supplementary_media"
OUTPUT_VIDEO = OUTPUT_DIR / "Supplementary_Video_1_MoSeq_syllable_atlas.mp4"
OUTPUT_INDEX = OUTPUT_DIR / "Supplementary_Video_1_source_index.csv"

FPS = 25
SIZE = (960, 540)
FRAMES_PER_SYLLABLE = 60  # Three complete 20-frame representative occurrences.
TITLE_FRAMES = 75
END_FRAMES = 50

# The earliest S34 examples include the animal leaving the camera field. Use
# three complete archived occurrences for the submission atlas. Values are
# zero-based occurrence indices in the source's 20-frame concatenation.
OCCURRENCE_SELECTIONS = {34: (6, 23, 60)}

CLUSTERS: list[tuple[str, list[int]]] = [
    ("Freeze", [0, 28]),
    ("Sniff", [18, 20]),
    ("Groom", [24]),
    ("Turn", [1, 3, 5, 6, 10, 15, 26, 27]),
    ("Locomotion", [11, 12, 14, 16, 19, 21, 25]),
    ("Climb", [111]),
    ("Jump", [23, 29, 30, 34]),
    ("Unassigned", [2, 4, 7, 8, 9, 13, 17, 22, 31, 32, 33]),
]

COLORS = {
    "Freeze": "#BE7AA5",
    "Sniff": "#5A9DAE",
    "Groom": "#8BC2D9",
    "Turn": "#AFD93D",
    "Locomotion": "#F5C12D",
    "Climb": "#F6A65A",
    "Jump": "#E85655",
    "Unassigned": "#A7A7A7",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONTS = {
    "title": font(38, True),
    "subtitle": font(24),
    "cluster": font(32, True),
    "syllable": font(42, True),
    "body": font(20),
    "small": font(16),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_video_segment(path: Path, count: int, start_frame: int = 0) -> list[Image.Image]:
    input_params = ["-ss", f"{start_frame / FPS:.6f}"] if start_frame else None
    reader = imageio_ffmpeg.read_frames(
        str(path), pix_fmt="rgb24", input_params=input_params
    )
    metadata = next(reader)
    width, height = metadata["size"]
    frames: list[Image.Image] = []
    try:
        for raw in reader:
            array = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
            frames.append(Image.fromarray(array.copy(), "RGB"))
            if len(frames) == count:
                break
    finally:
        reader.close()
    if len(frames) < count:
        raise RuntimeError(f"{path} contains only {len(frames)} readable frames")
    return frames


def load_skeleton_cells(path: Path) -> dict[int, list[Image.Image]]:
    animation = Image.open(path)
    frames: list[Image.Image] = []
    for frame_index in range(animation.n_frames):
        animation.seek(frame_index)
        frames.append(animation.convert("RGB").copy())

    union = np.zeros((frames[0].height, frames[0].width), dtype=bool)
    for frame in frames:
        union |= np.any(np.asarray(frame) < 245, axis=2)
    ys, xs = np.where(union)
    if not len(xs):
        raise RuntimeError(f"No skeleton content detected in {path}")
    xmin, xmax = int(xs.min()), int(xs.max()) + 1
    ymin, ymax = int(ys.min()), int(ys.max()) + 1

    ncols, nrows = 4, 9
    cell_width = (xmax - xmin) / ncols
    cell_height = (ymax - ymin) / nrows
    cells: dict[int, list[Image.Image]] = {}
    for syllable in range(35):
        row, col = divmod(syllable, ncols)
        # The source atlas prints labels at the top of each cell. The final
        # video provides its own larger label, so crop to the pose region and
        # inset the side boundaries to prevent neighbouring label fragments
        # from entering the enlarged canonical-skeleton panel.
        left = max(0, round(xmin + (col + 0.06) * cell_width))
        right = min(frames[0].width, round(xmin + (col + 0.94) * cell_width))
        top = max(0, round(ymin + (row + 0.18) * cell_height))
        bottom = min(frames[0].height, round(ymin + (row + 0.98) * cell_height))
        cells[syllable] = []
        for frame in frames:
            cell = frame.crop((left, top, right, bottom))
            # Remove the atlas's small embedded label. It is redundant with the
            # large, accessible label drawn by the final-video layout.
            ImageDraw.Draw(cell).rectangle(
                (0, 0, cell.width, round(cell.height * 0.42)), fill="white"
            )
            cells[syllable].append(cell)
    return cells


def centered_text(draw: ImageDraw.ImageDraw, y: int, text: str, text_font, fill: str) -> None:
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text(((SIZE[0] - (box[2] - box[0])) / 2, y), text, font=text_font, fill=fill)


def title_frame() -> Image.Image:
    image = Image.new("RGB", SIZE, "#FAFAFA")
    draw = ImageDraw.Draw(image)
    centered_text(draw, 112, "Supplementary Video 1", FONTS["subtitle"], "#555555")
    centered_text(draw, 168, "MoSeq syllable atlas", FONTS["title"], "#303030")
    centered_text(draw, 225, "Representative pose-overlaid occurrences and", FONTS["subtitle"], "#4A4A4A")
    centered_text(draw, 259, "canonical 14-point skeleton trajectories", FONTS["subtitle"], "#4A4A4A")
    centered_text(draw, 340, "Syllables 0–34 plus the geometry-derived climbing class", FONTS["body"], "#666666")
    centered_text(draw, 410, "Playback: 25 frames s⁻¹; no audio", FONTS["small"], "#777777")
    return image


def end_frame() -> Image.Image:
    image = Image.new("RGB", SIZE, "#FAFAFA")
    draw = ImageDraw.Draw(image)
    centered_text(draw, 155, "Interpretation", FONTS["title"], "#303030")
    centered_text(draw, 225, "The white dot marks frames assigned to the displayed syllable.", FONTS["body"], "#555555")
    centered_text(draw, 265, "Unassigned motifs were excluded from the seven curated", FONTS["body"], "#555555")
    centered_text(draw, 295, "ethological classes but retained in sequence-level analyses.", FONTS["body"], "#555555")
    centered_text(draw, 365, "Climbing (111) was defined by the documented floor-overlap rule.", FONTS["body"], "#555555")
    return image


def render_syllable_frame(
    source: Image.Image,
    skeleton: Image.Image | None,
    cluster: str,
    syllable: int,
    frame_index: int,
) -> Image.Image:
    canvas = Image.new("RGB", SIZE, "#F7F7F7")
    draw = ImageDraw.Draw(canvas)

    source_fit = ImageOps.contain(source, (500, 500), Image.Resampling.LANCZOS)
    canvas.paste(source_fit, (20 + (500 - source_fit.width) // 2, 20 + (500 - source_fit.height) // 2))
    draw.rectangle((20, 20, 520, 520), outline="#C8C8C8", width=2)
    draw.text((34, 34), "Representative pose overlay", font=FONTS["small"], fill="white",
              stroke_width=2, stroke_fill="#303030")

    color = COLORS[cluster]
    draw.rounded_rectangle((550, 28, 920, 78), radius=12, fill=color)
    draw.text((570, 34), cluster, font=FONTS["cluster"], fill="#202020")
    label = "Derived class 111" if syllable == 111 else f"MoSeq syllable {syllable}"
    draw.text((550, 95), label, font=FONTS["syllable"], fill="#303030")

    occurrence = frame_index // 20 + 1
    draw.text((552, 147), f"Representative occurrence {occurrence}", font=FONTS["small"], fill="#666666")

    if skeleton is not None:
        skeleton_fit = ImageOps.contain(skeleton, (370, 285), Image.Resampling.LANCZOS)
        x = 550 + (370 - skeleton_fit.width) // 2
        y = 180 + (285 - skeleton_fit.height) // 2
        canvas.paste(skeleton_fit, (x, y))
        draw.text((552, 465), "Canonical skeleton trajectory", font=FONTS["small"], fill="#666666")
    else:
        draw.text((552, 245), "Climbing is identified from", font=FONTS["body"], fill="#555555")
        draw.text((552, 278), "arena-floor overlap and does not", font=FONTS["body"], fill="#555555")
        draw.text((552, 311), "have a canonical MoSeq skeleton.", font=FONTS["body"], fill="#555555")

    if cluster == "Unassigned":
        draw.text((552, 495), "Not assigned to an ethological class", font=FONTS["small"], fill="#666666")
    return canvas


def frame_bytes(images: Iterable[Image.Image]) -> Iterable[bytes]:
    for image in images:
        yield np.asarray(image.convert("RGB"), dtype=np.uint8).tobytes()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--skeleton-gif", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_VIDEO)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.clip_dir.is_dir():
        raise FileNotFoundError(f"MoSeq clip directory not found: {args.clip_dir}")
    if not args.skeleton_gif.is_file():
        raise FileNotFoundError(f"Skeleton atlas not found: {args.skeleton_gif}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    skeleton_cells = load_skeleton_cells(args.skeleton_gif)
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

    for raw in frame_bytes([title_frame()] * TITLE_FRAMES):
        writer.send(raw)

    index_rows: list[dict[str, str | int]] = []
    for cluster, syllables in CLUSTERS:
        for syllable in syllables:
            clip = args.clip_dir / f"syllable_{syllable}_clip.mp4"
            if not clip.is_file():
                raise FileNotFoundError(f"Required source clip not found: {clip}")
            selected_occurrences = OCCURRENCE_SELECTIONS.get(syllable)
            if selected_occurrences:
                source_frames = []
                for occurrence_index in selected_occurrences:
                    source_frames.extend(
                        read_video_segment(
                            clip, 20, start_frame=occurrence_index * 20
                        )
                    )
                occurrence_record = ";".join(
                    str(index + 1) for index in selected_occurrences
                )
            else:
                source_frames = read_video_segment(clip, FRAMES_PER_SYLLABLE)
                occurrence_record = "1;2;3"
            skeleton_frames = skeleton_cells.get(syllable)
            for frame_index, source in enumerate(source_frames):
                skeleton = None
                if skeleton_frames:
                    skeleton = skeleton_frames[(frame_index * len(skeleton_frames) // FPS) % len(skeleton_frames)]
                writer.send(
                    np.asarray(
                        render_syllable_frame(source, skeleton, cluster, syllable, frame_index),
                        dtype=np.uint8,
                    ).tobytes()
                )
            index_rows.append(
                {
                    "display_order": len(index_rows) + 1,
                    "syllable": syllable,
                    "behavior_cluster": cluster,
                    "source_clip": clip.name,
                    "source_clip_sha256": sha256(clip),
                    "source_occurrences_1_based": occurrence_record,
                    "canonical_skeleton": "yes" if skeleton_frames else "no; derived climbing class",
                }
            )

    for raw in frame_bytes([end_frame()] * END_FRAMES):
        writer.send(raw)
    writer.close()

    with OUTPUT_INDEX.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer_csv.writeheader()
        writer_csv.writerows(index_rows)

    metadata = imageio_ffmpeg.read_frames(str(args.output), pix_fmt="rgb24")
    info = next(metadata)
    metadata.close()
    size_mb = args.output.stat().st_size / (1024 * 1024)
    if info.get("codec") != "h264" or info.get("pix_fmt", "").split("(")[0] != "yuv420p":
        raise RuntimeError(f"Unexpected output encoding: {info}")
    if size_mb > 30:
        raise RuntimeError(f"Output exceeds Nature's 30 MB per-file limit: {size_mb:.1f} MB")
    print(f"wrote {args.output} ({size_mb:.1f} MB; {info['duration']:.1f} s; {info['size']}; H.264)")
    print(f"wrote {OUTPUT_INDEX} ({len(index_rows)} indexed entries)")


if __name__ == "__main__":
    main()
