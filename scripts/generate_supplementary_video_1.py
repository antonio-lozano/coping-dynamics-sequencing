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
import sys
from pathlib import Path
from typing import Iterable

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageSequence
from scipy import ndimage


sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "supplementary_media"
OUTPUT_VIDEO = OUTPUT_DIR / "Supplementary_Video_1_MoSeq_syllable_atlas.mp4"
OUTPUT_INDEX = OUTPUT_DIR / "Supplementary_Video_1_source_index.csv"

FPS = 25
SIZE = (960, 540)
FRAMES_PER_SYLLABLE = 60  # Three complete 20-frame representative occurrences.
TITLE_FRAMES = 75

# The earliest S34 examples include the animal leaving the camera field. Use
# three complete archived occurrences for the submission atlas. Values are
# zero-based occurrence indices in the source's 20-frame concatenation.
OCCURRENCE_SELECTIONS = {34: (6, 23, 60)}

# Skeleton segmentation in the source atlas.
SKELETON_SATURATION = 40   # min chroma to count as pose ink, not label text
SKELETON_MIN_AREA = 150    # px, discards stray anti-aliasing speckles
SKELETON_COUNT = 35        # syllables 0-34 drawn in the atlas
SKELETON_ROW_GAP = 60      # px between atlas rows
SKELETON_PAD = 10          # px of breathing room around each pose
SKELETON_DILATE = 3        # px, keeps the anti-aliased rim and black outline

CLUSTERS: list[tuple[str, list[int]]] = [
    ("Freeze", [0, 28]),
    ("Sniff", [18, 20]),
    ("Groom", [24]),
    ("Turn", [1, 3, 5, 6, 10, 15, 26, 27]),
    ("Locomotion", [11, 12, 14, 16, 19, 21, 25]),
    ("Climb", [111]),
    ("Jump", [23, 29, 30, 34]),
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
    """Cut the atlas into one transparent, tightly-cropped clip per syllable.

    The atlas is a 4x9 grid of matplotlib panels, each with a black text label
    above a coloured 14-point skeleton. Slicing it on an assumed grid pitch cut
    the wider skeletons off at the cell edge and let neighbouring labels bleed
    in. Instead the skeletons are isolated by chroma: the pose is drawn in
    saturated yellows, oranges and reds, while every label is neutral black, so
    a saturation threshold segments the 35 skeletons and nothing else.

    Each syllable is then cropped to its own bounding box taken over the whole
    animation, so the pose never leaves the crop and never jitters between
    frames, and the white page is returned as alpha rather than as pixels.
    """
    animation = Image.open(path)
    frames = [f.convert("RGB").copy() for f in ImageSequence.Iterator(animation)]
    stack = np.stack([np.asarray(f).astype(np.int16) for f in frames])

    # Chroma mask: coloured ink only. Black labels have saturation ~0.
    saturation = stack.max(axis=3) - stack.min(axis=3)
    coloured = (saturation > SKELETON_SATURATION).any(axis=0)

    labels, count = ndimage.label(coloured, structure=np.ones((3, 3), dtype=int))
    if count < 1:
        raise RuntimeError(f"No coloured skeletons found in {path}")
    areas = ndimage.sum(coloured, labels, range(1, count + 1))
    slices = ndimage.find_objects(labels)
    blobs = [slices[i] for i, area in enumerate(areas) if area >= SKELETON_MIN_AREA]
    if len(blobs) != SKELETON_COUNT:
        raise RuntimeError(
            f"Expected {SKELETON_COUNT} skeletons in {path}, segmented {len(blobs)}"
        )

    # Row-major order matches the atlas's Syllable 0..34 layout. Rows are
    # quantised before sorting so a few pixels of vertical drift within a row
    # cannot reorder it.
    rows = sorted(blobs, key=lambda sl: ((sl[0].start + sl[0].stop) // 2))
    row_of: dict[int, int] = {}
    current, last = 0, None
    for sl in rows:
        centre = (sl[0].start + sl[0].stop) // 2
        if last is not None and centre - last > SKELETON_ROW_GAP:
            current += 1
        row_of[id(sl)] = current
        last = centre
    ordered = sorted(blobs, key=lambda sl: (row_of[id(sl)], sl[1].start))

    # Each syllable keeps only its own connected component. Padding the crop
    # can pull in a neighbouring pose or label, so the component mask - grown
    # slightly to keep the anti-aliased rim and the black outline - gates the
    # alpha channel.
    own = np.zeros_like(labels, dtype=bool)
    cells: dict[int, list[Image.Image]] = {}
    for syllable, sl in enumerate(ordered):
        own[:] = False
        component = labels[sl]
        own[sl] = component == np.bincount(component[component > 0].ravel()).argmax()
        mask = ndimage.binary_dilation(own, iterations=SKELETON_DILATE)

        pad = SKELETON_PAD
        top = max(0, sl[0].start - pad)
        bottom = min(stack.shape[1], sl[0].stop + pad)
        left = max(0, sl[1].start - pad)
        right = min(stack.shape[2], sl[1].stop + pad)
        window = mask[top:bottom, left:right]

        clips: list[Image.Image] = []
        for frame in frames:
            cell = frame.crop((left, top, right, bottom)).convert("RGBA")
            data = np.asarray(cell).astype(np.int16)
            # The page is white; turn it into alpha so the panel background of
            # the finished video shows through instead of a grey-white square.
            lightness = data[:, :, :3].min(axis=2)
            alpha = np.clip((250 - lightness) * 6, 0, 255)
            alpha = np.where(window, alpha, 0).astype(np.uint8)
            out = data.copy()
            out[:, :, 3] = alpha
            clips.append(Image.fromarray(out.astype(np.uint8), "RGBA"))
        cells[syllable] = clips
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
    return image


def render_syllable_frame(
    source: Image.Image,
    skeleton: Image.Image | None,
    cluster: str,
    syllable: int,
    frame_index: int,
    climb: Image.Image | None = None,
    occurrence_override: int | None = None,
) -> Image.Image:
    canvas = Image.new("RGB", SIZE, "#F7F7F7")
    draw = ImageDraw.Draw(canvas)

    # Climbing is defined geometrically, so its representative overlay is the
    # geometry itself: floor/wall segmentation, keypoint hull and tracked
    # skeleton, with the floor-overlap ratio the rule thresholds.
    panel_source = climb if (climb is not None and syllable == 111) else source
    caption = ("Representative pose overlay - floor/wall overlap"
               if panel_source is climb else "Representative pose overlay")

    source_fit = ImageOps.contain(panel_source, (500, 500), Image.Resampling.LANCZOS)
    canvas.paste(source_fit, (20 + (500 - source_fit.width) // 2, 20 + (500 - source_fit.height) // 2))
    draw.rectangle((20, 20, 520, 520), outline="#C8C8C8", width=2)
    draw.text((34, 34), caption, font=FONTS["small"], fill="white",
              stroke_width=2, stroke_fill="#303030")

    color = COLORS[cluster]
    draw.rounded_rectangle((550, 28, 920, 78), radius=12, fill=color)
    draw.text((570, 34), cluster, font=FONTS["cluster"], fill="#202020")
    label = "Derived class 111" if syllable == 111 else f"MoSeq syllable {syllable}"
    draw.text((550, 95), label, font=FONTS["syllable"], fill="#303030")

    if occurrence_override is not None:
        draw.text((552, 147), f"Climbing bout {occurrence_override}",
                  font=FONTS["small"], fill="#666666")
    else:
        occurrence = frame_index // 20 + 1
        draw.text((552, 147), f"Representative occurrence {occurrence}",
                  font=FONTS["small"], fill="#666666")

    if skeleton is not None:
        # Fit inside the panel rather than onto a fixed box, and composite
        # through the alpha built in load_skeleton_cells so the pose sits on the
        # video background instead of on a white tile.
        panel = (550, 180, 940, 455)
        box = (panel[2] - panel[0], panel[3] - panel[1])
        skeleton_fit = ImageOps.contain(skeleton, box, Image.Resampling.LANCZOS)
        x = panel[0] + (box[0] - skeleton_fit.width) // 2
        y = panel[1] + (box[1] - skeleton_fit.height) // 2
        canvas.paste(skeleton_fit, (x, y), skeleton_fit)
        draw.text((552, 465), "Canonical skeleton trajectory", font=FONTS["small"], fill="#666666")
    elif panel_source is climb:
        # The overlay itself carries the geometry, so the right-hand column
        # states the rule rather than repeating the footage.
        draw.text((552, 200), "Yellow: annotated arena floor", font=FONTS["body"], fill="#555555")
        draw.text((552, 232), "Pink: wall band", font=FONTS["body"], fill="#555555")
        draw.text((552, 264), "Purple: convex hull of the", font=FONTS["body"], fill="#555555")
        draw.text((552, 292), "tracked keypoints", font=FONTS["body"], fill="#555555")
        draw.text((552, 336), "Climbing when floor overlap", font=FONTS["body"], fill="#555555")
        draw.text((552, 364), "stays below 0.8 for at least", font=FONTS["body"], fill="#555555")
        draw.text((552, 392), "17 frames (425 ms).", font=FONTS["body"], fill="#555555")
    return canvas


def frame_bytes(images: Iterable[Image.Image]) -> Iterable[bytes]:
    for image in images:
        yield np.asarray(image.convert("RGB"), dtype=np.uint8).tobytes()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--skeleton-gif", type=Path, required=True)
    parser.add_argument("--climb-video", type=Path,
                        help="tracked session used for the climbing geometry panel")
    parser.add_argument("--climb-keypoints", type=Path,
                        help="DeepLabCut keypoints matching --climb-video")
    parser.add_argument("--climb-bouts", type=int, default=6,
                        help="separate climbing bouts to show in the 111 panel")
    parser.add_argument("--climb-bout-frames", type=int, default=45,
                        help="frames shown per climbing bout")
    parser.add_argument("--climb-lead", type=int, default=8,
                        help="frames of floor lead-in before each bout starts")
    parser.add_argument("--output", type=Path, default=OUTPUT_VIDEO)
    return parser.parse_args()


def load_climb_overlay(
    video: Path, keypoints: Path, bout_count: int, frames_per_bout: int,
    lead_frames: int,
) -> tuple[list[Image.Image], list[int]]:
    """Render climbing-overlay frames for several separate bouts.

    One continuous bout reads as a single event, so the panel samples bouts
    spread across the session. Each excerpt opens with a few frames of lead-in
    taken from before the bout starts, so the animal is first seen on the floor
    with a high overlap ratio and the transition onto the wall is visible
    rather than assumed.

    Returns the frames and, for each, the 1-based index of the bout it shows.
    """
    from generate_climbing_overlay_clip import (
        FLOOR_THRESHOLD, MIN_BOUT_FRAMES, check_alignment, compute_floor_ratios,
        detect_floor, draw_frame, find_bouts, load_keypoints, median_frame,
    )
    from matplotlib.path import Path as MplPath

    floor = detect_floor(median_frame(video))
    points = load_keypoints(keypoints)
    check_alignment(video, points)
    ratios = compute_floor_ratios(points, floor)
    valid = ~np.isnan(ratios)
    bouts = find_bouts((ratios < FLOOR_THRESHOLD) & valid, MIN_BOUT_FRAMES)
    if not bouts:
        raise RuntimeError(f"No climbing bouts detected in {keypoints.name}")

    # Spread the excerpts over the whole session rather than taking the first
    # few, which tend to cluster in the first minute.
    if len(bouts) > bout_count:
        step = len(bouts) / bout_count
        bouts = [bouts[int(i * step)] for i in range(bout_count)]

    wanted: list[int] = []
    bout_of: list[int] = []
    for number, (start, end) in enumerate(bouts, start=1):
        first = max(0, start - lead_frames)
        excerpt = list(range(first, min(end, first + frames_per_bout - 1) + 1))
        wanted.extend(excerpt)
        bout_of.extend([number] * len(excerpt))

    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
    metadata = next(reader)
    width, height = metadata["size"]
    raw_frames: dict[int, np.ndarray] = {}
    limit, needed = max(wanted), set(wanted)
    try:
        for i, raw in enumerate(reader):
            if i in needed:
                raw_frames[i] = np.frombuffer(raw, np.uint8).reshape(height, width, 3).copy()
            if i >= limit:
                break
    finally:
        reader.close()

    yy, xx = np.mgrid[0:height, 0:width]
    inside = MplPath(floor).contains_points(
        np.stack([xx.ravel(), yy.ravel()], axis=1)
    ).reshape(height, width)

    climbing = {i for start, end in bouts for i in range(start, end + 1)}
    images, numbers = [], []
    for frame, number in zip(wanted, bout_of):
        if frame not in raw_frames:
            continue
        images.append(draw_frame(raw_frames[frame], inside, floor,
                                 points[frame, :, :2], points[frame, :, 2],
                                 frame, frame in climbing))
        numbers.append(number)
    print(f"climbing overlay: {len(images)} frames from {len(bouts)} bouts")
    return images, numbers


def main() -> None:
    args = parse_args()
    if not args.clip_dir.is_dir():
        raise FileNotFoundError(f"MoSeq clip directory not found: {args.clip_dir}")
    if not args.skeleton_gif.is_file():
        raise FileNotFoundError(f"Skeleton atlas not found: {args.skeleton_gif}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    skeleton_cells = load_skeleton_cells(args.skeleton_gif)

    climb_frames: list[Image.Image] = []
    climb_bouts: list[int] = []
    if args.climb_video and args.climb_keypoints:
        climb_frames, climb_bouts = load_climb_overlay(
            args.climb_video, args.climb_keypoints, args.climb_bouts,
            args.climb_bout_frames, args.climb_lead,
        )

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
            # The climbing panel is driven by its own excerpts, so it runs for
            # as many frames as those excerpts provide rather than reusing the
            # archived clip's length.
            uses_climb = syllable == 111 and bool(climb_frames)
            total_frames = len(climb_frames) if uses_climb else len(source_frames)

            for frame_index in range(total_frames):
                source = source_frames[frame_index % len(source_frames)]
                skeleton = None
                if skeleton_frames:
                    skeleton = skeleton_frames[(frame_index * len(skeleton_frames) // FPS) % len(skeleton_frames)]
                climb = climb_frames[frame_index] if uses_climb else None
                occurrence = climb_bouts[frame_index] if uses_climb else None
                writer.send(
                    np.asarray(
                        render_syllable_frame(source, skeleton, cluster, syllable,
                                              frame_index, climb, occurrence),
                        dtype=np.uint8,
                    ).tobytes()
                )
            if uses_climb:
                occurrence_record = ";".join(str(n) for n in sorted(set(climb_bouts)))
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
