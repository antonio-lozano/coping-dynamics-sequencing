# SPDX-License-Identifier: MIT
"""Build Supplementary Video 2: the geometry-derived climbing class.

Climbing could not be recovered by keypoint MoSeq from top-down video, so it is
labelled by a floor-overlap rule: the convex hull of the animal's DeepLabCut
keypoints is intersected with a manually annotated arena-floor mask, and frames
whose hull falls largely off the floor are grouped into bouts and called
climbing (Supplementary Figure 4 shows the geometry on still frames).

This video shows the archived representative occurrences of that class so the
rule can be judged against the behaviour it selects, which a static figure
cannot convey.

Run with an explicit source path on another computer::

    python scripts/generate_supplementary_video_2.py \
        --clip PATH/TO/video_clips/syllable_111_clip.mp4
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
OUTPUT_VIDEO = OUTPUT_DIR / "Supplementary_Video_2_climbing_detection.mp4"
OUTPUT_INDEX = OUTPUT_DIR / "Supplementary_Video_2_source_index.csv"

FPS = 25
SIZE = (960, 540)
TITLE_FRAMES = 75
END_FRAMES = 50
OCCURRENCE_FRAMES = 20      # the archive concatenates 20-frame occurrences
N_OCCURRENCES = 12          # keep the video short enough to embed
CLIMB_COLOR = "#F4A259"


def load_fonts() -> dict[str, ImageFont.FreeTypeFont]:
    def pick(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        for name in (("arialbd.ttf", "seguisb.ttf") if bold else ("arial.ttf", "segoeui.ttf")):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()
    return {"title": pick(38, True), "subtitle": pick(22), "body": pick(19),
            "small": pick(15), "cluster": pick(26, True), "label": pick(30, True)}


FONTS = load_fonts()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_clip(path: Path) -> list[Image.Image]:
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    meta = next(reader)
    width, height = meta["size"]
    return [Image.frombytes("RGB", (width, height), raw) for raw in reader]


def centered(draw: ImageDraw.ImageDraw, y: int, text: str, font, fill: str) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((SIZE[0] - (box[2] - box[0])) / 2, y), text, font=font, fill=fill)


def title_frame() -> Image.Image:
    image = Image.new("RGB", SIZE, "#FAFAFA")
    draw = ImageDraw.Draw(image)
    centered(draw, 118, "Supplementary Video 2", FONTS["subtitle"], "#555555")
    centered(draw, 172, "Geometry-derived climbing detection", FONTS["title"], "#303030")
    centered(draw, 240, "Climbing is not recoverable from top-down keypoint MoSeq;", FONTS["body"], "#4A4A4A")
    centered(draw, 270, "it is labelled where the convex hull of the tracked points", FONTS["body"], "#4A4A4A")
    centered(draw, 300, "leaves the annotated arena floor (Supplementary Figure 4).", FONTS["body"], "#4A4A4A")
    centered(draw, 370, "Representative occurrences of the resulting class", FONTS["body"], "#666666")
    centered(draw, 420, "Playback: 25 frames s⁻¹; no audio", FONTS["small"], "#777777")
    return image


def end_frame() -> Image.Image:
    image = Image.new("RGB", SIZE, "#FAFAFA")
    draw = ImageDraw.Draw(image)
    centered(draw, 165, "Rule", FONTS["title"], "#303030")
    centered(draw, 235, "Frames with floor overlap below 0.8 are candidate climbing frames.", FONTS["body"], "#555555")
    centered(draw, 270, "Candidates are kept only in bouts longer than 425 ms (17 frames);", FONTS["body"], "#555555")
    centered(draw, 305, "shorter runs revert to the dominant non-climbing label.", FONTS["body"], "#555555")
    return image


def render(frame: Image.Image, index: int, total: int) -> Image.Image:
    canvas = Image.new("RGB", SIZE, "#F7F7F7")
    draw = ImageDraw.Draw(canvas)

    fit = ImageOps.contain(frame, (500, 500), Image.Resampling.LANCZOS)
    canvas.paste(fit, (20 + (500 - fit.width) // 2, 20 + (500 - fit.height) // 2))
    draw.rectangle((20, 20, 520, 520), outline="#C8C8C8", width=2)
    draw.text((34, 34), "Representative pose overlay", font=FONTS["small"], fill="white",
              stroke_width=2, stroke_fill="#303030")

    draw.rounded_rectangle((550, 28, 920, 78), radius=12, fill=CLIMB_COLOR)
    draw.text((570, 36), "Climb", font=FONTS["cluster"], fill="#202020")
    draw.text((550, 95), "Derived class 111", font=FONTS["label"], fill="#303030")
    draw.text((552, 140), f"Representative occurrence {index // OCCURRENCE_FRAMES + 1}"
                          f" of {total // OCCURRENCE_FRAMES}", font=FONTS["small"], fill="#666666")

    draw.text((552, 210), "Detected by convex-hull", font=FONTS["body"], fill="#555555")
    draw.text((552, 240), "floor overlap, not by", font=FONTS["body"], fill="#555555")
    draw.text((552, 270), "keypoint MoSeq.", font=FONTS["body"], fill="#555555")
    draw.text((552, 330), "Two-dimensional top-down video", font=FONTS["small"], fill="#666666")
    draw.text((552, 352), "cannot represent vertical motion,", font=FONTS["small"], fill="#666666")
    draw.text((552, 374), "so the pose model alone misses it.", font=FONTS["small"], fill="#666666")

    bar_w = int(360 * (index + 1) / total)
    draw.rectangle((550, 470, 910, 478), fill="#E2E2E2")
    draw.rectangle((550, 470, 550 + bar_w, 478), fill=CLIMB_COLOR)
    return canvas


def frame_bytes(images: Iterable[Image.Image]) -> Iterable[bytes]:
    for image in images:
        yield np.asarray(image.convert("RGB"), dtype=np.uint8).tobytes()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_VIDEO)
    args = parser.parse_args()
    if not args.clip.is_file():
        raise FileNotFoundError(f"Climbing clip not found: {args.clip}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source = read_clip(args.clip)
    total = min(len(source), OCCURRENCE_FRAMES * N_OCCURRENCES)
    body = [render(source[i], i, total) for i in range(total)]
    frames = [title_frame()] * TITLE_FRAMES + body + [end_frame()] * END_FRAMES

    writer = imageio_ffmpeg.write_frames(
        str(args.output), SIZE, fps=FPS, codec="libx264", quality=None,
        macro_block_size=1, ffmpeg_log_level="error",
        output_params=["-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    writer.send(None)
    for payload in frame_bytes(frames):
        writer.send(payload)
    writer.close()

    with OUTPUT_INDEX.open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["role", "source_file", "sha256", "frames_used"])
        w.writerow(["climbing clip", args.clip.name, sha256(args.clip), total])

    size_mb = args.output.stat().st_size / 1e6
    print(f"wrote {args.output} ({size_mb:.1f} MB; {len(frames)/FPS:.1f} s; {SIZE}; H.264)")
    print(f"wrote {OUTPUT_INDEX}")


if __name__ == "__main__":
    main()
