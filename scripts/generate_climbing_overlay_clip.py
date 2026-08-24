# SPDX-License-Identifier: MIT
"""Render the climbing-detection overlay panel used in Supplementary Video 1.

Supplementary Video 1 shows one panel per MoSeq syllable: a representative
pose overlay beside the canonical 14-point skeleton trajectory. The derived
climbing class (111) is the exception -- keypoint MoSeq never fitted it, so it
has no canonical trajectory and its panel carried three lines of explanatory
text where every other syllable shows an animation.

This script builds a real replacement for that text: the geometry that defines
climbing, drawn on the frames it actually selects. Each rendered frame shows

  * the annotated arena floor in yellow and the wall band in pink,
  * the convex hull of the tracked keypoints, with the part of the hull that
    still overlaps the floor filled solid,
  * the 14 tracked keypoints and the skeleton linking them,
  * the floor/wall overlap ratios that the climbing rule thresholds.

The skeleton drawn here is the *tracked* pose for each frame, not a canonical
MoSeq trajectory: class 111 has no fitted trajectory, and inventing one would
assert a model output that does not exist. The on-screen label says so.

Detection follows the documented rule (see docs/behavior_classifier.md):
frames whose hull-floor overlap falls below ``--floor-threshold`` (0.8) are
candidates, and only runs of at least ``--min-bout`` frames (17, i.e. 425 ms at
25 fps) are kept as climbing bouts.

Run against a tracked open-field session::

    python scripts/generate_climbing_overlay_clip.py \
        --video PATH/TO/Animal_10_Day_1.mpg \
        --keypoints PATH/TO/Animal_10_Day_1DLC_..._filtered.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from matplotlib.path import Path as MplPath
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "supplementary_media"

FPS = 25
FLOOR_THRESHOLD = 0.8  # hull-floor overlap below which a frame is a candidate
MIN_BOUT_FRAMES = 17  # 425 ms; shorter runs revert to the dominant label

# Keypoint hygiene. DeepLabCut occasionally places a low-confidence point on
# the far side of the arena; distance-from-median filtering alone tolerates
# such an outlier, and the inflated hull then drags the floor ratio down and
# fabricates a climbing frame. Points must be confident *and* anatomically
# plausible relative to the confident core of the pose.
LIKELIHOOD_THRESHOLD = 0.6
MAX_BODY_RADIUS = 140.0  # px from the pose anchor; hard anatomical ceiling
BODY_SPREAD_FACTOR = 2.5  # reject points this many times the pose's own spread
BODY_RADIUS_FLOOR = 60.0  # px, so a tightly curled pose stays intact
MIN_TRACKED_POINTS = 6

# The recording starts while the animal is still being handled, where DLC
# tracks the experimenter's hand. Require a sustained window of well-tracked
# frames before analysing.
SETTLE_WINDOW = 250
SETTLE_FRACTION = 0.9

FLOOR_RGB = np.array([255, 255, 0])  # yellow, as in the source analysis
WALL_RGB = np.array([255, 192, 203])  # pink
MASK_ALPHA = 0.45
HULL_FILL = (160, 32, 240, 110)
HULL_OVERLAP_FILL = (160, 32, 240, 205)
HULL_OUTLINE = (128, 0, 128)
SKELETON_RGB = (0, 225, 255)
ANNOTATION_SATURATION = 60  # min chroma for hand-drawn floor ink on a grey frame

# DeepLabCut bodypart order for this model, and the segments linking them.
BODYPARTS = [
    "nose",
    "H1R",
    "H2R",
    "H1L",
    "H2L",
    "B1R",
    "B2R",
    "B3R",
    "B1L",
    "B2L",
    "B3L",
    "tail",
    "S2",
    "S1",
]
SKELETON_EDGES = [
    (0, 1),
    (1, 2),
    (0, 3),
    (3, 4),
    (0, 5),
    (0, 8),
    (5, 6),
    (6, 7),
    (8, 9),
    (9, 10),
    (7, 11),
    (10, 11),
    (11, 12),
    (12, 13),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def font(size: int, bold: bool = False):
    for name in ("arialbd.ttf", "seguisb.ttf") if bold else ("arial.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


# --------------------------------------------------------------------------
# Geometry. The original analysis used shapely and OpenCV; neither is in this
# repository's environment, so the two operations actually needed -- exact
# polygon intersection area and point-in-polygon -- are implemented directly.
# --------------------------------------------------------------------------


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def clip_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of a convex subject against a convex window."""
    x, y = clip[:, 0], clip[:, 1]
    if 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) < 0:
        clip = clip[::-1]

    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0

    def intersect(p, q, a, b):
        r = np.array([q[0] - p[0], q[1] - p[1]])
        s = np.array([b[0] - a[0], b[1] - a[1]])
        denominator = r[0] * s[1] - r[1] * s[0]
        if abs(denominator) < 1e-12:
            return q
        t = ((a[0] - p[0]) * s[1] - (a[1] - p[1]) * s[0]) / denominator
        return np.array([p[0] + t * r[0], p[1] + t * r[1]])

    output = [np.asarray(p, dtype=float) for p in subject]
    for i in range(len(clip)):
        a, b = clip[i], clip[(i + 1) % len(clip)]
        if not output:
            break
        current, output = output, []
        for j in range(len(current)):
            point, previous = current[j], current[j - 1]
            if inside(point, a, b):
                if not inside(previous, a, b):
                    output.append(intersect(previous, point, a, b))
                output.append(point)
            elif inside(previous, a, b):
                output.append(intersect(previous, point, a, b))
    return np.array(output) if output else np.zeros((0, 2))


def clean_pose(points: np.ndarray, likelihood: np.ndarray) -> np.ndarray | None:
    """Confident keypoints lying within a plausible body radius, or None.

    Confidence alone is not enough: individual bodyparts (B1L in this model)
    are regularly placed a body-length away from the animal while still
    scoring above 0.9. The pose is therefore anchored on its confident core
    and scaled by its own spread, so a point is rejected when it is far
    relative to how compact the rest of the pose is, with a fixed ceiling for
    the anatomically impossible.
    """
    selected = likelihood > LIKELIHOOD_THRESHOLD
    if selected.sum() < MIN_TRACKED_POINTS:
        return None
    kept = points[selected]
    core = points[likelihood > 0.9]
    anchor = np.median(core, axis=0) if len(core) >= 4 else np.median(kept, axis=0)

    distances = np.linalg.norm(kept - anchor, axis=1)
    spread = np.median(distances)
    radius = min(MAX_BODY_RADIUS, max(BODY_RADIUS_FLOOR, BODY_SPREAD_FACTOR * spread))
    kept = kept[distances <= radius]
    return kept if len(kept) >= MIN_TRACKED_POINTS else None


def floor_overlap(points: np.ndarray, floor: np.ndarray) -> tuple[float, np.ndarray | None]:
    """Fraction of the keypoint convex hull that lies on the arena floor."""
    if points is None or len(points) < 3:
        return 1.0, None
    try:
        hull = points[ConvexHull(points).vertices]
    except Exception:
        return 1.0, None
    area = polygon_area(hull)
    if area <= 0:
        return 1.0, None
    return polygon_area(clip_polygon(hull, floor)) / area, hull


def find_bouts(condition: np.ndarray, minimum: int) -> list[tuple[int, int]]:
    bouts, start = [], None
    for i, flag in enumerate(condition):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= minimum:
                bouts.append((start, i - 1))
            start = None
    if start is not None and len(condition) - start >= minimum:
        bouts.append((start, len(condition) - 1))
    return bouts


# --------------------------------------------------------------------------
# Arena floor
# --------------------------------------------------------------------------


def median_frame(video: Path, samples: int = 40, stride: int = 120) -> np.ndarray:
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
    meta = next(reader)
    width, height = meta["size"]
    frames = []
    try:
        for i, raw in enumerate(reader):
            if i % stride == 0:
                frames.append(np.frombuffer(raw, np.uint8).reshape(height, width, 3).copy())
            if len(frames) >= samples:
                break
    finally:
        reader.close()
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def minimum_area_quad(points: np.ndarray) -> np.ndarray:
    """Smallest-area rectangle enclosing the points, via rotating calipers."""
    best = None
    for i in range(len(points)):
        edge = points[(i + 1) % len(points)] - points[i]
        norm = float(np.hypot(*edge))
        if norm < 1e-9:
            continue
        u = edge / norm
        v = np.array([-u[1], u[0]])
        a, b = points @ u, points @ v
        area = (a.max() - a.min()) * (b.max() - b.min())
        if best is None or area < best[0]:
            best = (area, u, v, a.min(), a.max(), b.min(), b.max())
    _, u, v, a0, a1, b0, b1 = best
    return np.array([u * a0 + v * b0, u * a1 + v * b0, u * a1 + v * b1, u * a0 + v * b1])


def snap_quad_to_edges(grey, quad, search=34, samples=90, smooth=2.0):
    """Move each side of `quad` onto the floor->wall intensity step.

    The coarse quad comes from a brightness threshold, so its sides sit
    wherever that threshold happened to cut the shading gradient. Each side is
    re-fitted here to the place where brightness actually falls off: along a
    set of normals to the side, the steepest inward-to-outward drop is located,
    and a line is fitted to those crossings by least squares on the inlier half
    (so a side occluded by clutter cannot drag the fit).
    """
    g = ndimage.gaussian_filter(grey, smooth)
    H, W = g.shape
    centre = quad.mean(axis=0)
    lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        edge = b - a
        length = np.hypot(*edge)
        if length < 1e-6:
            return quad
        tangent = edge / length
        normal = np.array([-tangent[1], tangent[0]])
        if np.dot(normal, centre - (a + b) / 2) > 0:
            normal = -normal  # normal points outward, floor -> wall

        offsets = np.arange(-search, search + 1, 1.0)
        found = []
        for t in np.linspace(0.08, 0.92, samples):
            base = a + edge * t
            pts = base[None, :] + offsets[:, None] * normal[None, :]
            xs = np.clip(pts[:, 0], 0, W - 1)
            ys = np.clip(pts[:, 1], 0, H - 1)
            profile = ndimage.map_coordinates(g, [ys, xs], order=1, mode="nearest")
            drop = -np.gradient(profile)  # positive where it darkens outward
            k = int(np.argmax(drop))
            if 0 < k < len(offsets) - 1 and drop[k] > 1.0:
                found.append(base + offsets[k] * normal)
        if len(found) < samples // 3:
            lines.append((a, b))
            continue
        found = np.array(found)

        # Fit x*sin - y*cos = r in the side's own frame, then drop outliers.
        proj = found @ normal
        keep = np.abs(proj - np.median(proj)) <= 2.5 * (
            np.median(np.abs(proj - np.median(proj))) + 1e-6
        )
        pts = found[keep] if keep.sum() >= 8 else found
        mean = pts.mean(axis=0)
        u, s, vt = np.linalg.svd(pts - mean)
        direction = vt[0]
        lines.append((mean, mean + direction))

    corners = []
    for i in range(4):
        p1, d1 = lines[i][0], lines[i][1] - lines[i][0]
        p2, d2 = lines[(i + 1) % 4][0], lines[(i + 1) % 4][1] - lines[(i + 1) % 4][0]
        denom = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(denom) < 1e-9:
            return quad
        t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / denom
        corners.append(p1 + t * d1)
    return np.array(corners)


def floor_from_annotation(image: Path) -> np.ndarray:
    """Read a hand-drawn arena floor back out of an annotated frame.

    Automatic segmentation has to infer the floor from brightness, which fails
    wherever shading, gloss or a visible wall face breaks the assumption. This
    restores what the legacy pipeline actually did -- a person marking the four
    floor corners -- by letting that person draw on an exported frame instead.

    Draw the floor in any strongly coloured ink (the frame itself is greyscale,
    so any saturated colour reads as annotation); outline it or fill it, either
    works. The marked pixels are taken as the floor and reduced to the
    quadrilateral that encloses them.
    """
    rgb = np.asarray(Image.open(image).convert("RGB"), dtype=int)
    saturation = rgb.max(axis=2) - rgb.min(axis=2)
    marked = saturation > ANNOTATION_SATURATION
    if marked.sum() < 200:
        raise SystemExit(
            f"{image.name}: found {int(marked.sum())} coloured pixels. Draw the "
            f"floor in a saturated colour (red, green, magenta) and save as PNG."
        )
    marked = ndimage.binary_closing(marked, np.ones((5, 5)))
    labels, count = ndimage.label(marked)
    sizes = ndimage.sum(marked, labels, range(1, count + 1))
    marked = labels == (int(np.argmax(sizes)) + 1)

    ys, xs = np.nonzero(marked)
    points = np.stack([xs, ys], axis=1).astype(float)
    quad = minimum_area_quad(points[ConvexHull(points).vertices])
    print(f"  floor: read from {image.name} ({int(marked.sum())} marked px)")
    return quad


def parse_corners(text: str) -> np.ndarray:
    """Parse 'x,y x,y x,y x,y' (or comma-separated) into four corners."""
    numbers = [float(v) for v in text.replace(",", " ").split()]
    if len(numbers) != 8:
        raise SystemExit(f"--floor-corners needs 8 numbers (4 x,y pairs); got {len(numbers)}")
    return np.array(numbers, dtype=float).reshape(4, 2)


def detect_floor(background: np.ndarray) -> np.ndarray:
    """Locate the arena floor as the bright plateau inside the wall band.

    The legacy pipeline read a hand-annotated ``*_annotated_mask.npy`` built by
    clicking the four floor corners. Those masks are not archived with this
    cohort, so the same quadrilateral is recovered from the video itself: the
    lit floor is markedly brighter than the wall band, and the largest bright
    component is the floor. The corner fit is reported so it can be checked.
    """
    grey = np.asarray(Image.fromarray(background).convert("L"), dtype=float)
    smooth = ndimage.gaussian_filter(grey, 3)

    height_, width_ = grey.shape
    best = None
    for threshold in range(100, 160, 2):
        mask = ndimage.binary_opening(smooth > threshold, np.ones((15, 15)))
        mask = ndimage.binary_fill_holes(mask)
        labels, count = ndimage.label(mask)
        if count == 0:
            continue
        sizes = ndimage.sum(mask, labels, range(1, count + 1))
        component = labels == (int(np.argmax(sizes)) + 1)
        if component.sum() < 0.15 * grey.size:
            continue
        ys, xs = np.nonzero(component)
        # A component reaching the frame border has leaked past the wall band
        # into the surround; the floor is always fully enclosed by the walls.
        if xs.min() <= 2 or ys.min() <= 2 or xs.max() >= width_ - 3 or ys.max() >= height_ - 3:
            continue
        extent = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
        rectangularity = component.sum() / extent
        # Take the most box-like region rather than the largest: dimmer
        # thresholds merge the wall band in, brighter ones break the floor up,
        # and the floor itself is the only strongly rectangular candidate.
        if best is None or rectangularity > best[2]:
            best = (threshold, component, rectangularity)
    if best is None:
        raise RuntimeError("Could not segment an arena floor from the video background")

    component = best[1]
    ys, xs = np.nonzero(component)
    points = np.stack([xs, ys], axis=1).astype(float)
    quad = minimum_area_quad(points[ConvexHull(points).vertices])
    # The threshold above lands somewhere on the shading gradient rather than
    # on the rim itself, so finish by snapping each side to the brightness step.
    quad = snap_quad_to_edges(grey, quad)

    height, width = grey.shape
    yy, xx = np.mgrid[0:height, 0:width]
    inside = (
        MplPath(quad)
        .contains_points(np.stack([xx.ravel(), yy.ravel()], axis=1))
        .reshape(height, width)
    )
    coverage = (inside & component).sum() / component.sum()
    spill = (inside & ~component).sum() / inside.sum()
    print(
        f"  floor: threshold {best[0]}, rectangularity {best[2]:.3f}, "
        f"coverage {coverage:.1%}, spill {spill:.1%}"
    )
    return quad


# --------------------------------------------------------------------------
# Keypoints
# --------------------------------------------------------------------------


def count_frames(video: Path) -> int:
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
    next(reader)
    total = sum(1 for _ in reader)
    reader.close()
    return total


def check_alignment(video: Path, keypoints: np.ndarray) -> None:
    """Refuse to draw a pose onto frames it was not tracked from.

    DeepLabCut reads videos through OpenCV while this script decodes with
    ffmpeg, and on MPEG sources the two disagree about how many frames the
    file contains. When they do, frame i of the decode is not frame i of the
    tracking, and the skeleton is silently drawn on the wrong picture. Use
    DeepLabCut's own labelled render, whose frame count matches the table by
    construction, rather than guessing an offset.
    """
    total = count_frames(video)
    if total != len(keypoints):
        raise SystemExit(
            f"Frame-count mismatch: {video.name} decodes {total} frames but "
            f"{len(keypoints)} were tracked (difference {total - len(keypoints)}). "
            f"Pass the matching '*_filtered_labeled.mp4' render instead, which "
            f"contains exactly the frames DeepLabCut processed."
        )


def load_keypoints(path: Path) -> np.ndarray:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    parts = rows[1][1:]
    names = [parts[i] for i in range(0, len(parts), 3)]
    if names != BODYPARTS:
        raise RuntimeError(f"Unexpected bodyparts in {path.name}: {names}")
    data = np.array([[float(v) for v in row[1:]] for row in rows[3:]])
    return data.reshape(len(data), len(BODYPARTS), 3)


def tracked_window(likelihood: np.ndarray) -> tuple[int, int]:
    """First and last frame of the sustained well-tracked period."""
    good = (likelihood > LIKELIHOOD_THRESHOLD).sum(axis=1) >= 8
    smoothed = np.convolve(good.astype(float), np.ones(SETTLE_WINDOW) / SETTLE_WINDOW, "same")
    indices = np.where(smoothed > SETTLE_FRACTION)[0]
    if len(indices) == 0:
        raise RuntimeError("No sustained well-tracked window found")
    return int(indices.min()), int(indices.max())


def compute_floor_ratios(keypoints: np.ndarray, floor: np.ndarray) -> np.ndarray:
    start, stop = tracked_window(keypoints[:, :, 2])
    ratios = np.full(len(keypoints), np.nan)
    for frame in range(start, stop + 1):
        pose = clean_pose(keypoints[frame, :, :2], keypoints[frame, :, 2])
        if pose is None:
            continue
        ratios[frame] = floor_overlap(pose, floor)[0]
    return ratios


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def shade_arena(frame: np.ndarray, inside: np.ndarray) -> Image.Image:
    tinted = frame.astype(float)
    tinted[inside] = tinted[inside] * (1 - MASK_ALPHA) + FLOOR_RGB * MASK_ALPHA
    tinted[~inside] = tinted[~inside] * (1 - MASK_ALPHA) + WALL_RGB * MASK_ALPHA
    return Image.fromarray(tinted.astype(np.uint8))


def draw_frame(
    frame: np.ndarray,
    inside: np.ndarray,
    floor: np.ndarray,
    points: np.ndarray,
    likelihood: np.ndarray,
    index: int,
    climbing: bool,
) -> Image.Image:
    image = shade_arena(frame, inside)
    draw = ImageDraw.Draw(image, "RGBA")

    pose = clean_pose(points, likelihood)
    ratio, hull = floor_overlap(pose, floor)
    if hull is not None:
        draw.polygon([tuple(p) for p in hull], fill=HULL_FILL, outline=HULL_OUTLINE)
        overlap = clip_polygon(hull, floor)
        if len(overlap) >= 3:
            draw.polygon([tuple(p) for p in overlap], fill=HULL_OVERLAP_FILL)

    kept = set() if pose is None else {tuple(np.round(p, 2)) for p in pose}
    for a, b in SKELETON_EDGES:
        if tuple(np.round(points[a], 2)) in kept and tuple(np.round(points[b], 2)) in kept:
            draw.line([tuple(points[a]), tuple(points[b])], fill=SKELETON_RGB, width=3)
    for i in range(len(BODYPARTS)):
        if tuple(np.round(points[i], 2)) in kept:
            x, y = points[i]
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 255, 255), outline=(0, 0, 0))

    label = font(26, True)
    small = font(20)
    draw.text(
        (16, 12),
        f"Floor {ratio:.2f}  |  Wall {1 - ratio:.2f}",
        font=label,
        fill=(255, 255, 0),
        stroke_width=2,
        stroke_fill=(40, 40, 40),
    )
    draw.text(
        (16, 46),
        f"Frame {index}",
        font=small,
        fill=(255, 255, 255),
        stroke_width=2,
        stroke_fill=(40, 40, 40),
    )
    if climbing:
        draw.text(
            (16, 74),
            "CLIMBING",
            font=label,
            fill=(246, 166, 90),
            stroke_width=2,
            stroke_fill=(40, 40, 40),
        )
    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--keypoints", type=Path, required=True)
    parser.add_argument(
        "--floor-image",
        type=Path,
        help="frame with the arena floor drawn on in colour; overrides automatic floor detection",
    )
    parser.add_argument(
        "--floor-corners",
        type=str,
        help="four floor corners as 'x,y x,y x,y x,y'; overrides automatic floor detection",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "climbing_overlay_clip.mp4")
    parser.add_argument("--floor-threshold", type=float, default=FLOOR_THRESHOLD)
    parser.add_argument("--min-bout", type=int, default=MIN_BOUT_FRAMES)
    parser.add_argument("--bouts", type=int, default=6, help="number of climbing bouts to render")
    parser.add_argument(
        "--context", type=int, default=10, help="frames of lead-in/lead-out around each bout"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.video, args.keypoints):
        if not path.is_file():
            raise FileNotFoundError(path)

    print(f"video     {args.video.name}")
    print(f"keypoints {args.keypoints.name}")

    background = median_frame(args.video)
    if args.floor_corners:
        floor = parse_corners(args.floor_corners)
        print(f"  floor: taken from --floor-corners {floor.astype(int).tolist()}")
    elif args.floor_image:
        floor = floor_from_annotation(args.floor_image)
    else:
        floor = detect_floor(background)
    keypoints = load_keypoints(args.keypoints)
    check_alignment(args.video, keypoints)
    print(f"  tracked {len(keypoints)} frames, {len(BODYPARTS)} bodyparts")

    ratios = compute_floor_ratios(keypoints, floor)
    valid = ~np.isnan(ratios)
    bouts = find_bouts((ratios < args.floor_threshold) & valid, args.min_bout)
    climbing_frames = sum(end - start + 1 for start, end in bouts)
    print(f"  floor overlap mean {np.nanmean(ratios):.3f}")
    print(
        f"  {len(bouts)} climbing bouts, {climbing_frames} frames "
        f"({100 * climbing_frames / max(valid.sum(), 1):.1f}% of tracked session)"
    )
    if not bouts:
        raise SystemExit("No climbing bouts detected under the documented rule")

    selected = bouts[: args.bouts]
    wanted: list[int] = []
    for start, end in selected:
        lo = max(0, start - args.context)
        hi = min(len(keypoints) - 1, end + args.context)
        wanted.extend(range(lo, hi + 1))
    wanted_set = set(wanted)

    reader = imageio_ffmpeg.read_frames(str(args.video), pix_fmt="rgb24")
    meta = next(reader)
    width, height = meta["size"]
    frames: dict[int, np.ndarray] = {}
    try:
        for i, raw in enumerate(reader):
            if i in wanted_set:
                frames[i] = np.frombuffer(raw, np.uint8).reshape(height, width, 3).copy()
            if i > max(wanted_set):
                break
    finally:
        reader.close()

    yy, xx = np.mgrid[0:height, 0:width]
    inside = (
        MplPath(floor)
        .contains_points(np.stack([xx.ravel(), yy.ravel()], axis=1))
        .reshape(height, width)
    )

    climbing_set = {i for start, end in selected for i in range(start, end + 1)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(args.output),
        (width, height),
        fps=FPS,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        output_params=["-crf", "20", "-preset", "medium", "-movflags", "+faststart", "-an"],
    )
    writer.send(None)
    for i in wanted:
        if i not in frames:
            continue
        image = draw_frame(
            frames[i], inside, floor, keypoints[i, :, :2], keypoints[i, :, 2], i, i in climbing_set
        )
        writer.send(np.asarray(image.convert("RGB"), dtype=np.uint8).tobytes())
    writer.close()

    sidecar = args.output.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "video": args.video.name,
                "video_sha256": sha256(args.video),
                "keypoints": args.keypoints.name,
                "keypoints_sha256": sha256(args.keypoints),
                "floor_quad_xy": np.round(floor, 2).tolist(),
                "floor_threshold": args.floor_threshold,
                "min_bout_frames": args.min_bout,
                "likelihood_threshold": LIKELIHOOD_THRESHOLD,
                "max_body_radius_px": MAX_BODY_RADIUS,
                "bouts_detected": len(bouts),
                "bouts_rendered": [[int(s), int(e)] for s, e in selected],
                "climbing_fraction_of_tracked": round(
                    float(climbing_frames / max(valid.sum(), 1)), 4
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"wrote {args.output} ({size_mb:.1f} MB, {len(wanted)} frames)")
    print(f"wrote {sidecar}")


if __name__ == "__main__":
    main()
