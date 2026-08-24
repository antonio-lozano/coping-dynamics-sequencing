# SPDX-License-Identifier: MIT
"""Automatic arena floor/wall segmentation and keypoint-hull geometry.

The climbing class (syllable 111) is defined geometrically: the convex hull of
the DeepLabCut keypoints is intersected with the arena floor, and frames whose
hull sits largely off the floor belong to climbing bouts. This script derives
that geometry from the video alone, with no hand annotation:

  1. A median background over sampled frames removes the animal.
  2. The background is illumination-flattened (divided by a heavily blurred
     copy), which removes the shading gradient that defeats a single global
     threshold on this footage.
  3. A threshold sweep segments the bright floor plateau; each candidate is
     scored by rectangularity and the most box-like enclosed component wins.
  4. The component is reduced to a quadrilateral (minimum-area rectangle of
     its convex hull) and each side is then snapped independently to the
     strongest brightness step nearby, so the fit lands on the wall rim
     rather than somewhere on the gradient.
  5. Per frame, keypoints above a likelihood threshold form a convex hull;
     the fraction of hull area over the floor is the floor ratio and its
     complement the wall ratio. Rendered frames show the floor in yellow,
     the wall band in pink, and the hull in purple with the floor-overlap
     part filled solid, matching Supplementary Figure 5.

Outputs: the fitted floor corners as JSON, per-frame floor/wall ratios as
CSV, and a montage of rendered frames (by default the strongest wall-overlap
moments, i.e. climbing candidates). Example::

    python scripts/segment_arena_geometry.py \
        --video "PATH/TO/Trial 12_mouse10.mp4" \
        --keypoints "PATH/TO/Trial 12_mouse10DLC_...filtered.csv" \
        --out-dir OUTPUT_DIR
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from matplotlib.path import Path as MplPath
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull

FLOOR_TINT = np.array([196, 190, 20], dtype=float)
WALL_TINT = np.array([196, 150, 170], dtype=float)
HULL_COLOR = (150, 20, 150)
TINT_ALPHA = 0.45
HULL_FILL_ALPHA = 0.85


# --------------------------------------------------------------------------
# Video access
# --------------------------------------------------------------------------


def stream_frames(video: Path):
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
    meta = next(reader)
    width, height = meta["size"]
    for raw in reader:
        yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)


def median_background(video: Path, samples: int = 48) -> np.ndarray:
    """Median of evenly spaced frames; the moving animal falls out of it."""
    meta = next(imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24"))
    estimated = max(int(meta.get("fps", 25) * meta.get("duration", 0)), samples)
    stride = max(1, estimated // samples)
    kept = [f for i, f in enumerate(stream_frames(video)) if i % stride == 0]
    return np.median(np.stack(kept), axis=0).astype(np.uint8)


# --------------------------------------------------------------------------
# Floor detection
# --------------------------------------------------------------------------


def flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Divide by a heavily blurred copy to level the shading gradient."""
    blur = ndimage.gaussian_filter(gray, sigma=min(gray.shape) * 0.25)
    flat = gray / np.maximum(blur, 1.0)
    return flat * (gray.mean() / flat.mean())


def best_floor_component(flat: np.ndarray) -> np.ndarray:
    """Sweep thresholds; keep the most rectangular enclosed bright region."""
    height, width = flat.shape
    best_component, best_score = None, -1.0
    for percentile in np.linspace(35, 85, 26):
        mask = ndimage.binary_opening(flat > np.percentile(flat, percentile), np.ones((13, 13)))
        mask = ndimage.binary_fill_holes(mask)
        labels, count = ndimage.label(mask)
        if count == 0:
            continue
        sizes = ndimage.sum(mask, labels, range(1, count + 1))
        component = labels == (int(np.argmax(sizes)) + 1)
        if component.sum() < 0.15 * flat.size:
            continue
        ys, xs = np.nonzero(component)
        # The floor is fully enclosed by the walls; a component reaching the
        # frame border has leaked into the surround.
        if xs.min() <= 2 or ys.min() <= 2 or xs.max() >= width - 3 or ys.max() >= height - 3:
            continue
        extent = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
        score = component.sum() / extent
        if score > best_score:
            best_component, best_score = component, score
    if best_component is None:
        raise RuntimeError("No enclosed rectangular bright region found; not an arena video?")
    return best_component


def min_area_rectangle(points: np.ndarray) -> np.ndarray:
    """Rotating calipers over the convex hull's edge directions."""
    hull = points[ConvexHull(points).vertices]
    edges = np.diff(np.vstack([hull, hull[:1]]), axis=0)
    angles = np.unique(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), np.pi / 2))
    best_corners, best_area = None, np.inf
    for angle in angles:
        rot = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        rotated = hull @ rot.T
        lo, hi = rotated.min(axis=0), rotated.max(axis=0)
        area = np.prod(hi - lo)
        if area < best_area:
            corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
            best_corners, best_area = corners @ rot, area
    return best_corners


def snap_sides_to_rim(gray: np.ndarray, quad: np.ndarray, search: int = 14) -> np.ndarray:
    """Move each side along its outward normal onto the strongest brightness step.

    The threshold sweep lands somewhere on the floor-to-wall gradient; the rim
    itself is the sharpest step in a small window around each fitted side.
    Sides move independently, so the rectangle relaxes into a quadrilateral.
    """
    smooth = ndimage.gaussian_filter(gray, 2)
    center = quad.mean(axis=0)
    shifted_lines = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        direction = (b - a) / np.linalg.norm(b - a)
        normal = np.array([-direction[1], direction[0]])
        if np.dot((a + b) / 2 - center, normal) < 0:
            normal = -normal
        ts = np.linspace(0.15, 0.85, 25)
        base = a[None, :] + ts[:, None] * (b - a)[None, :]
        strengths = []
        for offset in range(-search, search + 1):
            probe_in = base + (offset - 2) * normal
            probe_out = base + (offset + 2) * normal
            inside = ndimage.map_coordinates(smooth, [probe_in[:, 1], probe_in[:, 0]], order=1)
            outside = ndimage.map_coordinates(smooth, [probe_out[:, 1], probe_out[:, 0]], order=1)
            strengths.append(np.median(inside - outside))
        offset = int(np.argmax(strengths)) - search
        shifted_lines.append((a + offset * normal, direction))
    corners = []
    for i in range(4):
        (p1, d1), (p2, d2) = shifted_lines[i - 1], shifted_lines[i]
        matrix = np.array([d1, -d2]).T
        t = np.linalg.solve(matrix, p2 - p1)
        corners.append(p1 + t[0] * d1)
    return np.array(corners)


def detect_floor(background: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the floor quadrilateral and its rasterized mask."""
    gray = np.asarray(Image.fromarray(background).convert("L"), dtype=float)
    component = best_floor_component(flatten_illumination(gray))
    boundary = component ^ ndimage.binary_erosion(component)
    ys, xs = np.nonzero(boundary)
    quad = min_area_rectangle(np.stack([xs, ys], axis=1).astype(float))
    quad = snap_sides_to_rim(gray, quad)
    height, width = gray.shape
    yy, xx = np.mgrid[0:height, 0:width]
    mask = (
        MplPath(quad)
        .contains_points(np.stack([xx.ravel(), yy.ravel()], axis=1))
        .reshape(height, width)
    )
    coverage = (mask & component).sum() / component.sum()
    print(f"floor: corners {np.round(quad, 1).tolist()}, component coverage {coverage:.1%}")
    return quad, mask


# --------------------------------------------------------------------------
# Keypoints and hull geometry
# --------------------------------------------------------------------------


def load_dlc_keypoints(path: Path) -> tuple[list[str], np.ndarray]:
    """Read a DeepLabCut CSV -> (bodyparts, array of frames x parts x (x, y, p))."""
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    parts_row = rows[1][1:]
    bodyparts = [parts_row[i] for i in range(0, len(parts_row), 3)]
    data = np.array([[float(v) for v in row[1:]] for row in rows[3:]])
    return bodyparts, data.reshape(len(data), len(bodyparts), 3)


def hull_floor_ratio(points: np.ndarray, floor_mask: np.ndarray) -> tuple[float, np.ndarray]:
    """Fraction of the keypoint hull's area lying on the floor, plus the hull."""
    hull = ConvexHull(points)
    polygon = points[hull.vertices]
    height, width = floor_mask.shape
    x0, y0 = np.clip(polygon.min(axis=0).astype(int) - 1, 0, [width - 1, height - 1])
    x1, y1 = np.clip(polygon.max(axis=0).astype(int) + 2, 0, [width, height])
    yy, xx = np.mgrid[y0:y1, x0:x1]
    inside = (
        MplPath(polygon)
        .contains_points(np.stack([xx.ravel(), yy.ravel()], axis=1))
        .reshape(yy.shape)
    )
    area = inside.sum()
    if area == 0:
        return np.nan, polygon
    on_floor = (inside & floor_mask[y0:y1, x0:x1]).sum()
    return on_floor / area, polygon


# A hull from too few or poorly tracked points degenerates into a spike that
# misstates the overlap, so frames must clear both gates before they count.
MIN_HULL_POINTS = 6


def all_frame_ratios(
    keypoints: np.ndarray, floor_mask: np.ndarray, likelihood: float
) -> np.ndarray:
    ratios = np.full(len(keypoints), np.nan)
    for index, frame in enumerate(keypoints):
        good = frame[frame[:, 2] > likelihood, :2]
        if len(good) >= MIN_HULL_POINTS:
            ratios[index], _ = hull_floor_ratio(good, floor_mask)
    return ratios


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def render_frame(
    frame: np.ndarray,
    floor_mask: np.ndarray,
    keypoints: np.ndarray | None,
    likelihood: float,
) -> tuple[Image.Image, float]:
    tinted = frame.astype(float)
    tinted[floor_mask] = (1 - TINT_ALPHA) * tinted[floor_mask] + TINT_ALPHA * FLOOR_TINT
    tinted[~floor_mask] = (1 - TINT_ALPHA) * tinted[~floor_mask] + TINT_ALPHA * WALL_TINT
    ratio = np.nan
    image = Image.fromarray(tinted.astype(np.uint8))
    if keypoints is not None:
        good = keypoints[keypoints[:, 2] > likelihood, :2]
        if len(good) >= MIN_HULL_POINTS:
            ratio, polygon = hull_floor_ratio(good, floor_mask)
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            fill = (*HULL_COLOR, int(255 * HULL_FILL_ALPHA))
            draw.polygon(
                [tuple(p) for p in polygon], fill=fill, outline=(*HULL_COLOR, 255), width=2
            )
            # Solid fill only where the hull overlaps the floor; over the wall
            # the hull stays an outline so the underlying animal is visible.
            hull_layer = np.asarray(overlay).copy()
            off_floor = ~floor_mask & (hull_layer[:, :, 3] > 0)
            edge = np.asarray(overlay)[:, :, 3] == 255
            hull_layer[off_floor & ~edge, 3] = 60
            image = Image.alpha_composite(image.convert("RGBA"), Image.fromarray(hull_layer))
    return image.convert("RGB"), ratio


def montage(panels: list[tuple[int, Image.Image, float]], columns: int = 5) -> Image.Image:
    width, height = panels[0][1].size
    label_band = 26
    rows = int(np.ceil(len(panels) / columns))
    sheet = Image.new("RGB", (columns * width, rows * (height + label_band)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (frame_index, image, ratio) in enumerate(panels):
        x, y = (i % columns) * width, (i // columns) * (height + label_band)
        label = f"Frame {frame_index}   Floor: {ratio:.2f} | Wall: {1 - ratio:.2f}"
        draw.text((x + 6, y + 6), label, fill="black")
        sheet.paste(image, (x, y + label_band))
    return sheet


def pick_wall_moments(ratios: np.ndarray, count: int, min_gap: int = 25) -> list[int]:
    """Frames with the lowest floor ratio, kept at least min_gap frames apart."""
    order = np.argsort(ratios)
    picked: list[int] = []
    for index in order:
        if np.isnan(ratios[index]):
            continue
        if all(abs(index - p) >= min_gap for p in picked):
            picked.append(int(index))
        if len(picked) == count:
            break
    return sorted(picked)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--keypoints", type=Path, help="DeepLabCut CSV for the same video")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--likelihood", type=float, default=0.9)
    parser.add_argument("--panels", type=int, default=25, help="Montage panel count")
    parser.add_argument(
        "--frames",
        help="Comma-separated frame indices to render; default: strongest wall-overlap moments",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"background: median over sampled frames of {args.video.name}")
    background = median_background(args.video)
    quad, floor_mask = detect_floor(background)
    (args.out_dir / "floor_corners.json").write_text(
        json.dumps({"video": args.video.name, "corners": np.round(quad, 2).tolist()}, indent=2)
    )

    keypoints = None
    if args.keypoints is not None:
        bodyparts, keypoints = load_dlc_keypoints(args.keypoints)
        print(f"keypoints: {len(keypoints)} frames x {len(bodyparts)} bodyparts")
        ratios = all_frame_ratios(keypoints, floor_mask, args.likelihood)
        with (args.out_dir / "floor_ratios.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame", "floor_ratio", "wall_ratio"])
            for index, ratio in enumerate(ratios):
                if not np.isnan(ratio):
                    writer.writerow([index, f"{ratio:.4f}", f"{1 - ratio:.4f}"])
        wanted = (
            [int(v) for v in args.frames.split(",")]
            if args.frames
            else pick_wall_moments(ratios, args.panels)
        )
    else:
        wanted = [int(v) for v in args.frames.split(",")] if args.frames else [0]

    panels = []
    for index, frame in enumerate(stream_frames(args.video)):
        if index in wanted:
            pose = keypoints[index] if keypoints is not None else None
            image, ratio = render_frame(frame, floor_mask, pose, args.likelihood)
            panels.append((index, image, ratio))
    sheet = montage(panels)
    sheet.save(args.out_dir / "geometry_montage.png")
    print(f"wrote {args.out_dir / 'geometry_montage.png'} with {len(panels)} panels")


if __name__ == "__main__":
    main()
