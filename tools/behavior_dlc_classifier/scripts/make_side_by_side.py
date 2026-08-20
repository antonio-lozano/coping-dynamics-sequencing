from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def _iter_common_files(left_dir: Path, right_dir: Path, suffix: str) -> list[str]:
    left = {p.name for p in left_dir.glob(f"*{suffix}") if p.is_file()}
    right = {p.name for p in right_dir.glob(f"*{suffix}") if p.is_file()}
    return sorted(left.intersection(right))


def _scaled_size(width: int, height: int, target_height: int) -> tuple[int, int]:
    if height <= 0:
        return width, target_height
    scale = float(target_height) / float(height)
    out_w = max(1, int(round(width * scale)))
    return out_w, target_height


def _draw_header(canvas, cv2, title: str, left_label: str, right_label: str, split_x: int, header_h: int) -> None:
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], header_h), (28, 28, 36), thickness=-1)
    cv2.line(canvas, (split_x, 0), (split_x, header_h), (88, 88, 102), 1)
    cv2.putText(canvas, title, (12, 22), cv2.FONT_HERSHEY_DUPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, left_label, (12, 44), cv2.FONT_HERSHEY_DUPLEX, 0.44, (180, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, right_label, (split_x + 12, 44), cv2.FONT_HERSHEY_DUPLEX, 0.44, (200, 255, 200), 1, cv2.LINE_AA)


def make_side_by_side(
    left_dir: Path,
    right_dir: Path,
    out_dir: Path,
    left_label: str,
    right_label: str,
    suffix: str,
    output_fps: float | None,
) -> tuple[int, int]:
    import cv2
    import numpy as np

    common = _iter_common_files(left_dir, right_dir, suffix)
    out_dir.mkdir(parents=True, exist_ok=True)

    made = 0
    skipped = 0
    for name in common:
        left_path = left_dir / name
        right_path = right_dir / name

        cap_left = cv2.VideoCapture(str(left_path))
        cap_right = cv2.VideoCapture(str(right_path))
        if not cap_left.isOpened() or not cap_right.isOpened():
            print(f"[skip] could not open pair: {name}")
            cap_left.release()
            cap_right.release()
            skipped += 1
            continue

        left_w = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
        left_h = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
        right_w = int(cap_right.get(cv2.CAP_PROP_FRAME_WIDTH))
        right_h = int(cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_left = float(cap_left.get(cv2.CAP_PROP_FPS) or 0.0)
        fps_right = float(cap_right.get(cv2.CAP_PROP_FPS) or 0.0)

        heights = [h for h in (left_h, right_h) if h > 0]
        if not heights:
            print(f"[skip] no readable frame size: {name}")
            cap_left.release()
            cap_right.release()
            skipped += 1
            continue
        target_h = min(heights)
        out_left_w, _ = _scaled_size(left_w, left_h, target_h)
        out_right_w, _ = _scaled_size(right_w, right_h, target_h)
        body_h = target_h
        header_h = 56
        out_w = out_left_w + out_right_w
        out_h = body_h + header_h

        if output_fps is not None and output_fps > 0:
            fps_out = float(output_fps)
        else:
            fps_candidates = [x for x in (fps_left, fps_right) if x > 1e-6]
            fps_out = min(fps_candidates) if fps_candidates else 25.0

        out_name = name.replace(suffix, "_side_by_side.mp4")
        writer = cv2.VideoWriter(
            str(out_dir / out_name),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_out,
            (out_w, out_h),
        )
        if not writer.isOpened():
            print(f"[skip] could not open output for writing: {out_name}")
            cap_left.release()
            cap_right.release()
            skipped += 1
            continue

        try:
            while True:
                ok_left, frame_left = cap_left.read()
                ok_right, frame_right = cap_right.read()
                if not ok_left or not ok_right:
                    break

                if frame_left.shape[0] != body_h or frame_left.shape[1] != out_left_w:
                    frame_left = cv2.resize(frame_left, (out_left_w, body_h), interpolation=cv2.INTER_AREA)
                if frame_right.shape[0] != body_h or frame_right.shape[1] != out_right_w:
                    frame_right = cv2.resize(frame_right, (out_right_w, body_h), interpolation=cv2.INTER_AREA)

                canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                canvas[header_h:, :out_left_w] = frame_left
                canvas[header_h:, out_left_w:] = frame_right
                _draw_header(canvas, cv2, name, left_label, right_label, out_left_w, header_h)

                writer.write(canvas)
        finally:
            writer.release()
            cap_left.release()
            cap_right.release()
        made += 1
        print(f"[ok] {out_name}")

    return made, skipped


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create side-by-side comparison videos from two annotated video folders.")
    p.add_argument("--left-dir", required=True, help="Directory with first set of annotated videos")
    p.add_argument("--right-dir", required=True, help="Directory with second set of annotated videos")
    p.add_argument("--out-dir", required=True, help="Directory for side-by-side outputs")
    p.add_argument("--left-label", default="Refined")
    p.add_argument("--right-label", default="Baseline")
    p.add_argument("--suffix", default="_annotated.mp4")
    p.add_argument("--fps", type=float, default=0.0, help="Optional fixed output fps")
    return p.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)

    left_dir = Path(args.left_dir)
    right_dir = Path(args.right_dir)
    out_dir = Path(args.out_dir)

    if not left_dir.exists():
        raise FileNotFoundError(f"Left dir not found: {left_dir}")
    if not right_dir.exists():
        raise FileNotFoundError(f"Right dir not found: {right_dir}")

    fps = float(args.fps)
    fps_arg = fps if fps > 0 else None
    made, skipped = make_side_by_side(
        left_dir=left_dir,
        right_dir=right_dir,
        out_dir=out_dir,
        left_label=str(args.left_label),
        right_label=str(args.right_label),
        suffix=str(args.suffix),
        output_fps=fps_arg,
    )
    print(f"[done] made={made} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
