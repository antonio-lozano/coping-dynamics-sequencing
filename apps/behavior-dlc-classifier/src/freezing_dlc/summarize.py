from pathlib import Path

import numpy as np
import pandas as pd


def summarize_predictions(pred: np.ndarray, video_name: str, fps: int, bin_sec: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = np.asarray(pred).astype(int)
    totals = pd.DataFrame([{"video": video_name, "freezing_pct": float(100 * pred.mean()) if pred.size else 0.0}])

    frames_per_bin = int(fps) * int(bin_sec)
    if frames_per_bin <= 0:
        raise ValueError("fps and bin_sec must be > 0")

    rows = []
    for i in range(0, len(pred), frames_per_bin):
        p = pred[i:i + frames_per_bin]
        rows.append({"video": video_name, "bin": i // frames_per_bin, "freezing_pct": float(100 * p.mean()) if p.size else 0.0})

    return totals, pd.DataFrame(rows)


def write_summary_excel(path_xlsx: str, totals: pd.DataFrame, bins: pd.DataFrame) -> None:
    Path(path_xlsx).parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path_xlsx, engine="openpyxl") as writer:
        bins.to_excel(writer, sheet_name="bins", index=False)
        totals.to_excel(writer, sheet_name="total", index=False)
