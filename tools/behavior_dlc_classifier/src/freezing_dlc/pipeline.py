from pathlib import Path

import pandas as pd

from .annotate import annotate_video
from .config import PipelineConfig
from .dlc_stage import find_source_video
from .features import build_feature_set, dlc_filtered_csv_to_flat_df
from .model import load_model, resolve_threshold
from .predict import predict_freezing, save_frame_predictions
from .summarize import summarize_predictions, write_summary_excel


def run_pipeline(cfg: PipelineConfig, annotate: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg.pred_dir.mkdir(parents=True, exist_ok=True)
    cfg.features_dir.mkdir(parents=True, exist_ok=True)
    cfg.annotated_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(cfg.model_path)
    threshold = resolve_threshold(model, cfg.threshold, 0.88)
    print(f"[predict] Using freezing threshold {threshold:.2f}")
    totals_frames = []
    bins_frames = []

    # Sorted, so the summary workbook lists videos in the same order every run.
    for csv_path in sorted(cfg.filtered_dir.glob("*filtered.csv")):
        flat = dlc_filtered_csv_to_flat_df(csv_path)
        feat = build_feature_set(flat, mode=model.feature_mode)
        feat.to_csv(cfg.features_dir / f"{csv_path.stem}_features.csv", index=False)
        probs, pred = predict_freezing(feat, model, threshold)

        save_frame_predictions(cfg.pred_dir / f"{csv_path.stem}_pred.csv", probs, pred)

        totals, bins = summarize_predictions(pred, csv_path.stem, cfg.fps, cfg.bin_sec)
        totals_frames.append(totals)
        bins_frames.append(bins)

        if annotate:
            video_in = find_source_video(cfg.original_videos_dir, csv_path.stem)
            video_out = cfg.annotated_dir / f"{csv_path.stem}_annotated.mp4"
            annotate_video(video_in, video_out, probs, pred, cfg.fps)

    totals_df = pd.concat(totals_frames, ignore_index=True) if totals_frames else pd.DataFrame()
    bins_df = pd.concat(bins_frames, ignore_index=True) if bins_frames else pd.DataFrame()
    write_summary_excel(str(cfg.summary_xlsx), totals_df, bins_df)
    return totals_df, bins_df
