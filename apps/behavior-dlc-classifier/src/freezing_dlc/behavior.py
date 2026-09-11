from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import warnings

import joblib
import numpy as np
import pandas as pd

from .draw import blend_box as _blend_box


_REPO_ROOT = Path(__file__).resolve().parents[2]
BEHAVIOR_MODEL_PATH = _REPO_ROOT / "models" / "behavior" / "xgb_model.pkl"
BEHAVIOR_LABEL_ENCODER_PATH = _REPO_ROOT / "models" / "behavior" / "label_encoder.pkl"
#: Portable copy of the same classifier in the XGBoost JSON format. Pickles only
#: reload under a compatible XGBoost version, whereas this format is readable
#: across versions, so it is used automatically when the pickle cannot be loaded.
BEHAVIOR_MODEL_JSON_PATH = _REPO_ROOT / "models" / "behavior" / "xgb_model.json"
BEHAVIOR_ORDER = ["Freezing", "Sniffing", "Grooming", "Turn", "Locomotion", "Climbing", "Jump"]
BEHAVIOR_DISPLAY_ORDER = ["Jump", "Climbing", "Locomotion", "Turn", "Grooming", "Sniffing", "Freezing"]
#: The eighth class. It covers MoSeq syllables outside the seven hand-curated
#: clusters during training, and frames whose tracking is too poor to judge at
#: inference. The paper repository calls this class ``Unassigned``; the bundled
#: label encoder was pickled with the older name, which is translated away in
#: :func:`predict_behaviors` so that nothing downstream sees it.
UNASSIGNED = "Unassigned"
_ENCODER_UNASSIGNED = "Unspecified"
BEHAVIOR_COLORS = {
    "Freezing": "#d47aae",
    "Sniffing": "#6f9ead",
    "Grooming": "#a8d7e8",
    "Turn": "#b9dc43",
    "Locomotion": "#e2ba19",
    "Climbing": "#e88427",
    "Jump": "#e45d63",
    UNASSIGNED: "#b8b8b8",
}
BEHAVIOR_BODYPARTS = [
    "B1L",
    "B1R",
    "B2L",
    "B2R",
    "B3L",
    "B3R",
    "H1L",
    "H1R",
    "H2L",
    "H2R",
    "S1",
    "S2",
    "nose",
    "tail",
]


@dataclass(frozen=True)
class LoadedBehaviorModel:
    classifier: object
    label_encoder: object
    classes: list[str]
    feature_names: list[str]


def _load_behavior_classifier(model_path: Path) -> tuple[object, list[str]]:
    """Load the behavior classifier, preferring the pickle and falling back to the
    version-portable JSON export next to it."""
    json_path = Path(model_path).with_suffix(".json")
    meta_path = Path(model_path).with_name(Path(model_path).stem + "_meta.json")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            classifier = joblib.load(model_path)
        feature_names = list(getattr(classifier, "feature_names_in_", []))
        if feature_names:
            return classifier, feature_names
    except Exception:
        classifier = None

    if not json_path.exists():
        if classifier is None:
            raise FileNotFoundError(
                f"Could not load the behavior model from {model_path}, and no portable "
                f"copy was found at {json_path}."
            )
        raise ValueError(
            f"The behavior model at {model_path} does not carry feature names, and no "
            f"portable copy was found at {json_path}."
        )

    from xgboost import XGBClassifier

    portable = XGBClassifier()
    portable.load_model(str(json_path))
    feature_names = list(getattr(portable, "feature_names_in_", []))
    if not feature_names and meta_path.exists():
        feature_names = list(json.loads(meta_path.read_text(encoding="utf-8")).get("feature_names", []))
    if not feature_names:
        raise ValueError(f"Could not determine the behavior model feature names from {json_path}.")
    return portable, feature_names


def load_behavior_model(model_path: Path, label_encoder_path: Path) -> LoadedBehaviorModel:
    classifier, feature_names = _load_behavior_classifier(Path(model_path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        label_encoder = joblib.load(label_encoder_path)
    classes = [str(c) for c in getattr(label_encoder, "classes_", [])]
    return LoadedBehaviorModel(classifier, label_encoder, classes, feature_names)


#: Median nose-to-tail distance in the archived training-feature sample.  This is
#: a domain-QC reference, not a licence to silently rescale a new recording.  The
#: model was trained on raw pixel scale and any scale transfer must be validated
#: on labelled target recordings.
TRAINING_BODY_LENGTH_PX = 243.63


def training_coordinate_frame(df_flat: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the coordinate transform present in the archived training rows.

    The 2,638 original feature rows stored in
    ``classifier/legacy_shap/shap_values.npz`` provide a numerical record of the
    model input.  In every row (and in each of its five lags), the mean x and y
    coordinate across the 14 markers is zero.  Nose-to-tail orientation remains
    unconstrained and nose-to-tail length remains in pixels.  Training therefore
    centred each frame, but did not rotate or scale it.

    Keeping this transform small is important: orientation and raw pixel
    distances are themselves model features.  Rotating or normalising them after
    training changes the meaning of the existing model rather than making it
    invariant.
    """
    coords = [
        bp
        for bp in BEHAVIOR_BODYPARTS
        if f"{bp}_x" in df_flat.columns and f"{bp}_y" in df_flat.columns
    ]
    if not coords:
        return df_flat.copy()

    x = np.column_stack(
        [pd.to_numeric(df_flat[f"{bp}_x"], errors="coerce") for bp in coords]
    ).astype(float)
    y = np.column_stack(
        [pd.to_numeric(df_flat[f"{bp}_y"], errors="coerce") for bp in coords]
    ).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centroid_x = np.nanmean(x, axis=1)
        centroid_y = np.nanmean(y, axis=1)

    out = df_flat.copy()
    for i, bp in enumerate(coords):
        out[f"{bp}_x"] = x[:, i] - centroid_x
        out[f"{bp}_y"] = y[:, i] - centroid_y
    return out


def egocentric_align(df_flat: pd.DataFrame) -> pd.DataFrame:
    """Experimental rotation/scale transfer transform.

    Each frame is centred on the animal, rotated so the tail-to-nose axis lies on
    +x, and scaled to the median training nose-to-tail length.  This can be useful
    when developing a *new model trained with the same transform*, but it does not
    reproduce the bundled model's archived inputs.  It is retained only for
    explicit transfer experiments; normal inference uses
    :func:`training_coordinate_frame`.
    """
    coords = [bp for bp in BEHAVIOR_BODYPARTS if f"{bp}_x" in df_flat.columns and f"{bp}_y" in df_flat.columns]
    if not {"nose", "tail"}.issubset(coords):
        return df_flat

    x = np.column_stack([pd.to_numeric(df_flat[f"{bp}_x"], errors="coerce") for bp in coords]).astype(float)
    y = np.column_stack([pd.to_numeric(df_flat[f"{bp}_y"], errors="coerce") for bp in coords]).astype(float)
    nose, tail = coords.index("nose"), coords.index("tail")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN frames are expected
        centroid_x = np.nanmean(x, axis=1)
        centroid_y = np.nanmean(y, axis=1)
    angle = np.arctan2(y[:, nose] - y[:, tail], x[:, nose] - x[:, tail])
    cos, sin = np.cos(angle), np.sin(angle)

    centred_x = x - centroid_x[:, None]
    centred_y = y - centroid_y[:, None]
    # Rotating by -angle puts the tail-to-nose vector on the positive x axis.
    rotated_x = centred_x * cos[:, None] + centred_y * sin[:, None]
    rotated_y = -centred_x * sin[:, None] + centred_y * cos[:, None]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        body_length = np.nanmedian(np.hypot(x[:, nose] - x[:, tail], y[:, nose] - y[:, tail]))
    scale = TRAINING_BODY_LENGTH_PX / body_length if np.isfinite(body_length) and body_length > 0 else 1.0

    oriented = np.isfinite(angle)
    out = df_flat.copy()
    for i, bp in enumerate(coords):
        out[f"{bp}_x"] = np.where(oriented, rotated_x[:, i] * scale, np.nan)
        out[f"{bp}_y"] = np.where(oriented, rotated_y[:, i] * scale, np.nan)
    return out


def build_behavior_feature_set(
    df_flat: pd.DataFrame,
    feature_names: Iterable[str] | None = None,
    fps: float = 25.0,
    window: int = 5,
    *,
    coordinate_mode: str = "training_center",
) -> pd.DataFrame:
    """Generate the 719-frame behavior feature set used by the SHAP/XGBoost model.

    ``training_center`` is the only mode that reproduces the archived model
    inputs. ``experimental_egocentric_scaled`` is available for controlled
    transfer experiments and must not be presented as validated inference with
    the bundled model.
    """
    missing = [bp for bp in BEHAVIOR_BODYPARTS if f"{bp}_x" not in df_flat.columns or f"{bp}_y" not in df_flat.columns]
    if missing:
        raise ValueError(
            "This tracking file does not use the keypoint set the behavior model was trained on. "
            f"Missing body parts: {', '.join(missing)}. "
            f"Expected all of: {', '.join(BEHAVIOR_BODYPARTS)}."
        )

    if coordinate_mode == "training_center":
        df_flat = training_coordinate_frame(df_flat)
    elif coordinate_mode == "experimental_egocentric_scaled":
        df_flat = egocentric_align(df_flat)
    else:
        raise ValueError(
            "coordinate_mode must be 'training_center' or "
            "'experimental_egocentric_scaled'."
        )
    index = df_flat.index
    data: dict[str, np.ndarray] = {}

    for bp in BEHAVIOR_BODYPARTS:
        for axis in ("x", "y"):
            col = f"{bp}_{axis}"
            values = pd.to_numeric(df_flat[col], errors="coerce").astype(float)
            data[col] = values.interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype=np.float32)

    coords = {bp: np.column_stack([data[f"{bp}_x"], data[f"{bp}_y"]]).astype(np.float32) for bp in BEHAVIOR_BODYPARTS}
    pts = np.stack([coords[bp] for bp in BEHAVIOR_BODYPARTS], axis=1)
    centroid = np.nanmean(pts, axis=1)
    data["centroid_x"] = centroid[:, 0].astype(np.float32)
    data["centroid_y"] = centroid[:, 1].astype(np.float32)

    nose_vec = coords["nose"] - coords["tail"]
    angles = np.arctan2(nose_vec[:, 1], nose_vec[:, 0])
    data["orientation_angle"] = angles.astype(np.float32)

    ang_diff = np.diff(angles)
    ang_diff = np.mod(ang_diff + np.pi, 2 * np.pi) - np.pi
    angular_velocity = np.concatenate([ang_diff * float(fps), [np.nan]])
    data["angular_velocity"] = pd.Series(angular_velocity, index=index).rolling(3, center=True).mean().to_numpy(dtype=np.float32)

    global_orientation = _compute_body_axis_rotation(pts)
    global_ang_vel = np.diff(global_orientation)
    global_ang_vel = np.mod(global_ang_vel + np.pi, 2 * np.pi) - np.pi
    global_ang_vel = np.append(global_ang_vel * float(fps), np.nan)
    data["global_body_angle"] = global_orientation.astype(np.float32)
    data["global_body_angular_velocity"] = pd.Series(global_ang_vel, index=index).rolling(3, center=True).mean().to_numpy(dtype=np.float32)

    angle_diffs = angles - pd.Series(angles, index=index).shift(window).to_numpy()
    angle_diffs = np.mod(angle_diffs + np.pi, 2 * np.pi) - np.pi
    data[f"orientation_change_{window}"] = np.asarray(angle_diffs, dtype=np.float32)

    displacement = np.linalg.norm(centroid - pd.DataFrame(centroid, index=index).shift(window).to_numpy(), axis=1)
    data[f"centroid_displacement_{window}"] = np.asarray(displacement, dtype=np.float32)

    dist_cols: list[str] = []
    for i, bp1 in enumerate(BEHAVIOR_BODYPARTS):
        for bp2 in BEHAVIOR_BODYPARTS[i + 1 :]:
            col = f"dist_{bp1}_{bp2}"
            dist_cols.append(col)
            data[col] = np.linalg.norm(coords[bp1] - coords[bp2], axis=1).astype(np.float32)

    for bp in BEHAVIOR_BODYPARTS:
        velocity = np.linalg.norm(np.diff(coords[bp], axis=0), axis=1) * float(fps)
        data[f"{bp}_velocity"] = np.concatenate([velocity, [np.nan]]).astype(np.float32)
    centroid_velocity = np.linalg.norm(np.diff(centroid, axis=0), axis=1) * float(fps)
    data["centroid_velocity"] = np.concatenate([centroid_velocity, [np.nan]]).astype(np.float32)

    x_base = pd.DataFrame(data, index=index)
    lag_data: dict[str, np.ndarray] = {}
    for lag in range(1, window + 1):
        for bp in BEHAVIOR_BODYPARTS:
            for axis in ("x", "y"):
                col = f"{bp}_{axis}"
                lag_data[f"{col}_t-{lag}"] = x_base[col].shift(lag).to_numpy(dtype=np.float32)
        for feature in ["orientation_angle", "angular_velocity", "global_body_angle", "global_body_angular_velocity"]:
            lag_data[f"{feature}_t-{lag}"] = x_base[feature].shift(lag).to_numpy(dtype=np.float32)

    x_new = pd.concat([x_base, pd.DataFrame(lag_data, index=index)], axis=1)

    base_features = (
        [f"{bp}_{axis}" for bp in BEHAVIOR_BODYPARTS for axis in ("x", "y")]
        + [f"{bp}_velocity" for bp in BEHAVIOR_BODYPARTS]
        + [
            "centroid_x",
            "centroid_y",
            "centroid_velocity",
            "angular_velocity",
            "global_body_angle",
            "global_body_angular_velocity",
        ]
        + dist_cols
    )
    roll_stats = x_new[base_features].rolling(window, min_periods=1).agg(["mean", "std", "sum"])
    roll_stats.columns = [f"{col[0]}_roll_{col[1]}" for col in roll_stats.columns]
    x_new = pd.concat([x_new, roll_stats.astype(np.float32)], axis=1)

    x_new = x_new.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    if feature_names is not None:
        x_new = x_new.reindex(columns=list(feature_names), fill_value=0.0)
    return x_new


def _display_class(name: object) -> str:
    """Translate the label encoder's class names into the ones the paper uses."""
    name = str(name)
    return UNASSIGNED if name == _ENCODER_UNASSIGNED else name


def predict_behaviors(features: pd.DataFrame, model: LoadedBehaviorModel) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    aligned = features.reindex(columns=model.feature_names, fill_value=0.0)
    probs = model.classifier.predict_proba(aligned)
    pred_codes = np.asarray(np.argmax(probs, axis=1), dtype=int)
    decoded = model.label_encoder.inverse_transform(pred_codes)
    labels = np.asarray([_display_class(name) for name in decoded], dtype=object)
    prob_df = pd.DataFrame(probs, columns=[f"prob_{_display_class(c)}" for c in model.classes], index=features.index)
    return labels, pred_codes, prob_df


def confidence_for_labels(labels: np.ndarray, prob_df: pd.DataFrame) -> np.ndarray:
    """Percent probability the model gave to the behavior that is actually reported.

    The model's highest probability often belongs to a class other than the one
    finally reported, so reading it off ``prob_df.max()`` would print a number
    about a different behavior.
    """
    n = min(len(labels), len(prob_df))
    probs = prob_df.iloc[:n].reset_index(drop=True)
    out = np.full(n, np.nan, dtype=float)
    for name in pd.unique(np.asarray(labels[:n], dtype=str)):
        col = f"prob_{name}"
        if col in probs.columns:
            rows = np.asarray(labels[:n], dtype=str) == name
            out[rows] = probs.loc[rows, col].to_numpy(dtype=float) * 100.0
    return out


def refine_behavior_labels(
    raw_labels: np.ndarray,
    prob_df: pd.DataFrame,
    df_flat: pd.DataFrame,
    freezing_pred: np.ndarray | None = None,
    freezing_prob: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Apply the two overrides that outrank the behavior model's own prediction.

    The model's label stands on its own everywhere except two cases: the
    separately validated freezing classifier decides Freezing, and frames whose
    tracking is too poor to judge become `Unassigned`. Nothing else is
    second-guessed, so a behavior is only ever reported because the model
    predicted it.
    """
    n = min(len(raw_labels), len(prob_df), len(df_flat))
    raw = np.asarray(raw_labels[:n], dtype=str)
    probs = prob_df.iloc[:n].reset_index(drop=True)
    metrics = _behavior_rule_metrics(df_flat.iloc[:n])

    final = raw.copy()
    source = np.full(n, "xgboost", dtype=object)
    good_tracking = metrics["tracking_ok"].to_numpy(dtype=bool)
    freeze_model = np.zeros(n, dtype=bool)
    if freezing_pred is not None:
        freeze_model = np.asarray(freezing_pred[:n], dtype=int) == 1

    final[freeze_model] = "Freezing"
    source[freeze_model] = "freezing_model"

    tracking_bad = ~freeze_model & ~good_tracking
    final[tracking_bad] = UNASSIGNED
    source[tracking_bad] = "tracking_bad"

    confidence = confidence_for_labels(final, probs)
    if freezing_prob is not None:
        fp = np.asarray(freezing_prob[:n], dtype=float)
        metrics["freezing_model_probability"] = fp
        confidence[freeze_model] = fp[freeze_model] * 100.0
    else:
        confidence[freeze_model] = np.nan
    confidence[tracking_bad] = np.nan
    metrics["confidence"] = confidence
    metrics["raw_behavior"] = raw
    metrics["label_source"] = source
    return final, source, metrics


def save_behavior_predictions(
    out_csv: Path,
    labels: np.ndarray,
    pred_codes: np.ndarray,
    prob_df: pd.DataFrame,
    raw_labels: np.ndarray | None = None,
    label_source: np.ndarray | None = None,
    metrics_df: pd.DataFrame | None = None,
    confidence: np.ndarray | None = None,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if confidence is None:
        confidence = confidence_for_labels(labels, prob_df)
    confidence = np.asarray(confidence, dtype=float)
    out = pd.DataFrame(
        {
            "frame": np.arange(len(labels), dtype=int),
            "behavior": labels,
            "raw_behavior": raw_labels if raw_labels is not None else labels,
            "label_source": label_source if label_source is not None else "xgboost",
            "raw_behavior_code": pred_codes,
            "confidence": confidence,
        }
    )
    out = pd.concat([out, prob_df.reset_index(drop=True)], axis=1)
    if metrics_df is not None:
        extra = metrics_df.reset_index(drop=True)
        extra = extra[[c for c in extra.columns if c not in out.columns]]
        out = pd.concat([out, extra], axis=1)
    out.to_csv(out_csv, index=False)


def annotate_behavior_video(
    video_in: Path,
    video_out: Path,
    labels: np.ndarray,
    prob_df: pd.DataFrame,
    fps: int,
    metrics_df: pd.DataFrame | None = None,
    confidence: np.ndarray | None = None,
) -> None:
    import cv2

    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_in}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_height = max(height, 520)
    scale = output_height / max(float(height), 1.0)
    display_width = max(1, int(round(width * scale)))
    display_height = output_height
    panel_width = max(380, min(520, int(display_width * 0.55)))
    video_out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (display_width + panel_width, output_height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video for writing: {video_out}")

    n = min(len(labels), len(prob_df))
    if confidence is None:
        confidence = confidence_for_labels(labels, prob_df)
    confidence = np.asarray(confidence, dtype=float)
    safe_fps = max(float(fps), 1.0)
    labels_trimmed = np.asarray(labels[:n], dtype=str) if n else np.asarray([], dtype=str)
    cumulative_seconds = {
        name: np.cumsum(labels_trimmed == name).astype(np.float32) / safe_fps for name in BEHAVIOR_DISPLAY_ORDER
    }
    total_seconds = {name: float(np.sum(labels_trimmed == name) / safe_fps) for name in BEHAVIOR_DISPLAY_ORDER}
    # Plain arrays for the per-frame loop; pandas row lookups in here cost
    # seconds per video.
    prob_columns = {
        name: prob_df[f"prob_{name}"].to_numpy(dtype=float)
        for name in BEHAVIOR_DISPLAY_ORDER
        if f"prob_{name}" in prob_df.columns
    }
    sources = None
    if metrics_df is not None and "label_source" in metrics_df.columns:
        sources = metrics_df["label_source"].astype(str).to_numpy()
    try:
        _render_behavior_frames(
            cv2, cap, writer, labels, confidence, n, prob_columns, sources,
            cumulative_seconds, total_seconds,
            width, height, display_width, display_height, output_height, panel_width,
        )
    finally:
        cap.release()
        writer.release()


def _render_behavior_frames(
    cv2, cap, writer, labels, confidence, n, prob_columns, sources,
    cumulative_seconds, total_seconds,
    width, height, display_width, display_height, output_height, panel_width,
) -> None:
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if display_width != width or display_height != height:
            frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((output_height, display_width + panel_width, 3), dtype=np.uint8)
        canvas[:, :panel_width] = (18, 18, 24)
        canvas[:, panel_width:] = frame
        cv2.line(canvas, (panel_width - 1, 0), (panel_width - 1, output_height), (90, 90, 105), 1)

        idx = min(frame_idx, max(0, n - 1))
        behavior = str(labels[idx]) if n else UNASSIGNED
        conf = float(confidence[idx]) if idx < len(confidence) else float("nan")
        color = _hex_to_bgr(BEHAVIOR_COLORS.get(behavior, BEHAVIOR_COLORS[UNASSIGNED]))

        header_bottom = 138 if sources is not None else 116
        _blend_box(canvas, (12, 12), (panel_width - 12, header_bottom), (28, 28, 36), 0.9)
        cv2.rectangle(canvas, (12, 12), (panel_width - 12, header_bottom), (90, 90, 105), 1)
        _blend_box(canvas, (24, 24), (panel_width - 24, 60), color, 0.95)
        _put_text_fit(
            canvas,
            behavior.upper(),
            (34, 50),
            panel_width - 68,
            0.62,
            (255, 255, 255),
            thickness=1,
        )
        conf_text = "confidence n/a" if not np.isfinite(conf) else f"{conf:5.1f}% confidence"
        cv2.putText(canvas, conf_text, (24, 88), cv2.FONT_HERSHEY_DUPLEX, 0.38, (224, 224, 232), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"frame {frame_idx + 1:,}", (24, 111), cv2.FONT_HERSHEY_DUPLEX, 0.32, (185, 190, 205), 1, cv2.LINE_AA)
        if sources is not None and n and idx < len(sources):
            source = _short_source_label(str(sources[idx]))
            _put_text_fit(
                canvas,
                f"decided by {source}",
                (24, 123),
                panel_width - 48,
                0.30,
                (185, 190, 205),
            )

        row_top = header_bottom + 18
        row_gap = max(32, int((output_height - row_top - 14) / len(BEHAVIOR_DISPLAY_ORDER)))
        bar_x1 = max(116, int(panel_width * 0.34))
        bar_x2 = panel_width - 18
        bar_h = max(9, min(15, row_gap // 3))
        name_font = max(0.28, min(0.36, row_gap / 120.0))
        small_font = max(0.25, min(0.30, row_gap / 145.0))
        for row, name in enumerate(BEHAVIOR_DISPLAY_ORDER):
            y = row_top + row * row_gap
            prob = float(prob_columns[name][idx]) if n and name in prob_columns else 0.0
            running_sec = float(cumulative_seconds[name][idx]) if n else 0.0
            row_color = _hex_to_bgr(BEHAVIOR_COLORS[name])
            cv2.putText(canvas, name, (18, y + int(row_gap * 0.45)), cv2.FONT_HERSHEY_DUPLEX, name_font, (220, 222, 230), 1, cv2.LINE_AA)
            bar_y1 = y + 6
            bar_y2 = y + 6 + bar_h
            cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (70, 70, 84), 1)
            fill_x = bar_x1 + int((bar_x2 - bar_x1) * np.clip(prob, 0.0, 1.0))
            if fill_x > bar_x1:
                cv2.rectangle(canvas, (bar_x1, bar_y1), (fill_x, bar_y2), row_color, -1)
            info_y = min(y + row_gap - 6, output_height - 6)
            cv2.putText(canvas, f"{prob * 100:4.0f}%", (bar_x1, info_y), cv2.FONT_HERSHEY_DUPLEX, small_font, (178, 184, 198), 1, cv2.LINE_AA)
            counter_x = max(bar_x1 + 48, bar_x2 - 92)
            cv2.putText(
                canvas,
                f"{running_sec:4.1f}/{total_seconds[name]:4.1f}s",
                (counter_x, info_y),
                cv2.FONT_HERSHEY_DUPLEX,
                small_font,
                (178, 184, 198),
                1,
                cv2.LINE_AA,
            )

        writer.write(canvas)
        frame_idx += 1


def _behavior_rule_metrics(df_flat: pd.DataFrame) -> pd.DataFrame:
    """Per-frame tracking quality: how much of the animal was actually seen."""
    bps = [c[: -len("_x")] for c in df_flat.columns if c.endswith("_x") and f"{c[:-2]}_y" in df_flat.columns]
    n = len(df_flat)
    if not bps:
        return pd.DataFrame(
            {
                "tracking_ok": np.zeros(n, dtype=bool),
                "likelihood_mean": np.zeros(n, dtype=np.float32),
                "valid_keypoints": np.zeros(n, dtype=np.int16),
            }
        )

    x = np.vstack([pd.to_numeric(df_flat[f"{bp}_x"], errors="coerce").to_numpy(dtype=float) for bp in bps]).T
    y = np.vstack([pd.to_numeric(df_flat[f"{bp}_y"], errors="coerce").to_numpy(dtype=float) for bp in bps]).T
    lik_cols = [f"{bp}_likelihood" for bp in bps]
    if all(col in df_flat.columns for col in lik_cols):
        likelihood = np.vstack([pd.to_numeric(df_flat[col], errors="coerce").to_numpy(dtype=float) for col in lik_cols]).T
    else:
        likelihood = np.ones_like(x, dtype=float)

    valid_count = (np.isfinite(x) & np.isfinite(y) & (likelihood >= 0.60)).sum(axis=1)
    likelihood_mean = np.nanmean(likelihood, axis=1)
    tracking_ok = (likelihood_mean >= 0.38) & (valid_count >= max(5, min(8, len(bps) // 2)))

    return pd.DataFrame(
        {
            "tracking_ok": tracking_ok,
            "likelihood_mean": likelihood_mean,
            "valid_keypoints": valid_count.astype(np.int16),
        }
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
    return b, g, r


def _put_text_fit(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    max_width: int,
    font_scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    import cv2

    scale = float(font_scale)
    while scale > 0.18:
        width = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)[0][0]
        if width <= max_width:
            break
        scale -= 0.03
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)


def _short_source_label(source: str) -> str:
    labels = {
        "xgboost": "the model",
        "freezing_model": "the freezing model",
        "tracking_bad": "poor tracking",
    }
    return labels.get(source, source[:18])


def _compute_body_axis_rotation(pts: np.ndarray) -> np.ndarray:
    orientations = np.full(pts.shape[0], np.nan, dtype=np.float32)
    for idx, frame in enumerate(pts):
        if np.isnan(frame).any():
            continue
        try:
            cov = np.cov(frame.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            principal_axis = eigvecs[:, np.argmax(eigvals)]
            orientations[idx] = np.arctan2(principal_axis[1], principal_axis[0])
        except Exception:
            continue
    return orientations
