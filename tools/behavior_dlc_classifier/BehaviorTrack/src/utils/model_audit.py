"""Model, feature-schema, and input-domain checks for BehaviorTrack.

These checks separate three claims that were previously blurred together:

* the file is the intended legacy classifier;
* the 719 columns have the schema and coordinate meaning used during training;
* a new recording resembles the training domain.

Passing the first two makes inference reproducible.  It does not manufacture
target-domain validation, which requires labelled animals from the target rig.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LEGACY_MODEL_SHA256 = "dcc88056b0c87ff746061291daf5e1e5629dee252fbf2b71dcefb55490918782"
LEGACY_ENCODER_SHA256 = "0ef0951ad58ace99217c8186f564f1e910528794dd77e55f96ea8dfcece39bb6"
FEATURE_SCHEMA_VERSION = "legacy-moseq-dlc-719-training-center-v1"
EXPECTED_FEATURE_COUNT = 719
EXPECTED_CLASSES = (
    "Climbing",
    "Freezing",
    "Grooming",
    "Jump",
    "Locomotion",
    "Sniffing",
    "Turn",
    "Unspecified",
)

# Central 90% of nose-to-tail distances in the 2,638 archived training rows.
# The full range contains obvious tracking failures and is not useful for domain
# checks.  This is a warning band, not a normalization target.
TRAINING_BODY_LENGTH_MEDIAN_PX = 243.19805908
TRAINING_BODY_LENGTH_P05_PX = 161.59365540
TRAINING_BODY_LENGTH_P95_PX = 333.28683929


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelAudit:
    model_path: str
    model_sha256: str
    encoder_path: str
    encoder_sha256: str
    feature_schema_version: str
    feature_count: int
    classes: tuple[str, ...]
    intended_model: bool
    encoder_matches: bool
    schema_matches: bool
    archive_matches: bool | None
    source_validation: str
    target_validation: str
    warnings: tuple[str, ...]

    @property
    def contract_ok(self) -> bool:
        return (
            self.intended_model
            and self.encoder_matches
            and self.schema_matches
            and self.archive_matches is not False
        )

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["contract_ok"] = self.contract_ok
        return out


@dataclass(frozen=True)
class DomainAudit:
    frames: int
    median_nose_tail_px: float
    training_median_nose_tail_px: float
    scale_ratio_to_training: float
    inside_training_central_90_percent: bool
    median_tracked_bodyparts: float
    percent_frames_with_at_least_8_bodyparts: float
    warning: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _find_archive(model_path: Path) -> Path | None:
    for parent in [model_path.resolve(), *model_path.resolve().parents]:
        candidate = parent / "classifier" / "legacy_shap" / "shap_values.npz"
        if candidate.is_file():
            return candidate
    return None


def audit_model(model_path: Path, encoder_path: Path, loaded_model: Any) -> ModelAudit:
    model_path = Path(model_path).resolve()
    encoder_path = Path(encoder_path).resolve()
    model_hash = sha256_file(model_path)
    encoder_hash = sha256_file(encoder_path)
    names = tuple(str(name) for name in loaded_model.feature_names)
    classes = tuple(str(name) for name in loaded_model.classes)
    warnings: list[str] = []

    intended = model_hash.lower() == LEGACY_MODEL_SHA256
    if not intended:
        warnings.append("The selected model is not the archived MoSeq/DLC 719-feature artifact.")

    schema_matches = len(names) == EXPECTED_FEATURE_COUNT and classes == EXPECTED_CLASSES
    if not schema_matches:
        warnings.append(
            f"Model schema differs from the expected 719 features / 8 encoded classes "
            f"({len(names)} features; classes={classes})."
        )

    archive_matches: bool | None = None
    archive = _find_archive(model_path)
    if archive is not None:
        with np.load(archive, allow_pickle=True) as stored:
            archive_names = tuple(str(name) for name in stored["feature_names"])
            archive_classes = tuple(str(name) for name in stored["class_names"])
        archive_matches = names == archive_names and classes == archive_classes
        if not archive_matches:
            warnings.append("Model feature/class schema does not match the archived SHAP inputs.")

    encoder_matches = encoder_hash.lower() == LEGACY_ENCODER_SHA256
    if not encoder_matches:
        warnings.append("The selected label encoder is not the archived encoder paired with this model.")

    warnings.append(
        "Historical 0.627 accuracy used pooled frame folds; it is not animal/session-held-out validation."
    )
    warnings.append("No labelled target-rig hold-out is bundled; target-domain performance is unknown.")
    return ModelAudit(
        model_path=str(model_path),
        model_sha256=model_hash,
        encoder_path=str(encoder_path),
        encoder_sha256=encoder_hash,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        feature_count=len(names),
        classes=classes,
        intended_model=intended,
        encoder_matches=encoder_matches,
        schema_matches=schema_matches,
        archive_matches=archive_matches,
        source_validation="pooled-frame CV accuracy 0.627; animal/session leakage risk",
        target_validation="not available",
        warnings=tuple(warnings),
    )


def audit_input_domain(df_flat: pd.DataFrame, bodyparts: list[str]) -> DomainAudit:
    nose_x = pd.to_numeric(df_flat["nose_x"], errors="coerce").to_numpy(dtype=float)
    nose_y = pd.to_numeric(df_flat["nose_y"], errors="coerce").to_numpy(dtype=float)
    tail_x = pd.to_numeric(df_flat["tail_x"], errors="coerce").to_numpy(dtype=float)
    tail_y = pd.to_numeric(df_flat["tail_y"], errors="coerce").to_numpy(dtype=float)
    length = np.hypot(nose_x - tail_x, nose_y - tail_y)
    finite_length = length[np.isfinite(length)]
    median_length = float(np.median(finite_length)) if finite_length.size else float("nan")
    ratio = median_length / TRAINING_BODY_LENGTH_MEDIAN_PX
    inside = bool(
        np.isfinite(median_length)
        and TRAINING_BODY_LENGTH_P05_PX <= median_length <= TRAINING_BODY_LENGTH_P95_PX
    )

    likelihood_cols = [f"{bp}_likelihood" for bp in bodyparts if f"{bp}_likelihood" in df_flat]
    if likelihood_cols:
        likelihood = df_flat[likelihood_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
        tracked = np.sum(likelihood >= 0.6, axis=1)
    else:
        coordinate_ok = []
        for bp in bodyparts:
            coordinate_ok.append(
                np.isfinite(pd.to_numeric(df_flat[f"{bp}_x"], errors="coerce"))
                & np.isfinite(pd.to_numeric(df_flat[f"{bp}_y"], errors="coerce"))
            )
        tracked = np.sum(np.column_stack(coordinate_ok), axis=1)

    warning = None
    if not inside:
        warning = (
            f"Median nose-tail length is {median_length:.1f}px versus {TRAINING_BODY_LENGTH_MEDIAN_PX:.1f}px "
            "in training. The model uses raw pixel distances/velocities, so this recording is outside "
            "the central training scale; predictions are not validated for this camera scale."
        )
    return DomainAudit(
        frames=len(df_flat),
        median_nose_tail_px=median_length,
        training_median_nose_tail_px=TRAINING_BODY_LENGTH_MEDIAN_PX,
        scale_ratio_to_training=float(ratio),
        inside_training_central_90_percent=inside,
        median_tracked_bodyparts=float(np.median(tracked)) if len(tracked) else 0.0,
        percent_frames_with_at_least_8_bodyparts=(100.0 * float(np.mean(tracked >= 8))) if len(tracked) else 0.0,
        warning=warning,
    )


def write_provenance(path: Path, model: ModelAudit, domain: DomainAudit) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_contract": model.to_dict(),
        "input_domain": domain.to_dict(),
        "classification_policy": {
            "coordinate_transform": "per-frame centroid subtraction only",
            "behavior_source": "single 719-feature XGBoost argmax for all seven behaviors",
            "behavior_heuristics": "none",
            "separate_freezing_override": False,
            "tracking_failure_policy": "Unassigned",
            "canonical_smoothing": "none",
        },
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
