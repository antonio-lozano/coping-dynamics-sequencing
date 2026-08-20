from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The model predictions are made with: 4046 features, 700 trees, threshold 0.40.
#: About 1 GB, so it cannot be committed and ships as a release asset instead.
FULL_MODEL_PATH = REPO_ROOT / "models" / "freezing_model_full.sav"

#: The 61-feature model that is small enough to commit. It stands in when the
#: full model has not been downloaded, so a fresh clone still runs.
COMPACT_MODEL_PATH = REPO_ROOT / "models" / "freezing_model.sav"


def default_model_path(project_root: Optional[Path] = None) -> Path:
    """The freezing model to use when nobody asked for a particular one.

    The full model wins whenever it is present. Which one actually ran is worth
    knowing, so callers log the path rather than the word "default".
    """
    roots = [Path(project_root)] if project_root is not None else []
    roots.append(REPO_ROOT)
    for root in roots:
        candidate = root / "models" / FULL_MODEL_PATH.name
        if candidate.is_file():
            return candidate
    for root in roots:
        candidate = root / "models" / COMPACT_MODEL_PATH.name
        if candidate.is_file():
            return candidate
    return COMPACT_MODEL_PATH


@dataclass(frozen=True)
class PipelineConfig:
    root: Path
    model_path: Path
    output_root: Optional[Path] = None
    threshold: Optional[float] = None
    fps: int = 25
    bin_sec: int = 30

    @property
    def filtered_dir(self) -> Path:
        return self.root / "DLC_filtered"

    @property
    def original_videos_dir(self) -> Path:
        return self.root / "original_videos"

    @property
    def pred_dir(self) -> Path:
        if self.output_root is not None:
            return self.output_root / "predictions"
        return self.root / "freezing_predictions"

    @property
    def features_dir(self) -> Path:
        if self.output_root is not None:
            return self.output_root / "features"
        return self.root / "features"

    @property
    def annotated_dir(self) -> Path:
        if self.output_root is not None:
            return self.output_root / "annotated_videos"
        return self.root / "annotated_videos"

    @property
    def summary_xlsx(self) -> Path:
        if self.output_root is not None:
            return self.output_root / "summaries" / "freezing_summary.xlsx"
        return self.root / "freezing_summary.xlsx"
