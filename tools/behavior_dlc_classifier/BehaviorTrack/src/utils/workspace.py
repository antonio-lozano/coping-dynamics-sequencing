"""Where everything lives, and what state it is in.

Every window needs the same handful of answers — which folder holds the videos,
which models resolved, how many recordings are tracked, how many are classified —
and if each window worked them out for itself they would eventually disagree.
``Workspace.status()`` is the single computation behind the launcher's status
column and behind every step's own header, and it is pure enough to assert in a
self-test without opening a window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config_loader import (
    ensure_engine_importable,
    load_session_context,
    resolve_under_root,
    workspace_root,
)
from .session_parser import SESSIONS_CSV, discover_sessions, read_sessions, summarise


@dataclass(frozen=True)
class ResolvedModels:
    freezing: Path
    behavior: Path
    behavior_label_encoder: Path
    dlc_config: Path
    #: True when the freezing model is the ~1 GB full model rather than the
    #: committed 16 MB stand-in. The two do not agree, so which one ran is
    #: reported rather than assumed.
    freezing_is_full: bool

    @property
    def all_present(self) -> bool:
        return all(
            p.is_file()
            for p in (self.freezing, self.behavior, self.behavior_label_encoder, self.dlc_config)
        )

    def missing(self) -> list[str]:
        names = {
            "freezing model": self.freezing,
            "behavior model": self.behavior,
            "behavior class names": self.behavior_label_encoder,
            "DeepLabCut config": self.dlc_config,
        }
        return [label for label, path in names.items() if not path.is_file()]


@dataclass(frozen=True)
class Workspace:
    config_path: Path
    config: dict[str, Any]
    root: Path
    tool_root: Path
    raw_videos: Path
    results: Path

    @property
    def sessions_csv(self) -> Path:
        return self.results / SESSIONS_CSV

    @property
    def tracking_dir(self) -> Path:
        return self.results / "DLC"

    @property
    def features_dir(self) -> Path:
        return self.results / "Features"

    @property
    def behaviors_dir(self) -> Path:
        return self.results / "Behaviors"

    @property
    def figures_dir(self) -> Path:
        return self.results / "Figures"

    @property
    def annotated_dir(self) -> Path:
        return self.results / "Annotated_Videos"

    @property
    def summary_xlsx(self) -> Path:
        return self.results / "behavior_summary.xlsx"

    @property
    def videotype(self) -> str:
        raw = str((self.config.get("video", {}) or {}).get("videotype", ".mp4")).strip()
        return raw if raw.startswith(".") else f".{raw}"

    @property
    def fps(self) -> float:
        return float((self.config.get("video", {}) or {}).get("fps", 25) or 25)

    def section(self, name: str) -> dict[str, Any]:
        return self.config.get(name, {}) or {}

    # ------------------------------------------------------------------ #

    def ensure_dirs(self) -> None:
        for path in (
            self.raw_videos,
            self.results,
            self.tracking_dir,
            self.features_dir,
            self.behaviors_dir,
            self.figures_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def session_for_stem(self, stem: str):
        """Return the saved recording row whose source stem matches ``stem``."""
        clean = str(stem).split("DLC")[0].strip()
        for session in sorted(self.sessions(), key=lambda row: len(row.stem), reverse=True):
            saved = session.stem.strip()
            if saved == clean or clean.startswith(saved + "DLC"):
                return session
        return None

    def session_segment(self, stem: str) -> str:
        """Filesystem-safe session name, or blank when sessions are not used."""
        import re

        session = self.session_for_stem(stem)
        raw = session.session_type.strip() if session else ""
        if not raw:
            return ""
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
        return safe

    def scoped(self, base: Path, stem: str) -> Path:
        session = self.session_segment(stem)
        return Path(base) / session if session else Path(base)

    def dlc_filtered_dir(self, stem: str) -> Path:
        return self.scoped(self.tracking_dir, stem) / "Filtered_CSV"

    def dlc_tracked_videos_dir(self, stem: str) -> Path:
        return self.scoped(self.tracking_dir, stem) / "Tracked_Videos"

    def has_dlc_tracked_video(self, stem: str) -> bool:
        folder = self.dlc_tracked_videos_dir(stem)
        return folder.is_dir() and any(folder.glob(f"{stem}*filtered_labeled.mp4"))

    def behavior_predictions_dir(self, stem: str) -> Path:
        return self.scoped(self.behaviors_dir, stem) / "Predictions"

    def behavior_summaries_dir(self, stem: str) -> Path:
        return self.scoped(self.behaviors_dir, stem) / "Summaries"

    def behavior_provenance_dir(self, stem: str) -> Path:
        return self.scoped(self.behaviors_dir, stem) / "Provenance"

    def feature_output_dir(self, stem: str) -> Path:
        return self.scoped(self.features_dir, stem)

    def figure_output_dir(self, stem: str) -> Path:
        return self.scoped(self.figures_dir, stem)

    def annotated_output_dir(self, stem: str) -> Path:
        return self.scoped(self.annotated_dir, stem)

    def tracking_for_stem(self, stem: str) -> Path | None:
        matches = sorted(self.tracking_dir.rglob(f"{stem}*filtered.csv")) if self.tracking_dir.is_dir() else []
        return matches[0] if matches else None

    def sessions(self) -> list:
        """The saved session table when it exists, otherwise a fresh scan."""
        saved = read_sessions(self.sessions_csv)
        if saved:
            return saved
        onboarding = self.section("onboarding")
        return discover_sessions(
            self.raw_videos,
            self.videotype,
            animal_id_pattern=str(onboarding.get("animal_id_pattern", "") or ""),
            filename_patterns=onboarding.get("filename_patterns", []) or [],
        )

    def rescan(self) -> list:
        onboarding = self.section("onboarding")
        return discover_sessions(
            self.raw_videos,
            self.videotype,
            animal_id_pattern=str(onboarding.get("animal_id_pattern", "") or ""),
            filename_patterns=onboarding.get("filename_patterns", []) or [],
        )

    def models(self) -> ResolvedModels:
        """Resolve every model path, falling back to the engine's own defaults."""
        ensure_engine_importable(self.config_path)
        from freezing_dlc.behavior import BEHAVIOR_LABEL_ENCODER_PATH, BEHAVIOR_MODEL_PATH
        from freezing_dlc.config import FULL_MODEL_PATH, default_model_path
        from freezing_dlc.dlc_stage import DEFAULT_DLC_CONFIG_PATH

        section = self.section("models")
        freezing = section.get("freezing", "") or ""
        freezing_path = (
            resolve_under_root(self.root, freezing, "") if freezing else default_model_path()
        )
        behavior = section.get("behavior", "") or ""
        encoder = section.get("behavior_label_encoder", "") or ""
        dlc = self.section("tracking").get("dlc_config", "") or ""
        return ResolvedModels(
            freezing=Path(freezing_path),
            behavior=resolve_under_root(self.root, behavior, "") if behavior else BEHAVIOR_MODEL_PATH,
            behavior_label_encoder=(
                resolve_under_root(self.root, encoder, "") if encoder else BEHAVIOR_LABEL_ENCODER_PATH
            ),
            dlc_config=resolve_under_root(self.root, dlc, "") if dlc else DEFAULT_DLC_CONFIG_PATH,
            freezing_is_full=Path(freezing_path).name == Path(FULL_MODEL_PATH).name,
        )

    def classified_stems(self) -> set[str]:
        if not self.behaviors_dir.is_dir():
            return set()
        return {
            p.name[: -len("_behaviors.csv")]
            for p in self.behaviors_dir.rglob("*_behaviors.csv")
        }

    def status(self) -> dict[str, Any]:
        sessions = self.sessions()
        counts = summarise(sessions)
        tracked = sum(
            1
            for session in sessions
            if session.has_tracking or self.tracking_for_stem(session.stem) is not None
        )
        counts["tracked"] = tracked
        counts["untracked"] = counts["videos"] - tracked
        classified = self.classified_stems()
        done = sum(1 for s in sessions if s.stem in classified)
        try:
            models = self.models()
            missing = models.missing()
            full = models.freezing_is_full
            freezing_name = models.freezing.name
            behavior_name = models.behavior.name
        except Exception as exc:  # engine not importable, or a bad path
            missing = [f"could not resolve models: {exc}"]
            full = False
            freezing_name = "unresolved"
            behavior_name = "unresolved"
        return {
            **counts,
            "classified": done,
            "unclassified": counts["videos"] - done,
            "sessions_saved": self.sessions_csv.is_file(),
            "models_missing": missing,
            "freezing_model": freezing_name,
            "freezing_is_full": full,
            "behavior_model": behavior_name,
            "raw_videos_exists": self.raw_videos.is_dir(),
        }


def load_workspace(config_path: Path) -> Workspace:
    from .config_loader import load_config, tool_root

    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    context = load_session_context(config_path)
    root = workspace_root(config, config_path)
    paths = config.get("paths", {}) or {}
    raw_videos = (
        Path(str(context["raw_videos"])).expanduser().resolve()
        if context.get("raw_videos")
        else resolve_under_root(root, paths.get("raw_videos"), "Data/Raw_Videos")
    )
    results = (
        Path(str(context["results"])).expanduser().resolve()
        if context.get("results")
        else resolve_under_root(root, paths.get("results"), "Results")
    )
    return Workspace(
        config_path=config_path,
        config=config,
        root=root,
        tool_root=tool_root(config_path),
        raw_videos=raw_videos,
        results=results,
    )
