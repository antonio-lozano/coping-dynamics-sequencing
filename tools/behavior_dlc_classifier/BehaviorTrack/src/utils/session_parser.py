"""Turn a folder of recordings into a named session table.

The tool's own pipeline is happy with a bare folder of videos, so this layer is
about bookkeeping: which animal a recording belongs to, which session type it is,
and whether tracking already exists for it. Everything is written to
``sessions.csv`` so the later steps do not re-derive names from filenames and
risk deriving them differently.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

SESSIONS_CSV = "sessions.csv"

#: Written by DeepLabCut into the name of every tracking file it produces.
_DLC_MARKER = "DLC"


@dataclass
class Session:
    video: str
    animal_id: str
    session_type: str
    stem: str
    tracking: str = ""

    @property
    def has_tracking(self) -> bool:
        return bool(self.tracking)


def parse_animal_id(stem: str, pattern: str) -> str:
    """Pull the animal ID out of a filename stem.

    An empty or invalid pattern yields no ID rather than raising: a cohort whose
    filenames carry no ID is a normal case, and the workflow still runs with each
    recording treated on its own.
    """
    if not pattern:
        return ""
    try:
        match = re.search(pattern, stem)
    except re.error:
        return ""
    return match.group(0) if match else ""


def parse_session_type(stem: str, tokens: Sequence[str]) -> str:
    """Name the session from the first token that appears in the stem.

    Matching is case-insensitive and bounded by anything that is not a letter or
    digit, so ``Test`` matches in ``Trial_3_Test_C5122`` but not inside
    ``Contest``. The boundary is deliberately *not* ``\\b``: ``\\w`` counts the
    underscore as a word character, and underscore-separated filenames are the
    normal case here, so ``\\bTest\\b`` would never fire on ``_Test_``.

    Tokens are tried in the order given, so a more specific token should be
    listed before a prefix of itself.
    """
    for token in tokens or []:
        token = str(token).strip()
        if not token:
            continue
        if re.search(
            rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])",
            stem,
            flags=re.IGNORECASE,
        ):
            return token
    return ""


def find_tracking_for(video: Path) -> Path | None:
    """The filtered DLC CSV belonging to one recording, if it exists.

    Looks beside the video, which is where DeepLabCut writes and where the engine
    looks. The stem is escaped before globbing, or a legal square bracket in a
    recording's name would be read as glob syntax and never match.
    """
    from glob import escape as glob_escape

    matches = sorted(video.parent.glob(f"{glob_escape(video.stem)}*filtered.csv"))
    return matches[0] if matches else None


def discover_sessions(
    videos_dir: Path,
    videotype: str,
    *,
    animal_id_pattern: str = "",
    filename_patterns: Iterable[str] = (),
) -> list[Session]:
    """Build the session table for a folder of recordings.

    Videos DeepLabCut itself produced are skipped: a labelled overlay sitting
    beside its source would otherwise be onboarded as a second recording of the
    same animal.
    """
    ext = videotype if str(videotype).startswith(".") else f".{videotype}"
    ext = ext.lower()
    if not Path(videos_dir).is_dir():
        return []

    tokens = list(filename_patterns or ())
    sessions: list[Session] = []
    for path in sorted(Path(videos_dir).iterdir()):
        if not path.is_file() or path.suffix.lower() != ext:
            continue
        if _DLC_MARKER in path.stem or path.stem.endswith("_labeled"):
            continue
        tracking = find_tracking_for(path)
        sessions.append(
            Session(
                video=path.name,
                animal_id=parse_animal_id(path.stem, animal_id_pattern),
                session_type=parse_session_type(path.stem, tokens),
                stem=path.stem,
                tracking=tracking.name if tracking else "",
            )
        )
    return sessions


def write_sessions(path: Path, sessions: Sequence[Session]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["video", "animal_id", "session_type", "stem", "tracking"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for session in sessions:
            writer.writerow({key: asdict(session)[key] for key in fields})
    return path


def read_sessions(path: Path) -> list[Session]:
    path = Path(path)
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        Session(
            video=row.get("video", ""),
            animal_id=row.get("animal_id", ""),
            session_type=row.get("session_type", ""),
            stem=row.get("stem", "") or Path(row.get("video", "")).stem,
            tracking=row.get("tracking", ""),
        )
        for row in rows
    ]


def summarise(sessions: Sequence[Session]) -> dict[str, int]:
    """Counts the status panel reports without re-walking the folder."""
    return {
        "videos": len(sessions),
        "tracked": sum(1 for s in sessions if s.has_tracking),
        "untracked": sum(1 for s in sessions if not s.has_tracking),
        "animals": len({s.animal_id for s in sessions if s.animal_id}),
        "unnamed": sum(1 for s in sessions if not s.animal_id),
    }
