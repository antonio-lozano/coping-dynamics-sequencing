"""Config loading, path resolution and engine discovery for BehaviorTrack.

Two things are deliberate here.

**No new dependency.** The tool does not declare PyYAML, and the engine parses
the DeepLabCut ``config.yaml`` by hand rather than adding it. BehaviorTrack keeps
that promise: PyYAML is used when it happens to be installed, and otherwise a
small parser reads the subset of YAML this project's own config is written in
(nested maps, scalars, flat lists, comments). It is not a general YAML reader and
does not pretend to be — it raises on anything outside that subset instead of
guessing.

**The engine is found, not imported by luck.** ``freezing_dlc`` lives in
``<tool root>/src``. BehaviorTrack sits one level below the tool root, the same
way BarnesTrack sits below its engine, so the root is the parent of this project
folder and is located by looking for that ``src/freezing_dlc`` rather than by
counting ``..`` segments.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

CONFIG_NAME = "behaviortrack_config.yaml"
SESSION_CONTEXT_NAME = "behaviortrack_session_context.yaml"


# --------------------------------------------------------------------------- #
# Minimal YAML
# --------------------------------------------------------------------------- #

def _strip_comment(line: str) -> str:
    """Remove a trailing ``#`` comment that is not inside quotes."""
    out: list[str] = []
    quote: str | None = None
    for char in line:
        if quote:
            out.append(char)
            if char == quote:
                quote = None
            continue
        if char in "\"'":
            quote = char
            out.append(char)
            continue
        if char == "#":
            break
        out.append(char)
    return "".join(out).rstrip()


def _scalar(text: str) -> Any:
    text = text.strip()
    if not text:
        return ""
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    lowered = text.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "~", "none"}:
        return None
    if lowered in {"{}", "[]"}:
        return {} if lowered == "{}" else []
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _mini_yaml_parse(text: str) -> dict[str, Any]:
    """Parse the indentation-based subset this project's configs are written in.

    Two passes. A bare ``key:`` can open either a nested map or a list, and which
    one it is only becomes visible at the following line, so the first pass notes
    every key whose next deeper line starts with ``- `` and the second pass builds
    the document knowing which containers are lists.
    """
    lines = text.splitlines()

    # Pre-scan: a "key:" whose next non-blank line is a deeper "- " is a list.
    list_keys: set[tuple[int, str]] = set()
    for index, raw in enumerate(lines):
        line = _strip_comment(raw)
        if not line.strip() or line.strip().startswith("- ") or ":" not in line:
            continue
        if line.partition(":")[2].strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key = line.strip().rstrip(":").strip()
        for following in lines[index + 1 :]:
            nxt = _strip_comment(following)
            if not nxt.strip():
                continue
            nxt_indent = len(nxt) - len(nxt.lstrip(" "))
            if nxt_indent > indent and nxt.strip().startswith("- "):
                list_keys.add((indent, key))
            break

    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    for lineno, raw in enumerate(lines, start=1):
        line = _strip_comment(raw)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        container = stack[-1][1]

        if body.startswith("- "):
            if not isinstance(container, list):
                raise ValueError(f"{CONFIG_NAME}: line {lineno}: list item outside a list")
            container.append(_scalar(body[2:]))
            continue

        if ":" not in body:
            raise ValueError(f"{CONFIG_NAME}: line {lineno}: expected 'key: value'")
        key, _, value = body.partition(":")
        key, value = key.strip(), value.strip()
        if not isinstance(container, dict):
            raise ValueError(f"{CONFIG_NAME}: line {lineno}: mapping key inside a list")

        if value == "":
            child: Any = [] if (indent, key) in list_keys else {}
            container[key] = child
            stack.append((indent, child))
        else:
            container[key] = _scalar(value)
    return root


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file, preferring PyYAML and falling back to the subset parser."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return _mini_yaml_parse(text)
    loaded = yaml.safe_load(text)
    return loaded if isinstance(loaded, dict) else {}


def dump_yaml(data: dict[str, Any], path: Path) -> None:
    """Write a shallow mapping back out as YAML, without needing PyYAML."""
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        lines: list[str] = []

        def emit(mapping: dict[str, Any], indent: int) -> None:
            pad = " " * indent
            for key, value in mapping.items():
                if isinstance(value, dict):
                    lines.append(f"{pad}{key}:")
                    emit(value, indent + 2)
                elif isinstance(value, (list, tuple)):
                    lines.append(f"{pad}{key}:")
                    for item in value:
                        lines.append(f"{pad}  - {item}")
                elif value is None:
                    lines.append(f"{pad}{key}: null")
                elif isinstance(value, bool):
                    lines.append(f"{pad}{key}: {'true' if value else 'false'}")
                else:
                    lines.append(f"{pad}{key}: {value}")

        emit(data, 0)
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

def resolve_config_path(path: Path | str | None) -> Path:
    """Locate the BehaviorTrack config, defaulting to the one beside this project."""
    if path:
        candidate = Path(path).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f"BehaviorTrack config not found: {candidate}")
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "config" / CONFIG_NAME
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find config/{CONFIG_NAME} above {here}")


def project_root_from_config(config_path: Path) -> Path:
    """The BehaviorTrack folder (the parent of ``config/``)."""
    return Path(config_path).resolve().parent.parent


def tool_root(config_path: Path | None = None) -> Path:
    """The behavior_dlc_classifier folder that holds ``src/freezing_dlc``.

    Located by looking for the engine rather than by counting parents, so moving
    BehaviorTrack one level deeper does not silently break the import.
    """
    start = project_root_from_config(config_path) if config_path else Path(__file__).resolve()
    for parent in [start, *start.parents]:
        if (parent / "src" / "freezing_dlc" / "__init__.py").is_file():
            return parent
    raise FileNotFoundError(
        "Could not find the freezing_dlc engine. BehaviorTrack expects to live "
        "inside the behavior_dlc_classifier folder that holds src/freezing_dlc."
    )


def ensure_engine_importable(config_path: Path | None = None) -> Path:
    """Put ``<tool root>/src`` on sys.path and return the tool root."""
    root = tool_root(config_path)
    src = str(root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return root


def load_config(config_path: Path) -> dict[str, Any]:
    return load_yaml(Path(config_path))


def session_context_path(config_path: Path) -> Path:
    return project_root_from_config(config_path) / SESSION_CONTEXT_NAME


def load_session_context(config_path: Path) -> dict[str, Any]:
    path = session_context_path(config_path)
    return load_yaml(path) if path.is_file() else {}


def save_session_context(config_path: Path, context: dict[str, Any]) -> Path:
    path = session_context_path(config_path)
    dump_yaml(context, path)
    return path


def resolve_under_root(root: Path, value: str | Path | None, default: str) -> Path:
    """Resolve a possibly-relative config path against the workspace root."""
    raw = str(value).strip() if value not in (None, "") else default
    candidate = Path(raw).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (Path(root) / candidate).resolve()


def workspace_root(config: dict[str, Any], config_path: Path) -> Path:
    """Where videos and results live. Defaults to the BehaviorTrack folder."""
    project = config.get("project", {}) or {}
    configured = project.get("workspace_root", "")
    base = project_root_from_config(config_path)
    if not configured:
        return base
    candidate = Path(str(configured)).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()
