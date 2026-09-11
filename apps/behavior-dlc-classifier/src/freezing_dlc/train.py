from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

from .features import FEATURE_SET_VERSION, build_feature_set, dlc_filtered_csv_to_flat_df
from .model import model_from_payload


#: Criteria available for choosing the decision threshold.
#:
#: F1 rises with the fraction of frames that are positive, so maximising it on a
#: recording where the animal freezes most of the time drags the cutoff down until
#: nearly every frame is called freezing. That is how the previously shipped model
#: came to sit within two points of the do-nothing classifier on all three
#: labelled videos. Matthews correlation weighs both classes, so it does not pay
#: for that trade. Changing this criterion alone, and nothing else, raises
#: leave-one-video-out agreement on the demo data from 0.21 to 0.30.
THRESHOLD_CRITERIA = ("mcc", "balanced_accuracy", "f1")
DEFAULT_THRESHOLD_CRITERION = "mcc"

#: The nineteen candidate decision thresholds every selection step scans. One
#: constant, so cross-validation measures the same selection procedure the
#: trainer ships.
THRESHOLD_GRID = np.linspace(0.05, 0.95, 19)


@dataclass(frozen=True)
class TrainingMetrics:
    precision: float
    recall: float
    f1: float
    threshold: float
    mcc: float = 0.0
    balanced_accuracy: float = 0.0

    def criterion_score(self, criterion: str) -> float:
        """This threshold's value under the criterion used to choose thresholds."""
        try:
            return {
                "mcc": self.mcc,
                "balanced_accuracy": self.balanced_accuracy,
                "f1": self.f1,
            }[criterion]
        except KeyError:
            raise ValueError(
                f"Unknown threshold criterion: {criterion!r}. "
                f"Choose one of {', '.join(THRESHOLD_CRITERIA)}."
            ) from None


def _fmt_seconds(seconds: float) -> str:
    s = int(max(0, round(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m > 0:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _progress_bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    ratio = min(1.0, max(0.0, done / total))
    fill = int(ratio * width)
    return "[" + ("#" * fill) + ("-" * (width - fill)) + "]"


def _normalize_key(path: Path) -> str:
    stem = path.stem
    m = re.search(r"animal[_ ]?\d+_\d+", stem, flags=re.IGNORECASE)
    if m:
        return m.group(0).replace(" ", "_").lower()
    stem = re.sub(r"dlc.*$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_?filtered$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_?labels?$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_?pred(?:ictions?)?$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[\W_]+", "", stem).lower()
    return stem


def _index_by_key(csvs: List[Path], what: str) -> Dict[str, Path]:
    """Map each normalized recording key to its file, refusing ties.

    Two files reducing to the same key means one of them would be dropped, and
    which one depends on the order the filesystem happened to list them. Training
    silently on a different subset from one machine to the next is worse than
    stopping here.
    """
    by_key: Dict[str, List[Path]] = {}
    for path in sorted(csvs):
        by_key.setdefault(_normalize_key(path), []).append(path)

    collisions = {key: paths for key, paths in by_key.items() if len(paths) > 1}
    if collisions:
        detail = "; ".join(
            f"{key!r} <- {', '.join(p.name for p in paths)}" for key, paths in sorted(collisions.items())
        )
        raise ValueError(f"Several {what} files share one recording key: {detail}")
    return {key: paths[0] for key, paths in by_key.items()}


def _match_training_pairs(dlc_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path, str]]:
    dlc_map = _index_by_key(sorted(dlc_dir.glob("*.csv")), "tracking")
    labels_map = _index_by_key(sorted(labels_dir.glob("*.csv")), "label")
    keys = sorted(set(dlc_map).intersection(labels_map))
    return [(dlc_map[k], labels_map[k], k) for k in keys]


def _find_label_column(df: pd.DataFrame) -> str:
    preferred = ["pred", "label", "freezing", "freeze", "y", "target"]
    lowered = {str(c).lower(): str(c) for c in df.columns}
    for name in preferred:
        if name in lowered:
            return lowered[name]

    candidates = [
        c
        for c in df.columns
        if str(c).lower() != "unnamed: 0" and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not candidates:
        raise ValueError("No numeric label column found in labels CSV")
    return str(candidates[0])


def _load_label_array(label_path: Path) -> np.ndarray:
    ldf = pd.read_csv(label_path)
    col = _find_label_column(ldf)
    y = ldf[col].to_numpy(dtype=float)
    y = np.nan_to_num(y, nan=0.0)
    return (y >= 0.5).astype(np.uint8)


def _count_dlc_frames(dlc_path: Path) -> int:
    with open(dlc_path, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
    return max(0, line_count - 3)


def _sample_class_indices(y: np.ndarray, max_frames: int, random_state: int) -> np.ndarray:
    if max_frames <= 0 or len(y) <= max_frames:
        return np.arange(len(y), dtype=int)

    rng = np.random.default_rng(random_state)
    idx = np.arange(len(y))
    pos = idx[y == 1]
    neg = idx[y == 0]
    target_pos = int(max_frames * (len(pos) / len(idx))) if len(idx) else 0
    target_pos = max(1, min(len(pos), target_pos)) if len(pos) > 0 else 0
    target_neg = max_frames - target_pos
    target_neg = max(0, min(len(neg), target_neg))

    pick_pos = rng.choice(pos, size=target_pos, replace=False) if target_pos > 0 else np.array([], dtype=int)
    pick_neg = rng.choice(neg, size=target_neg, replace=False) if target_neg > 0 else np.array([], dtype=int)
    keep = np.concatenate([pick_pos, pick_neg])
    rng.shuffle(keep)
    return keep.astype(int, copy=False)


def _align_sequence_for_label_lag(
    x_seq: np.ndarray,
    y_seq: np.ndarray,
    label_lag_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align one time-ordered sequence for delayed/advanced manual labels.

    Positive lag means labels were clicked late and should be shifted earlier.
    """
    lag = int(label_lag_frames)
    n = min(len(x_seq), len(y_seq))
    if n <= 0:
        return x_seq[:0], y_seq[:0]

    x_seq = x_seq[:n]
    y_seq = y_seq[:n]
    if lag == 0:
        return x_seq, y_seq

    if lag > 0:
        if n <= lag:
            return x_seq[:0], y_seq[:0]
        return x_seq[:-lag], y_seq[lag:]

    lead = -lag
    if n <= lead:
        return x_seq[:0], y_seq[:0]
    return x_seq[lead:], y_seq[:-lead]


def _apply_label_lag_to_dataset(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    label_lag_frames: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    if not (len(X) == len(y) == len(groups)):
        raise ValueError("X, y, and groups must have the same length")

    lag = int(label_lag_frames)
    if lag == 0:
        info = {
            "label_lag_frames": 0,
            "kept_frames": int(len(y)),
            "dropped_frames": 0,
        }
        return X, y, groups, info

    group_rows: Dict[str, List[int]] = {}
    group_order: List[str] = []
    for idx, group in enumerate(groups.tolist()):
        key = str(group)
        if key not in group_rows:
            group_rows[key] = []
            group_order.append(key)
        group_rows[key].append(int(idx))

    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    group_parts: List[np.ndarray] = []
    dropped = 0

    for key in group_order:
        idx = np.asarray(group_rows[key], dtype=int)
        x_seq = np.asarray(X[idx])
        y_seq = np.asarray(y[idx], dtype=np.uint8)
        x_aligned, y_aligned = _align_sequence_for_label_lag(x_seq, y_seq, lag)
        dropped += int(len(y_seq) - len(y_aligned))
        if len(y_aligned) == 0:
            continue
        x_parts.append(x_aligned)
        y_parts.append(y_aligned.astype(np.uint8, copy=False))
        group_parts.append(np.full(len(y_aligned), key, dtype=groups.dtype))

    if not x_parts:
        raise ValueError(
            "Label lag removed all frames; reduce absolute label_lag_frames or provide longer sequences"
        )

    X_out = np.concatenate(x_parts, axis=0)
    y_out = np.concatenate(y_parts, axis=0)
    groups_out = np.concatenate(group_parts, axis=0)
    info = {
        "label_lag_frames": lag,
        "kept_frames": int(len(y_out)),
        "dropped_frames": int(dropped),
    }
    return X_out, y_out, groups_out, info


def _build_sample_plan(
    pairs: List[Tuple[Path, Path, str]],
    max_frames: Optional[int],
    random_state: int,
) -> Tuple[List[np.ndarray], np.ndarray]:
    usable_labels: List[np.ndarray] = []
    usable_lengths: List[int] = []
    total_rows = 0

    for dlc_path, label_path, _ in pairs:
        y = _load_label_array(label_path)
        usable_n = min(len(y), _count_dlc_frames(dlc_path))
        usable_lengths.append(usable_n)
        usable_labels.append(y[:usable_n])
        total_rows += usable_n

    if max_frames is None or max_frames <= 0 or total_rows <= max_frames:
        sample_plan = [np.arange(n, dtype=int) for n in usable_lengths]
    else:
        stacked_y = np.concatenate(usable_labels) if usable_labels else np.array([], dtype=np.uint8)
        keep_flat = _sample_class_indices(stacked_y, max_frames=max_frames, random_state=random_state)
        offsets = np.cumsum([0, *usable_lengths])
        sample_plan = []
        for idx, usable_n in enumerate(usable_lengths):
            start = int(offsets[idx])
            end = int(offsets[idx + 1])
            local = keep_flat[(keep_flat >= start) & (keep_flat < end)] - start
            sample_plan.append(local.astype(int, copy=False))

    return sample_plan, np.asarray(usable_lengths, dtype=int)


def build_training_dataset(
    dlc_dir: Path,
    labels_dir: Path,
    feature_mode: str = "compact",
    max_frames: Optional[int] = None,
    random_state: int = 42,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    pairs = _match_training_pairs(dlc_dir, labels_dir)
    if not pairs:
        raise FileNotFoundError(f"No matched CSV pairs between '{dlc_dir}' and '{labels_dir}'.")

    sample_plan, usable_lengths = _build_sample_plan(pairs, max_frames=max_frames, random_state=random_state)

    feat_blocks: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    keys: List[str] = []
    feature_names: Optional[List[str]] = None

    total_pairs = len(pairs)
    t0 = time.time()
    for idx, (dlc_path, label_path, key) in enumerate(pairs, start=1):
        flat = dlc_filtered_csv_to_flat_df(dlc_path)
        feat = build_feature_set(flat, mode=feature_mode)
        y = _load_label_array(label_path)

        n = min(len(feat), len(y), int(usable_lengths[idx - 1]))
        if n == 0:
            continue

        if feature_names is None:
            feature_names = [str(c) for c in feat.columns]

        keep = np.sort(sample_plan[idx - 1])
        # The plan was sized from the raw line count, which sees trailing blank
        # lines that pandas skips; indexing past the parsed rows would raise.
        keep = keep[keep < n]
        if len(keep) == 0:
            continue

        feat_np = feat.iloc[keep].to_numpy(dtype=np.float32, copy=False)
        y_keep = y[keep].astype(np.uint8, copy=False)

        feat_blocks.append(feat_np)
        labels.append(y_keep)
        keys.extend([key] * len(keep))

        if show_progress:
            elapsed = time.time() - t0
            per_item = elapsed / idx
            eta = per_item * (total_pairs - idx)
            bar = _progress_bar(idx, total_pairs)
            print(
                f"\r[features] {bar} {idx}/{total_pairs} elapsed={_fmt_seconds(elapsed)} eta={_fmt_seconds(eta)}",
                end="",
                flush=True,
            )

    if show_progress:
        print()

    if not feat_blocks or feature_names is None:
        raise ValueError("No usable matched files after alignment")

    X = np.concatenate(feat_blocks, axis=0)
    y_out = np.concatenate(labels)
    return X, y_out, keys, feature_names


def _dataset_signature(
    pairs: List[Tuple[Path, Path, str]],
    feature_mode: str,
    max_frames: Optional[int] = None,
    random_state: Optional[int] = None,
) -> str:
    h = hashlib.sha256()
    h.update(feature_mode.encode("utf-8"))
    h.update(FEATURE_SET_VERSION.encode("utf-8"))
    if max_frames is not None and max_frames > 0:
        h.update(f"max_frames={int(max_frames)}".encode("utf-8"))
        h.update(f"random_state={int(random_state or 0)}".encode("utf-8"))
    for dlc_path, label_path, key in pairs:
        ds = dlc_path.stat()
        ls = label_path.stat()
        h.update(str(dlc_path).encode("utf-8"))
        h.update(str(ds.st_mtime_ns).encode("utf-8"))
        h.update(str(ds.st_size).encode("utf-8"))
        h.update(str(label_path).encode("utf-8"))
        h.update(str(ls.st_mtime_ns).encode("utf-8"))
        h.update(str(ls.st_size).encode("utf-8"))
        h.update(key.encode("utf-8"))
    return h.hexdigest()[:16]


def _cache_paths(cache_dir: Path, signature: str) -> Dict[str, Path]:
    return {
        "x": cache_dir / f"X_{signature}.npy",
        "y": cache_dir / f"y_{signature}.npy",
        "keys": cache_dir / f"keys_{signature}.npy",
        "columns": cache_dir / f"columns_{signature}.json",
        "meta": cache_dir / f"meta_{signature}.json",
    }


def _align_feature_columns(
    X: np.ndarray,
    source_columns: Sequence[str],
    target_columns: Sequence[str],
) -> np.ndarray:
    if list(source_columns) == list(target_columns):
        return np.asarray(X, dtype=np.float32)

    target_lookup = {name: idx for idx, name in enumerate(target_columns)}
    mapped = [(src_idx, target_lookup[name]) for src_idx, name in enumerate(source_columns) if name in target_lookup]
    out = np.zeros((len(X), len(target_columns)), dtype=np.float32)
    for src_idx, dst_idx in mapped:
        out[:, dst_idx] = np.asarray(X[:, src_idx], dtype=np.float32)
    return out


def load_or_build_training_dataset(
    dlc_dir: Path,
    labels_dir: Path,
    feature_mode: str,
    cache_dir: Optional[Path],
    force_rebuild_cache: bool,
    show_progress: bool,
    max_frames: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], Dict[str, object]]:
    pairs = _match_training_pairs(dlc_dir, labels_dir)
    if not pairs:
        raise FileNotFoundError(f"No matched CSV pairs between '{dlc_dir}' and '{labels_dir}'.")

    signature = _dataset_signature(pairs, feature_mode, max_frames=max_frames, random_state=random_state)
    cache_info: Dict[str, object] = {
        "enabled": bool(cache_dir),
        "hit": False,
        "signature": signature,
    }
    if max_frames is not None and max_frames > 0:
        cache_info["max_frames"] = int(max_frames)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        paths = _cache_paths(cache_dir, signature)
        all_exist = all(p.exists() for p in paths.values())
        if all_exist and not force_rebuild_cache:
            if show_progress:
                print(f"[cache] loading cached features: {signature}")
            x = np.load(paths["x"], mmap_mode="r")
            y = np.load(paths["y"])
            keys_np = np.load(paths["keys"])
            with open(paths["columns"], "r", encoding="utf-8") as f:
                columns = json.load(f)
            keys = [str(k) for k in keys_np.tolist()]
            cache_info["hit"] = True
            cache_info["path"] = str(cache_dir)
            return x, y, keys, [str(c) for c in columns], cache_info

    X, y, keys, columns = build_training_dataset(
        dlc_dir,
        labels_dir,
        feature_mode=feature_mode,
        max_frames=max_frames,
        random_state=random_state,
        show_progress=show_progress,
    )

    if cache_dir is not None:
        paths = _cache_paths(cache_dir, signature)
        if show_progress:
            print(f"[cache] saving features: {signature}")
        np.save(paths["x"], np.asarray(X, dtype=np.float32))
        np.save(paths["y"], y.astype(np.uint8))
        np.save(paths["keys"], np.asarray(keys, dtype=str))
        with open(paths["columns"], "w", encoding="utf-8") as f:
            json.dump(columns, f)
        with open(paths["meta"], "w", encoding="utf-8") as f:
            json.dump(
                {
                    "signature": signature,
                    "feature_mode": feature_mode,
                    "n_rows": int(len(X)),
                    "n_cols": int(X.shape[1]),
                    "max_frames": int(max_frames) if max_frames is not None and max_frames > 0 else None,
                },
                f,
            )
        cache_info["path"] = str(cache_dir)

    return X, y, keys, columns, cache_info


def _threshold_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> TrainingMetrics:
    pred = (probs >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y_true == 1)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Derived from the same four counts rather than by calling scikit-learn once
    # per candidate threshold, which the tree-count search does many times over.
    denominator = float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    mcc = ((tp * tn) - (fp * fn)) / (denominator**0.5) if denominator > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_accuracy = (recall + specificity) / 2.0

    return TrainingMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        threshold=threshold,
        mcc=float(mcc),
        balanced_accuracy=float(balanced_accuracy),
    )


def _best_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    grid: Sequence[float],
    criterion: str = DEFAULT_THRESHOLD_CRITERION,
) -> TrainingMetrics:
    best: Optional[TrainingMetrics] = None
    best_score = -np.inf
    for t in grid:
        m = _threshold_metrics(y_true, probs, float(t))
        score = m.criterion_score(criterion)
        if score > best_score:
            best, best_score = m, score
    if best is None:
        raise RuntimeError("Failed to compute threshold metrics")
    return best


@dataclass(frozen=True)
class DataSplit:
    """One partition of the labelled frames into fit, selection and report sets.

    ``test_idx`` is empty when there were too few videos to hold any back. The
    caller must consult ``holdout_is_honest`` before quoting a number, because a
    score measured on the same frames that chose the threshold is optimistic.
    """

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    mode: str
    holdout_is_honest: bool
    n_groups_train: int
    n_groups_val: int
    n_groups_test: int


def _split_dataset(
    y: np.ndarray,
    groups: np.ndarray,
    split_mode: str,
    test_size: float,
    random_state: int,
) -> DataSplit:
    """Partition frames into a fit set, a selection set and a held-out report set.

    The decision threshold and the tree count are both chosen on the selection
    set, so a score measured there is the best of nineteen thresholds on the very
    frames that picked it. Only the report set, which nothing selects on, gives a
    number that transfers to new animals.

    Videos are kept whole. Frames from one recording are strongly correlated, so
    splitting inside a video would leave near-duplicate frames on both sides and
    inflate every score regardless of the threshold.
    """
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    n_groups = int(len(np.unique(groups)))
    frames = np.arange(len(y))

    if split_mode.lower().strip() == "group" and n_groups >= 2:
        outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        rest_idx, test_idx = next(outer.split(frames, y, groups=groups))

        # A third partition only exists if what remains still holds two videos:
        # one to fit on and one to select the threshold on.
        if n_groups >= 3 and len(np.unique(groups[rest_idx])) >= 2:
            inner = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state + 1)
            train_local, val_local = next(
                inner.split(rest_idx, y[rest_idx], groups=groups[rest_idx])
            )
            train_idx, val_idx = rest_idx[train_local], rest_idx[val_local]
            return DataSplit(
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                mode="group",
                holdout_is_honest=True,
                n_groups_train=int(len(np.unique(groups[train_idx]))),
                n_groups_val=int(len(np.unique(groups[val_idx]))),
                n_groups_test=int(len(np.unique(groups[test_idx]))),
            )

        # Two videos: fit on one, select on the other, and admit that the score
        # that comes back was measured where the threshold was chosen.
        return DataSplit(
            train_idx=rest_idx,
            val_idx=test_idx,
            test_idx=np.array([], dtype=int),
            mode="group",
            holdout_is_honest=False,
            n_groups_train=int(len(np.unique(groups[rest_idx]))),
            n_groups_val=int(len(np.unique(groups[test_idx]))),
            n_groups_test=0,
        )

    stratify = y if len(np.unique(y)) > 1 else None
    train_idx, val_idx = train_test_split(
        frames, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return DataSplit(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=np.array([], dtype=int),
        mode="random",
        holdout_is_honest=False,
        n_groups_train=n_groups,
        n_groups_val=0,
        n_groups_test=0,
    )


def train_model(
    dlc_dir: Path,
    labels_dir: Path,
    out_model: Path,
    feature_mode: str = "stillness",
    n_estimators: int = 600,
    step_estimators: int = 100,
    threshold_criterion: str = DEFAULT_THRESHOLD_CRITERION,
    min_samples_leaf: int = 20,
    refit_on_all: bool = True,
    stabilize: bool = False,
    stability_patience: int = 4,
    stability_min_delta: float = 0.0005,
    min_estimators_for_stability: int = 300,
    max_train_frames: Optional[int] = None,
    split_mode: str = "group",
    feature_cache_dir: Optional[Path] = None,
    force_rebuild_cache: bool = False,
    n_jobs: int = -1,
    random_state: int = 42,
    test_size: float = 0.2,
    extra_dlc_dir: Optional[Path] = None,
    extra_labels_dir: Optional[Path] = None,
    label_lag_frames: int = 0,
    show_progress: bool = True,
) -> Dict[str, object]:
    from sklearn.ensemble import RandomForestClassifier

    has_extra = (extra_dlc_dir is not None) or (extra_labels_dir is not None)
    if has_extra and (extra_dlc_dir is None or extra_labels_dir is None):
        raise ValueError("Both extra_dlc_dir and extra_labels_dir must be provided together")

    dataset_pairs: List[Tuple[Path, Path]] = [(dlc_dir, labels_dir)]
    if has_extra and extra_dlc_dir is not None and extra_labels_dir is not None:
        dataset_pairs.append((extra_dlc_dir, extra_labels_dir))

    # The label lag shifts labels by row position, which only equals a frame
    # offset while rows are consecutive. Sampling inside the load would break
    # that silently, so with both requested the frame cap waits until after
    # the lag has been applied.
    defer_sampling = bool(label_lag_frames) and bool(max_train_frames)
    load_max_frames = None if defer_sampling else max_train_frames

    loaded_sets: List[Tuple[np.ndarray, np.ndarray, List[str], List[str], Dict[str, object]]] = []
    for curr_dlc_dir, curr_labels_dir in dataset_pairs:
        loaded_sets.append(
            load_or_build_training_dataset(
                curr_dlc_dir,
                curr_labels_dir,
                feature_mode=feature_mode,
                cache_dir=feature_cache_dir,
                force_rebuild_cache=force_rebuild_cache,
                show_progress=show_progress,
                max_frames=load_max_frames,
                random_state=random_state,
            )
        )

    merged_feature_names: List[str] = []
    seen_features = set()
    for _, _, _, columns, _ in loaded_sets:
        for name in columns:
            if name not in seen_features:
                seen_features.add(name)
                merged_feature_names.append(name)

    x_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    keys: List[str] = []
    cache_sources: List[Dict[str, object]] = []
    for ds_idx, (x_curr, y_curr, keys_curr, columns_curr, cache_curr) in enumerate(loaded_sets):
        x_aligned = _align_feature_columns(x_curr, columns_curr, merged_feature_names)
        x_parts.append(x_aligned)
        y_parts.append(np.asarray(y_curr, dtype=np.uint8))
        keys.extend([f"ds{ds_idx}:{k}" for k in keys_curr])

        source_info = {
            "dataset_index": int(ds_idx),
            "dlc_dir": str(dataset_pairs[ds_idx][0]),
            "labels_dir": str(dataset_pairs[ds_idx][1]),
        }
        source_info.update(cache_curr)
        cache_sources.append(source_info)

    if not x_parts:
        raise ValueError("No training data loaded from any dataset source")

    X = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts)
    feature_names = merged_feature_names
    if len(cache_sources) == 1:
        cache_info = cache_sources[0]
    else:
        cache_info = {
            "enabled": any(bool(src.get("enabled")) for src in cache_sources),
            "multi_source": True,
            "n_sources": int(len(cache_sources)),
            "sources": cache_sources,
        }

    if show_progress and len(dataset_pairs) > 1:
        print(
            f"[dataset] combined sources={len(dataset_pairs)} frames={len(y)} features={len(feature_names)}",
            flush=True,
        )

    groups = np.asarray(keys)
    X, y, groups, lag_info = _apply_label_lag_to_dataset(
        X,
        y,
        groups,
        label_lag_frames=label_lag_frames,
    )

    if show_progress:
        print(
            f"[align] label_lag_frames={lag_info['label_lag_frames']} "
            f"kept={lag_info['kept_frames']} dropped={lag_info['dropped_frames']}",
            flush=True,
        )

    if defer_sampling and max_train_frames and len(y) > max_train_frames:
        keep = np.sort(_sample_class_indices(y, max_frames=int(max_train_frames), random_state=random_state))
        X, y, groups = X[keep], y[keep], groups[keep]
        if show_progress:
            print(f"[sample] capped to {len(y)} frames after applying the label lag", flush=True)

    split = _split_dataset(y, groups, split_mode, test_size, random_state)
    train_idx, val_idx, test_idx = split.train_idx, split.val_idx, split.test_idx
    split_mode_norm = split.mode
    n_groups_train, n_groups_val = split.n_groups_train, split.n_groups_val
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    _require_both_classes(y_train, "fit split")

    n_train_frames = int(len(train_idx))
    n_val_frames = int(len(val_idx))
    n_test_frames = int(len(test_idx))
    total_frames = int(len(y))
    train_ratio = (n_train_frames / total_frames) if total_frames > 0 else 0.0
    val_ratio = (n_val_frames / total_frames) if total_frames > 0 else 0.0

    if show_progress:
        split_msg = (
            f"[split] mode={split_mode_norm} fit={n_train_frames} ({train_ratio:.1%}) "
            f"select={n_val_frames} ({val_ratio:.1%})"
        )
        if n_test_frames:
            split_msg += f" held-out={n_test_frames} ({n_test_frames / total_frames:.1%})"
        if split_mode_norm == "group":
            split_msg += (
                f" videos={n_groups_train}/{n_groups_val}/{split.n_groups_test}"
            )
        print(split_msg, flush=True)
        if not split.holdout_is_honest:
            print(
                "[split] WARNING: too few videos to hold any back. The reported score "
                "is measured on the same frames that chose the threshold, so it is "
                "optimistic. Label at least three videos for a score that transfers.",
                flush=True,
            )

    forest_params = {
        "criterion": "entropy",
        "max_features": "sqrt",
        "min_samples_leaf": int(min_samples_leaf),
        "n_jobs": int(n_jobs),
        "random_state": random_state,
    }
    clf = RandomForestClassifier(
        n_estimators=max(1, min(step_estimators, n_estimators)),
        warm_start=True,
        **forest_params,
    )

    x_train_np = np.asarray(X[train_idx], dtype=np.float32)
    x_val_np = np.asarray(X[val_idx], dtype=np.float32)
    x_test_np = np.asarray(X[test_idx], dtype=np.float32)
    del X

    max_estimators = int(n_estimators)
    step = max(1, int(step_estimators))
    fitted_estimators = 0
    grid = THRESHOLD_GRID

    best: Optional[TrainingMetrics] = None
    best_score = -np.inf
    no_improve_rounds = 0
    stopped_early = False
    history: List[Dict[str, float]] = []
    train_t0 = time.time()

    while fitted_estimators < max_estimators:
        next_estimators = min(fitted_estimators + step, max_estimators)
        setattr(clf, "n_estimators", int(next_estimators))
        step_t0 = time.time()
        clf.fit(x_train_np, y_train)
        step_elapsed = time.time() - step_t0

        fitted_estimators = int(next_estimators)
        probs = clf.predict_proba(x_val_np)[:, 1]
        curr = _best_threshold(y_val, probs, grid, criterion=threshold_criterion)
        # Early stopping watches the same quantity the threshold is chosen by, so
        # adding trees stops when the thing being optimised stops improving.
        curr_score = curr.criterion_score(threshold_criterion)
        improved = curr_score > (best_score + stability_min_delta)

        if improved or best is None:
            best = curr
            best_score = curr_score
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1

        elapsed = time.time() - train_t0
        trees_left = max_estimators - fitted_estimators
        eta = (elapsed / fitted_estimators) * trees_left if fitted_estimators > 0 else 0.0
        history.append(
            {
                "n_estimators": float(fitted_estimators),
                "f1": float(curr.f1),
                "mcc": float(curr.mcc),
                "balanced_accuracy": float(curr.balanced_accuracy),
                "precision": float(curr.precision),
                "recall": float(curr.recall),
                "threshold": float(curr.threshold),
                "step_seconds": float(step_elapsed),
                "elapsed_seconds": float(elapsed),
            }
        )

        if show_progress:
            bar = _progress_bar(fitted_estimators, max_estimators)
            print(
                f"\r[train] {bar} {fitted_estimators}/{max_estimators} trees "
                f"p={curr.precision:.4f} r={curr.recall:.4f} "
                f"f1={curr.f1:.4f} {threshold_criterion}={curr_score:.4f} best={best_score:.4f} "
                f"patience={no_improve_rounds}/{stability_patience} "
                f"eta={_fmt_seconds(eta)}",
                end="",
                flush=True,
            )

        if (
            stabilize
            and fitted_estimators >= min_estimators_for_stability
            and no_improve_rounds >= stability_patience
        ):
            stopped_early = True
            break

    if show_progress:
        print()

    if best is None:
        raise RuntimeError("Training failed to produce validation metrics")

    # The threshold and the tree count were both chosen on the selection set, so
    # `best` is the highest of nineteen thresholds on the frames that picked it.
    # Applying that fixed threshold to frames nothing selected on is the number
    # that describes performance on a new animal.
    def as_dict(m: TrainingMetrics, y_true: np.ndarray) -> Dict[str, float]:
        # F1 alone cannot be read without knowing how much the animal froze, so
        # the base rate and the score of the classifier that calls every frame
        # freezing travel with it.
        rate = float(y_true.mean()) if len(y_true) else 0.0
        return {
            "precision": float(m.precision),
            "recall": float(m.recall),
            "f1": float(m.f1),
            "mcc": float(m.mcc),
            "balanced_accuracy": float(m.balanced_accuracy),
            "freezing_fraction": rate,
            "f1_if_everything_called_freezing": _trivial_f1(rate),
        }

    selection_metrics = as_dict(best, y_val)
    if split.holdout_is_honest and len(test_idx):
        held_out = _threshold_metrics(y_test, clf.predict_proba(x_test_np)[:, 1], best.threshold)
        headline = as_dict(held_out, y_test)
        metrics_source = "held_out_videos"
    else:
        headline = dict(selection_metrics)
        metrics_source = "selection_set_optimistic"

    if show_progress:
        print(
            f"[metrics] {metrics_source}: "
            f"p={headline['precision']:.4f} r={headline['recall']:.4f} f1={headline['f1']:.4f} "
            f"mcc={headline['mcc']:.4f} at threshold {best.threshold:.2f}",
            flush=True,
        )
        print(
            f"[metrics] the animal froze on {headline['freezing_fraction']:.0%} of these frames, "
            f"so calling every frame freezing would already score "
            f"f1={headline['f1_if_everything_called_freezing']:.4f} (mcc 0). "
            "Read the f1 against that number.",
            flush=True,
        )
        if metrics_source == "held_out_videos":
            print(
                f"[metrics] the selection set reported f1={selection_metrics['f1']:.4f} "
                f"mcc={selection_metrics['mcc']:.4f}; the held-out figures above are the ones to quote.",
                flush=True,
            )

    # The split has now done its measuring. The model that ships should still see
    # every labelled frame, because holding two thirds of the data back costs real
    # accuracy: on the demo videos, fitting on two recordings rather than one
    # lifts agreement on a held-out third from 0.11 to 0.63 for the animal least
    # like the others. The metrics above therefore describe a model fitted on less
    # data than the one saved here, which makes them conservative rather than
    # flattering.
    refitted = False
    if refit_on_all and (len(val_idx) or len(test_idx)):
        if show_progress:
            print(
                f"[refit] fitting the saved model on all {len(y)} labelled frames "
                f"from {int(len(np.unique(groups)))} videos",
                flush=True,
            )
        final = RandomForestClassifier(n_estimators=int(fitted_estimators), **forest_params)
        final.fit(
            np.concatenate([x_train_np, x_val_np, x_test_np]),
            np.concatenate([y_train, y_val, y_test]),
        )
        clf = final
        refitted = True

    out_model.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, object] = {
        "classifier": clf,
        "feature_names": feature_names,
        "feature_mode": str(feature_mode),
        "threshold": float(best.threshold),
        "threshold_criterion": str(threshold_criterion),
        "min_samples_leaf": int(min_samples_leaf),
        # True means the saved forest saw every labelled frame, so `metrics`
        # describe a model fitted on the training split only and understate it.
        "refit_on_all_frames": refitted,
        "metrics": headline,
        "metrics_source": metrics_source,
        "selection_metrics": selection_metrics,
        "n_training_frames": int(len(y)),
        "n_train_frames": n_train_frames,
        "n_val_frames": n_val_frames,
        "n_test_frames": n_test_frames,
        "n_features": int(len(feature_names)),
        "n_estimators": int(fitted_estimators),
        "requested_n_estimators": int(n_estimators),
        "max_train_frames": int(max_train_frames) if max_train_frames else None,
        "n_jobs": int(n_jobs),
        "test_size": float(test_size),
        "label_lag_frames": int(lag_info["label_lag_frames"]),
        "split_mode": split_mode_norm,
        "n_groups_train": n_groups_train,
        "n_groups_val": n_groups_val,
        "n_groups_test": int(split.n_groups_test),
        "holdout_is_honest": bool(split.holdout_is_honest),
        # Which videos went where, so a later evaluation can tell whether it is
        # scoring frames the model was fitted on. `groups_fitted` is the one that
        # answers that for the saved forest: after a refit it covers every video,
        # not just the training split.
        "groups_train": sorted(set(groups[train_idx].tolist())),
        "groups_val": sorted(set(groups[val_idx].tolist())),
        "groups_test": sorted(set(groups[test_idx].tolist())),
        "groups_fitted": sorted(
            set(groups.tolist() if refitted else groups[train_idx].tolist())
        ),
        "stopped_early": bool(stopped_early),
        "alignment": lag_info,
        "stability": {
            "enabled": bool(stabilize),
            "patience": int(stability_patience),
            "min_delta": float(stability_min_delta),
            "min_estimators": int(min_estimators_for_stability),
        },
        "training_history": history,
        "feature_cache": cache_info,
    }
    joblib.dump(payload, out_model)
    return payload


def _require_both_classes(y_arr: np.ndarray, where: str) -> None:
    """Fail naming the cause, instead of an IndexError deep inside predict_proba."""
    if len(np.unique(y_arr)) < 2:
        raise ValueError(
            f"The {where} contains only one class (all freezing or all moving frames). "
            "Label more videos, or change random_state so both classes land in every split."
        )


def _trivial_f1(positive_rate: float) -> float:
    """F1 of the do-nothing classifier that calls every frame freezing."""
    if positive_rate <= 0.0:
        return 0.0
    return float(2.0 * positive_rate / (1.0 + positive_rate))


def _balance_free_metrics(
    y_true: np.ndarray, probs: np.ndarray, metrics: TrainingMetrics
) -> Dict[str, float]:
    """Scores that do not move with the fraction of frames that are freezing.

    F1 rises with the positive rate, so a model looks excellent on an animal that
    freezes constantly and poor on one that rarely does, even when it separates
    the two states equally well in both. Matthews correlation and balanced
    accuracy stay at the chosen threshold but account for both classes, so a
    shortfall in either is real rather than an artefact of how much the animal
    froze. Both come from ``metrics``, which already holds them, rather than
    being computed a second way here.

    Only the area under the ROC curve is new, and it answers a different question
    entirely: ignoring the threshold, are freezing frames ranked above moving
    ones at all. That separates "cannot tell the states apart" from "ranks them
    correctly but the cutoff suits a different animal".
    """
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        # One class present: none of the three is defined.
        return {"balanced_accuracy": float("nan"), "mcc": float("nan"), "roc_auc": float("nan")}
    return {
        "balanced_accuracy": float(metrics.balanced_accuracy),
        "mcc": float(metrics.mcc),
        "roc_auc": float(roc_auc_score(y_true, probs)),
    }


def cross_validate_model(
    dlc_dir: Path,
    labels_dir: Path,
    feature_mode: str = "stillness",
    n_estimators: int = 200,
    label_lag_frames: int = 0,
    threshold: Optional[float] = None,
    threshold_criterion: str = DEFAULT_THRESHOLD_CRITERION,
    min_samples_leaf: int = 20,
    max_train_frames: Optional[int] = None,
    feature_cache_dir: Optional[Path] = None,
    force_rebuild_cache: bool = False,
    n_jobs: int = -1,
    random_state: int = 42,
    show_progress: bool = True,
) -> Dict[str, object]:
    """Score every video once, always fitting on the others.

    A single train/test split of a handful of recordings says more about which
    video happened to land where than about the classifier: on the three bundled
    videos the same model scores anywhere from 0.67 to 0.93 depending on the
    draw. Holding out each recording in turn uses all of them and reports the
    spread, which is what shows whether a number would survive another animal.

    Inside each fold the threshold is chosen on a further split of the *training*
    videos, so the held-out recording contributes to nothing except its own
    score. Pass ``threshold`` to fix it instead, which is what to do when there
    are too few videos to spare one for tuning.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupShuffleSplit

    # Same ordering rule as train_model: with a label lag, the frame cap waits
    # until after the lag, because the lag only means a frame offset on
    # consecutive rows.
    defer_sampling = bool(label_lag_frames) and bool(max_train_frames)
    X, y, keys, _, _ = load_or_build_training_dataset(
        dlc_dir,
        labels_dir,
        feature_mode=feature_mode,
        cache_dir=feature_cache_dir,
        force_rebuild_cache=force_rebuild_cache,
        show_progress=show_progress,
        max_frames=None if defer_sampling else max_train_frames,
        random_state=random_state,
    )
    X, y, groups, lag_info = _apply_label_lag_to_dataset(
        X, y, np.asarray(keys), label_lag_frames=label_lag_frames
    )
    if defer_sampling and max_train_frames and len(y) > max_train_frames:
        keep = np.sort(_sample_class_indices(y, max_frames=int(max_train_frames), random_state=random_state))
        X, y, groups = X[keep], y[keep], groups[keep]

    # The same forest settings the trainer uses, so this measures the model that
    # actually ships rather than a differently regularised stand-in.
    forest_params = {
        "criterion": "entropy",
        "max_features": "sqrt",
        "min_samples_leaf": int(min_samples_leaf),
        "n_jobs": int(n_jobs),
        "random_state": random_state,
    }

    video_names = sorted(set(groups.tolist()))
    if len(video_names) < 2:
        raise ValueError(
            "Leave-one-video-out needs at least two labelled videos; "
            f"found {len(video_names)}."
        )

    grid = THRESHOLD_GRID
    folds: List[Dict[str, object]] = []

    for position, held_out in enumerate(video_names, start=1):
        test_mask = groups == held_out
        train_mask = ~test_mask
        train_groups = groups[train_mask]

        fold_threshold = threshold
        if fold_threshold is None:
            if len(np.unique(train_groups)) < 2:
                raise ValueError(
                    "Choosing a threshold inside each fold needs at least three "
                    "labelled videos. Pass an explicit threshold, or label more videos."
                )
            inner = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state + position)
            train_idx = np.flatnonzero(train_mask)
            fit_local, select_local = next(
                inner.split(train_idx, y[train_mask], groups=train_groups)
            )
            _require_both_classes(y[train_idx[fit_local]], "threshold-tuning split")
            tuner = RandomForestClassifier(n_estimators=n_estimators, **forest_params)
            tuner.fit(X[train_idx[fit_local]], y[train_idx[fit_local]])
            select_probs = tuner.predict_proba(X[train_idx[select_local]])[:, 1]
            fold_threshold = _best_threshold(
                y[train_idx[select_local]], select_probs, grid, criterion=threshold_criterion
            ).threshold

        _require_both_classes(y[train_mask], "training fold")
        clf = RandomForestClassifier(n_estimators=n_estimators, **forest_params)
        clf.fit(X[train_mask], y[train_mask])
        probs = clf.predict_proba(X[test_mask])[:, 1]
        y_test = y[test_mask]
        metrics = _threshold_metrics(y_test, probs, fold_threshold)

        folds.append(
            {
                "held_out_video": held_out,
                "threshold": float(fold_threshold),
                "precision": float(metrics.precision),
                "recall": float(metrics.recall),
                "f1": float(metrics.f1),
                **_balance_free_metrics(y_test, probs, metrics),
                # What a classifier that simply called every frame freezing would
                # score here. On an animal that freezes most of the time that
                # number is already high, so an F1 near it means nothing was learnt.
                "f1_if_everything_called_freezing": _trivial_f1(float(y_test.mean())),
                "n_frames": int(test_mask.sum()),
                "freezing_fraction": float(y_test.mean()) if test_mask.any() else 0.0,
            }
        )
        if show_progress:
            fold = folds[-1]
            print(
                f"[cv] {position}/{len(video_names)} held out {held_out}: "
                f"f1={metrics.f1:.4f} (call-everything baseline "
                f"{fold['f1_if_everything_called_freezing']:.4f}) "
                f"mcc={fold['mcc']:.4f} auc={fold['roc_auc']:.4f} "
                f"freezing={fold['freezing_fraction']:.0%}",
                flush=True,
            )

    def spread(name: str) -> Dict[str, float]:
        values = np.array([float(fold[name]) for fold in folds])
        return {
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }

    summary = {
        "n_videos": len(video_names),
        "n_frames": int(len(y)),
        "feature_mode": feature_mode,
        "n_estimators": int(n_estimators),
        "label_lag_frames": int(lag_info["label_lag_frames"]),
        "fixed_threshold": float(threshold) if threshold is not None else None,
        "threshold_criterion": str(threshold_criterion),
        "min_samples_leaf": int(min_samples_leaf),
        "precision": spread("precision"),
        "recall": spread("recall"),
        "f1": spread("f1"),
        "balanced_accuracy": spread("balanced_accuracy"),
        "mcc": spread("mcc"),
        "roc_auc": spread("roc_auc"),
        "f1_if_everything_called_freezing": spread("f1_if_everything_called_freezing"),
        "folds": folds,
    }
    if show_progress:
        for name in ("f1", "mcc", "roc_auc"):
            stat = summary[name]
            print(
                f"[cv] {name:>8s} = {stat['mean']:.4f} +/- {stat['sd']:.4f} "
                f"(range {stat['min']:.4f}-{stat['max']:.4f})",
                flush=True,
            )
        print(f"[cv] over {len(video_names)} videos, each held out in turn", flush=True)
    return summary


def evaluate_model(
    model_path: Path,
    dlc_dir: Path,
    labels_dir: Path,
    threshold: Optional[float] = None,
    label_lag_frames: Optional[int] = None,
    max_eval_frames: Optional[int] = None,
    eval_chunk_size: int = 100_000,
    feature_cache_dir: Optional[Path] = None,
    force_rebuild_cache: bool = False,
) -> Dict[str, object]:
    obj = joblib.load(model_path)
    if not isinstance(obj, dict) or "classifier" not in obj:
        raise TypeError("Model file must contain a dict with key 'classifier'")

    loaded = model_from_payload(obj)
    clf = loaded.classifier
    feat_names = loaded.feature_names
    feature_mode = loaded.feature_mode
    default_t = loaded.threshold if loaded.threshold is not None else 0.5
    t = default_t if threshold is None else float(threshold)
    lag = int(obj.get("label_lag_frames", 0)) if label_lag_frames is None else int(label_lag_frames)

    X, y, keys, columns, _ = load_or_build_training_dataset(
        dlc_dir,
        labels_dir,
        feature_mode=feature_mode,
        cache_dir=feature_cache_dir,
        force_rebuild_cache=force_rebuild_cache,
        show_progress=False,
    )
    X, y, groups, _ = _apply_label_lag_to_dataset(X, y, np.asarray(keys), label_lag_frames=lag)

    # A score over videos the model was fitted on measures memory, not skill. The
    # payload records which videos went where, so say plainly which this is.
    # `groups_fitted` covers the refit; older payloads only recorded the training
    # split, which for them was the same thing.
    fitted_on = {str(g) for g in obj.get("groups_fitted", obj.get("groups_train", []))}
    evaluated = {str(g).split(":", 1)[-1] for g in np.unique(groups).tolist()}
    overlap = sorted(evaluated & {g.split(":", 1)[-1] for g in fitted_on})

    if max_eval_frames is not None and max_eval_frames > 0 and len(X) > max_eval_frames:
        keep = _sample_class_indices(y, max_frames=max_eval_frames, random_state=42)
        X = X[keep]
        y = y[keep]

    chunk_size = max(1, int(eval_chunk_size))
    probs_parts: List[np.ndarray] = []
    for start in range(0, len(X), chunk_size):
        chunk = _align_feature_columns(X[start : start + chunk_size], columns, feat_names)
        probs_parts.append(clf.predict_proba(chunk)[:, 1])
    probs = np.concatenate(probs_parts) if probs_parts else np.array([], dtype=float)

    m = _threshold_metrics(y, probs, t)
    has_fit_record = "groups_fitted" in obj or "groups_train" in obj
    return {
        "threshold": float(m.threshold),
        "precision": float(m.precision),
        "recall": float(m.recall),
        "f1": float(m.f1),
        # This module's own header argues F1 misleads on its own; ship the
        # base-rate-free scores next to it.
        "mcc": float(m.mcc),
        "balanced_accuracy": float(m.balanced_accuracy),
        "label_lag_frames": float(lag),
        "n_frames": float(len(y)),
        "max_eval_frames": float(max_eval_frames) if max_eval_frames else float(len(y)),
        # Empty means every video here was unseen during fitting, which is the
        # only case where these numbers describe performance on new data.
        "videos_also_used_for_fitting": overlap,
        # None means the file predates the fit records, so nobody can say.
        "is_out_of_sample": (not overlap) if has_fit_record else None,
    }
