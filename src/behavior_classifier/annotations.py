"""Hand-curated annotations for Figure 7A-C and Supplementary Figure 4."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


BEHAVIOR_CLUSTER_MAP: dict[str, list[int]] = {
    "Freeze": [0, 28],
    "Sniff": [18, 20],
    "Groom": [24],
    "Turn": [1, 3, 5, 6, 10, 15, 26, 27],
    "Locomotion": [11, 12, 14, 16, 19, 21, 25],
    "Climb": [111],
    "Jump": [23, 29, 30, 34],
}

BEHAVIOR_ORDER: list[str] = [
    "Freeze",
    "Sniff",
    "Groom",
    "Turn",
    "Locomotion",
    "Climb",
    "Jump",
    "Unassigned",
]


def behavior_lookup(
    cluster_map: Mapping[str, Sequence[int]] = BEHAVIOR_CLUSTER_MAP,
) -> dict[int, str]:
    """Return a syllable -> behavior lookup and reject duplicate annotations."""

    out: dict[int, str] = {}
    duplicates: dict[int, list[str]] = {}
    for behavior, syllables in cluster_map.items():
        for syllable in syllables:
            syllable = int(syllable)
            if syllable in out:
                duplicates.setdefault(syllable, [out[syllable]]).append(behavior)
            out[syllable] = behavior

    if duplicates:
        details = ", ".join(f"{s}: {labels}" for s, labels in sorted(duplicates.items()))
        raise ValueError(f"Duplicate syllable annotations found: {details}")

    return out
