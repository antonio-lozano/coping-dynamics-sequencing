from pathlib import Path

from freezing_dlc.dlc_stage import _collect_filtered_csvs, find_source_video


def test_find_source_video_prefers_the_exact_stem(tmp_path: Path):
    """"Trial 1"'s tracking must not claim a lookalike when Trial 1.mp4 exists.

    Names that merely start the same way can sort ahead of the exact one --
    a space sorts before a dot -- so first-match-after-sort picked wrongly.
    """
    (tmp_path / "Trial 1 (2).mp4").write_bytes(b"x")
    (tmp_path / "Trial 1.mp4").write_bytes(b"x")

    found = find_source_video(tmp_path, "Trial 1DLC_resnet50_somethingfiltered")

    assert found.name == "Trial 1.mp4"


def test_find_source_video_still_accepts_a_prefix_when_no_exact_stem(tmp_path: Path):
    (tmp_path / "Trial 1_compressed.mp4").write_bytes(b"x")

    found = find_source_video(tmp_path, "Trial 1DLC_resnet50_somethingfiltered")

    assert found.name == "Trial 1_compressed.mp4"


def test_filtered_csvs_are_found_for_bracketed_video_names(tmp_path: Path):
    """Brackets are legal in filenames but are glob syntax unless escaped."""
    video = tmp_path / "Trial [A].mp4"
    video.write_bytes(b"x")
    csv = tmp_path / "Trial [A]DLC_resnet50filtered.csv"
    csv.write_text("a\n", encoding="utf-8")

    assert _collect_filtered_csvs([video]) == [csv]
