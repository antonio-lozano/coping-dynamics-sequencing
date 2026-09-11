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


def test_legacy_single_animal_config_renders_without_changing_source(tmp_path: Path):
    from freezing_dlc.dlc_stage import prepare_dlc_config

    config = tmp_path / 'config.yaml'
    original = f'project_path: {tmp_path}\nbodyparts:\n- nose\niteration: 1\n'
    config.write_text(original)
    runtime = prepare_dlc_config(config)
    assert runtime != config
    assert 'multianimalproject: false\n' in runtime.read_text()
    assert 'iteration: 1\n' in runtime.read_text()
    assert config.read_text() == original


def test_explicit_multianimal_config_is_not_reclassified(tmp_path: Path):
    from freezing_dlc.dlc_stage import prepare_dlc_config

    config = tmp_path / 'config.yaml'
    original = 'project_path: .\nmultianimalproject: true\nbodyparts: []\n'
    config.write_text(original)
    runtime = prepare_dlc_config(config)
    assert 'multianimalproject: true\n' in runtime.read_text()
    assert 'multianimalproject: false' not in runtime.read_text()
    assert config.read_text() == original
