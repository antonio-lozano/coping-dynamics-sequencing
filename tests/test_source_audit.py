"""Source audit must align identifiers, detect ambiguity, and preserve inputs."""

import pandas as pd
import pytest

from scripts.audit_source_data import KINEMATICS, audit, read_frames, sha256


def write(tmp_path, name, rows):
    defaults = dict.fromkeys(KINEMATICS, 0.0) | {
        "group": "Control",
        "onset": "TRUE",
        "syllable": "4",
    }
    path = tmp_path / name
    pd.DataFrame([defaults | row for row in rows]).to_csv(path, index=False)
    return path


def test_aligns_exact_string_keys_not_row_order_and_preserves_inputs(tmp_path):
    rows = [{"name": "01.10", "frame_index": "0"}, {"name": "1.1", "frame_index": "0"}]
    external = write(tmp_path, "external.csv", rows)
    bundled = write(tmp_path, "bundled.csv", [rows[1] | {"onset": "True"}, rows[0]])
    hashes = sha256(external), sha256(bundled)
    result = audit(external, bundled)
    assert result["matched_frames"] == 2
    assert result["inputs"]["external"]["recordings"] == 2
    assert result["label_differences"] == dict(group=0, onset=0, syllable=0)
    assert hashes == (sha256(external), sha256(bundled))
    assert result["input_hashes_unchanged"]


def test_missing_extra_keys_labels_remaps_and_kinematics(tmp_path):
    external = write(
        tmp_path,
        "external.csv",
        [{"name": "a", "frame_index": "0"}, {"name": "a", "frame_index": "1"}],
    )
    bundled = write(
        tmp_path,
        "bundled.csv",
        [
            {
                "name": "a",
                "frame_index": "0",
                "group": "ELS",
                "onset": "False",
                "syllable": "111",
                "heading": 0.2,
            },
            {"name": "b", "frame_index": "0"},
        ],
    )
    result = audit(external, bundled)
    assert result["external_keys_not_bundled"]["count"] == 1
    assert result["bundled_keys_not_external"]["count"] == 1
    assert result["label_differences"] == dict(group=1, onset=1, syllable=1)
    assert result["syllable_remaps"] == [{"external": "4", "bundled": "111", "frames": 1}]
    assert result["kinematics"]["heading"]["max_absolute_difference_finite"] == 0.2


def test_duplicate_keys_block_comparison_instead_of_multiplying_rows(tmp_path):
    row = {"name": "a", "frame_index": "0"}
    external = write(tmp_path, "external.csv", [row, row])
    bundled = write(tmp_path, "bundled.csv", [row])
    result = audit(external, bundled)
    assert result["comparison_status"] == "BLOCKED_DUPLICATE_KEYS"
    assert result["duplicate_keys"]["external"]["count"] == 1
    assert "matched_frames" not in result


def test_nonfinite_values_do_not_disappear_from_comparison(tmp_path):
    row = {"name": "a", "frame_index": "0"}
    external = write(tmp_path, "external.csv", [row | {"heading": float("nan")}])
    bundled = write(tmp_path, "bundled.csv", [row])
    result = audit(external, bundled)
    assert result["kinematics"]["heading"]["nonfinite_mismatch_frames"] == 1
    assert result["kinematics"]["heading"]["max_absolute_difference_finite"] is None


def test_invalid_boolean_rejected(tmp_path):
    path = write(tmp_path, "invalid.csv", [{"name": "a", "frame_index": "0", "onset": "maybe"}])
    with pytest.raises(ValueError, match="Non-boolean onset"):
        read_frames(path)


def test_cli_never_overwrites_existing_evidence(tmp_path, monkeypatch):
    from scripts.audit_source_data import main

    output = tmp_path / "evidence.json"
    output.write_text("preserve this evidence\n")
    monkeypatch.setattr(
        "sys.argv",
        ["audit", "--external", "missing.csv", "--bundled", "missing.csv", "--output", str(output)],
    )
    with pytest.raises(SystemExit) as failure:
        main()
    assert failure.value.code == 2
    assert output.read_text() == "preserve this evidence\n"
