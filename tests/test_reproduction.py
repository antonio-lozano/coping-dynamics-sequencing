"""Behavioral contracts for isolated reproduction and frozen-reference checks."""

import errno
from pathlib import Path

import pytest

from coping_dynamics.reproduction import compare_csv, copy_checkout, summarize_run


def directory_alias(link, target):
    """Exercise link contracts where the filesystem permits real symlinks."""
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314 or exc.errno in {
            errno.EPERM,
            errno.EACCES,
            errno.ENOTSUP,
        }:
            pytest.skip(f"Directory symlink creation unavailable: {exc}")
        raise


def test_numeric_roundoff_is_distinguished_from_changed_science(tmp_path):
    reference = tmp_path / "reference.csv"
    candidate = tmp_path / "candidate.csv"
    reference.write_text("animal,p,value\na,0.049,1.0\n")
    candidate.write_text("animal,p,value\na,0.049,1.000000001\n")
    assert compare_csv(reference, candidate)["status"] == "numerically_equal"
    candidate.write_text("animal,p,value\na,0.051,1.0\n")
    assert compare_csv(reference, candidate)["status"] == "different"


def test_nan_and_identifier_changes_never_pass_as_roundoff(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("animal,value\n150.10,NaN\n")
    b.write_text("animal,value\n150.1,NaN\n")
    assert compare_csv(a, b)["status"] == "different"
    b.write_text("animal,value\n150.10,0\n")
    assert compare_csv(a, b)["status"] == "different"


def test_missing_rows_and_columns_fail(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("group,p\nControl,0.1\nELS,0.2\n")
    b.write_text("group,p\nControl,0.1\n")
    assert compare_csv(a, b)["status"] == "different"


def test_isolated_copy_cannot_overwrite_source_or_existing_destination(tmp_path):

    source = tmp_path / "source"
    source.mkdir()
    (source / "data.txt").write_text("immutable")
    with pytest.raises(ValueError):
        copy_checkout(source, source / "output", ["data.txt"])
    destination = tmp_path / "destination"
    copy_checkout(source, destination, ["data.txt"])
    (destination / "data.txt").write_text("regenerated")
    assert (source / "data.txt").read_text() == "immutable"
    with pytest.raises(FileExistsError):
        copy_checkout(source, destination, ["data.txt"])


def test_pipeline_success_does_not_hide_numeric_or_missing_artifacts():
    assert summarize_run([0], [{"status": "different", "kind": "csv"}], []) == "FAILED"
    assert summarize_run([0], [{"status": "missing", "kind": "figure"}], []) == "FAILED"
    assert summarize_run([1], [], []) == "FAILED"
    assert summarize_run([0], [], ["data/raw/input.csv"]) == "FAILED"
    assert (
        summarize_run([0], [{"status": "byte_different", "kind": "figure"}], [])
        == "REVIEW_REQUIRED"
    )
    assert summarize_run([0], [{"status": "numerically_equal", "kind": "csv"}], []) == "PASS"


def test_raw_companion_serialization_is_distinct_from_input_mutation(tmp_path):
    import gzip

    from coping_dynamics.reproduction import compare_raw_inputs

    a, b = tmp_path / "source", tmp_path / "candidate"
    for root in (a, b):
        (root / "data/raw").mkdir(parents=True)
    name = "data/raw/freezing_predictions_light.csv.gz"
    (a / name).write_bytes(gzip.compress(b"animal,value\r\na,1\r\n"))
    (b / name).write_bytes(gzip.compress(b"animal,value\na,1\n"))
    changed, serialization = compare_raw_inputs(a, b, [name])
    assert changed == [] and serialization == [name]
    (b / name).write_bytes(gzip.compress(b"animal,value\na,2\n"))
    assert compare_raw_inputs(a, b, [name])[0] == [name]


def test_primary_inputs_require_exact_bytes_even_with_line_ending_changes(tmp_path):
    from coping_dynamics.reproduction import compare_raw_inputs

    a, b = tmp_path / "source", tmp_path / "candidate"
    for root in (a, b):
        (root / "data/raw").mkdir(parents=True)
    name = "data/raw/primary.csv"
    (a / name).write_bytes(b"x\r\n1\r\n")
    (b / name).write_bytes(b"x\n1\n")
    assert compare_raw_inputs(a, b, [name])[0] == [name]


def test_failed_model_rows_are_not_successful_scientific_outputs(tmp_path):
    from coping_dynamics.reproduction import failed_models

    (tmp_path / "statistics").mkdir()
    p = tmp_path / "statistics/fits.csv"
    p.write_text("analysis,parameter,note\nCombined,model_failed,Singular matrix\n")
    failures = failed_models(tmp_path)
    assert len(failures) == 1
    assert failures[0]["path"] == "statistics/fits.csv"
    assert failures[0]["row"]["note"] == "Singular matrix"


def test_untouched_preexisting_outputs_are_not_counted_as_regenerated(tmp_path):
    from coping_dynamics.reproduction import observe_outputs

    output = tmp_path / "result.csv"
    output.write_text("x\n1\n")
    before = {"result.csv": output.stat().st_mtime_ns}
    assert observe_outputs(tmp_path, before)[0]["write_observed"] is False
    output.write_text("x\n2\n")
    # Do not depend on immediate writes advancing a filesystem clock (Windows).
    import os

    stamp = before["result.csv"] + 2_000_000_000
    os.utime(output, ns=(stamp, stamp))
    assert observe_outputs(tmp_path, before)[0]["write_observed"] is True
    output.unlink()
    assert observe_outputs(tmp_path, before)[0]["status"] == "missing"


def test_isolated_copy_preserves_internal_directory_alias(tmp_path):
    from coping_dynamics.reproduction import digest

    source = tmp_path / "source"
    (source / "web/site").mkdir(parents=True)
    (source / "docs").mkdir()
    (source / "web/site/index.html").write_text("preview")
    directory_alias(source / "docs/web", Path("../web/site"))
    # Windows stores link text verbatim: use native separators in the fixture.
    # Establish a working source alias before testing the isolation copy.
    assert (source / "docs/web/index.html").read_text() == "preview"
    before = digest(source / "docs/web")
    destination = tmp_path / "copy"
    copy_checkout(source, destination, ["docs/web", "web/site/index.html"])
    assert (destination / "docs/web").is_symlink()
    assert (destination / "docs/web/index.html").read_text() == "preview"
    assert digest(destination / "docs/web") == before
    (destination / "docs/web/index.html").write_text("changed")
    assert (source / "web/site/index.html").read_text() == "preview"


def test_isolated_copy_rejects_external_directory_alias_before_writing(tmp_path):

    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    directory_alias(source / "escape", Path("../outside"))
    destination = tmp_path / "copy"
    with pytest.raises(ValueError):
        copy_checkout(source, destination, ["escape"])
    assert not destination.exists()


def test_source_integrity_tracks_alias_and_target_changes_separately(tmp_path):
    from coping_dynamics.reproduction import changed_sources, digest

    (tmp_path / "assets").mkdir()
    target = tmp_path / "assets/input.csv"
    target.write_text("original")
    alias = tmp_path / "alias"
    directory_alias(alias, Path("assets"))
    names = ["alias", "assets/input.csv"]
    before = {name: digest(tmp_path / name) for name in names}
    assert changed_sources(tmp_path, before) == []
    target.write_text("changed")
    assert changed_sources(tmp_path, before) == ["assets/input.csv"]
    target.write_text("original")
    alias.unlink()
    directory_alias(alias, Path("missing"))
    assert changed_sources(tmp_path, before) == ["alias"]
    alias.unlink()
    assert changed_sources(tmp_path, before) == ["alias"]
    alias.mkdir()
    assert changed_sources(tmp_path, before) == ["alias"]
    target.unlink()
    assert changed_sources(tmp_path, before) == names


def test_comparison_summary_separates_cached_missing_and_unobserved_outputs():
    from coping_dynamics.reproduction import comparison_summary

    rows = [{"path": name, "status": "byte_equal"} for name in ("written", "cached", "unknown")]
    rows.append({"path": "missing", "status": "missing"})
    observations = [
        {"path": "written", "write_observed": True},
        {"path": "cached", "write_observed": False},
        {"path": "missing", "write_observed": False},
    ]
    assert comparison_summary(rows, observations) == {
        "write_observed": {"byte_equal": 1},
        "no_write_observed": {"byte_equal": 1, "missing": 1},
        "unknown": {"byte_equal": 1},
    }


def test_symlink_permission_denial_is_an_explicit_capability_skip(tmp_path, monkeypatch):
    def denied(*args, **kwargs):
        raise OSError(errno.EPERM, "symlinks unavailable")

    monkeypatch.setattr(Path, "symlink_to", denied)
    with pytest.raises(pytest.skip.Exception, match="Directory symlink creation unavailable"):
        directory_alias(tmp_path / "link", Path("target"))
