"""Downloader integrity and safe extraction boundaries, without network access."""

import hashlib
import zipfile
from pathlib import Path

import pytest

from coping_dynamics import datasets


def archive(tmp_path, name="recording/data.csv"):
    path = tmp_path / "input.zip"
    with zipfile.ZipFile(path, "w") as target:
        info = zipfile.ZipInfo(name)
        info.filename = name  # Preserve hostile separators on Windows too.
        target.writestr(info, "frame,label\n0,Freeze\n")
    return path


def test_verified_existing_archive_is_reused_offline(tmp_path, monkeypatch):
    path = archive(tmp_path)
    entry = {
        "filename": path.name,
        "size": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    monkeypatch.setattr(datasets, "catalog", lambda: {"files": {"sample": entry}})
    monkeypatch.setattr(
        datasets.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used")
    )
    assert datasets.download("sample", tmp_path, unpack=True) == path
    assert (tmp_path / "sample/recording/data.csv").read_text().startswith("frame,label")
    with pytest.raises(FileExistsError):
        datasets.download("sample", tmp_path, unpack=True)


def test_checksum_failure_preserves_existing_file(tmp_path):
    path = archive(tmp_path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Checksum"):
        datasets.verify(path, {"filename": path.name, "size": len(before), "sha256": "0" * 64})
    assert path.read_bytes() == before


@pytest.mark.parametrize("name", ["../escape", "/escape", "C:/escape", "dir\\escape"])
def test_unsafe_archive_does_not_extract(tmp_path, name):
    path = archive(tmp_path, name)
    with pytest.raises(ValueError, match="Unsafe"):
        datasets.extract(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_symlink_archive_rejected(tmp_path):
    path = tmp_path / "link.zip"
    info = zipfile.ZipInfo("link")
    info.external_attr = 0o120777 << 16
    with zipfile.ZipFile(path, "w") as target:
        target.writestr(info, "../outside")
    with pytest.raises(ValueError, match="Unsafe"):
        datasets.extract(path, tmp_path / "out")


def test_catalog_has_only_figshare_archives():
    entries = datasets.catalog()["files"]
    assert len(entries) == 7
    for entry in entries.values():
        assert "url" not in entry  # Embargoed files must not expose access links.
        assert Path(entry["filename"]).name == entry["filename"]
        assert entry["size"] > 0


def test_embargoed_download_fails_before_network(tmp_path, monkeypatch):
    monkeypatch.setattr(
        datasets.urllib.request, "urlopen", lambda *a, **k: pytest.fail("network used")
    )
    with pytest.raises(ValueError, match="embargoed"):
        datasets.download("freezing_predictions", tmp_path)
