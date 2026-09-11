import hashlib
import json
import zipfile

import pytest

from scripts.install_app_assets import install


def fixture(tmp_path):
    name = "apps/behavior-dlc-classifier/models/test.bin"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({name: {"size": 4, "sha256": hashlib.sha256(b"data").hexdigest()}})
    )
    archive = tmp_path / "assets.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(name, b"data")
    return archive, manifest, tmp_path / name


def test_install_and_idempotent_reuse(tmp_path):
    archive, manifest, target = fixture(tmp_path)
    assert install(archive, tmp_path, manifest) == 1
    assert target.read_bytes() == b"data"
    assert install(archive, tmp_path, manifest) == 0


def test_existing_different_data_never_overwritten(tmp_path):
    archive, manifest, target = fixture(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"user data")
    with pytest.raises(FileExistsError):
        install(archive, tmp_path, manifest)
    assert target.read_bytes() == b"user data"


def test_corrupt_archive_installs_nothing(tmp_path):
    archive, manifest, target = fixture(tmp_path)
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("apps/behavior-dlc-classifier/models/test.bin", b"evil")
    with pytest.raises(ValueError, match="checksum"):
        install(archive, tmp_path, manifest)
    assert not target.exists()
