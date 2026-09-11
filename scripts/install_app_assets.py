"""Install an author-supplied Behavior Studio archive without publishing embargoed assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[1]


def install(archive: Path, root: Path = REPO, manifest_path: Path | None = None) -> int:
    manifest_path = manifest_path or root / "apps/behavior-dlc-classifier/assets-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prefix = "apps/behavior-dlc-classifier/"
    with zipfile.ZipFile(archive) as source, tempfile.TemporaryDirectory() as staging:
        names = source.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest):
            raise ValueError("Archive file inventory does not match the approved manifest")
        pending = []
        for name, expected in manifest.items():
            relative = PurePosixPath(name)
            if (
                not name.startswith(prefix)
                or relative.is_absolute()
                or ".." in relative.parts
                or "\\" in name
                or ":" in name
            ):
                raise ValueError("Unsafe asset path")
            target = root / name
            if any(p.is_symlink() for p in [target, *target.parents]):
                raise ValueError("Refusing a symbolic-link destination")
            data = source.read(name)
            if (
                len(data) != expected["size"]
                or hashlib.sha256(data).hexdigest() != expected["sha256"]
            ):
                raise ValueError(f"Asset checksum mismatch: {name}")
            if target.exists():
                if (
                    not target.is_file()
                    or hashlib.sha256(target.read_bytes()).hexdigest() != expected["sha256"]
                ):
                    raise FileExistsError(f"Refusing to overwrite: {target}")
                continue
            staged = Path(staging) / name
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(data)
            pending.append((staged, target))
        # Validate the complete archive and all destinations before creating files.
        created = []
        try:
            for staged, target in pending:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    created.append(target)
                    output.write(staged.read_bytes())
        except Exception:
            for target in created:
                target.unlink()
            raise
    return len(pending)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    try:
        count = install(args.archive)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(1, f"Asset installation failed: {error}\n")
    print(f"Installed {count} verified assets; existing matching assets preserved.")


if __name__ == "__main__":
    main()
