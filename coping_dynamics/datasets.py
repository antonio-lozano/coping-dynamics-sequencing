"""Download the study's Figshare archives without changing bundled analysis inputs.

Usage: python -m coping_dynamics.datasets --list
       python -m coping_dynamics.datasets freezing_predictions --output ../coping-data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


def catalog() -> dict:
    return json.loads(Path(__file__).with_name("datasets.json").read_text(encoding="utf-8"))


def verify(path: Path, entry: dict) -> str:
    if path.stat().st_size != entry["size"]:
        raise ValueError(f"Unexpected size for {entry['filename']}; download is incomplete")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    if entry.get("sha256") and value != entry["sha256"]:
        raise ValueError(f"Checksum mismatch for {entry['filename']}")
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Corrupt ZIP member: {bad}")
    return value


def extract(path: Path, destination: Path) -> None:
    """Extract into a new directory, rejecting traversal and symbolic links."""
    if destination.exists():
        raise FileExistsError(f"Extraction destination already exists: {destination}")
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            if (
                name.is_absolute()
                or ".." in name.parts
                or "\\" in member.filename
                or ":" in member.filename
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            archive.extractall(temporary)
            # Rename only after the complete archive is extracted.
            Path(temporary).rename(destination)


def download(name: str, output: Path, *, unpack: bool = False) -> Path:
    entry = catalog()["files"][name]
    output.mkdir(parents=True, exist_ok=True)
    destination = output / entry["filename"]
    if destination.is_symlink():
        raise ValueError(f"Refusing symbolic link: {destination}")
    if destination.exists():
        digest = verify(destination, entry)
        print(f"Using verified archive: {destination}")
    else:
        if not entry.get("url"):
            raise ValueError(
                "Dataset is embargoed. Obtain the archive from the authors and place it "
                "in --output, then rerun to verify/extract. No private access link is shipped."
            )
        with tempfile.NamedTemporaryFile(dir=output, suffix=".part", delete=False) as target:
            temporary = Path(target.name)
            try:
                request = urllib.request.Request(
                    entry["url"], headers={"User-Agent": "CopingDynamics/1.0"}
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 200:
                        raise ValueError(f"Figshare returned HTTP {response.status}")
                    shutil.copyfileobj(response, target, length=1024 * 1024)
                target.flush()
                digest = verify(temporary, entry)
                # Exclusive destination creation: never overwrite a user's file.
                os.link(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        print(f"Downloaded: {destination}")
    print(f"SHA256 {digest}")
    if unpack:
        extracted = output / name
        extract(destination, extracted)
        print(f"Extracted: {extracted}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help="Archive names from --list, or all")
    parser.add_argument("--list", action="store_true", help="List files and download sizes")
    parser.add_argument("--output", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--extract", action="store_true", help="Extract each archive into a new directory"
    )
    args = parser.parse_args()
    files = catalog()["files"]
    if args.list:
        for name, entry in files.items():
            print(f"{name:22} {entry['size'] / 1024**2:9.2f} MiB")
        return
    names = list(files) if args.datasets == ["all"] else args.datasets
    if not names or any(name not in files for name in names):
        parser.error("Choose archive names from --list, or all")
    try:
        for name in dict.fromkeys(names):
            download(name, args.output.resolve(), unpack=args.extract)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        parser.exit(1, f"Download failed: {exc}\n")


if __name__ == "__main__":
    main()
