#!/usr/bin/env python
"""Save an openpyxl workbook with run-independent bytes."""
from __future__ import annotations

import datetime
import re
import shutil
import tempfile
import zipfile
from pathlib import Path


# The earliest instant a zip container can store. Stamping every archive entry
# and the document properties with the same fixed instant makes rebuilding an
# unchanged workbook a byte-level no-op, so MANIFEST.csv can treat the report
# workbooks like every other regenerated artifact.
FIXED_INSTANT = datetime.datetime(1980, 1, 1)
_CORE_PROPS = "docProps/core.xml"
_STAMP = FIXED_INSTANT.strftime("%Y-%m-%dT%H:%M:%SZ")
_DCTERMS = re.compile(
    r"(<dcterms:(created|modified)[^>]*>)[^<]*(</dcterms:\2>)"
)


def save_workbook(workbook, output: Path) -> None:
    """Write ``workbook`` to ``output`` so identical content gives identical bytes.

    openpyxl stamps ``docProps/core.xml`` with the wall-clock save time (it
    overwrites ``properties.modified`` inside ``save``, so setting it up front
    does not help) and gives every zip entry the current local time. Rewriting
    the container afterwards fixes both; worksheet content itself serializes
    deterministically.
    """
    workbook.save(output)
    _rewrite_container(Path(output))


def _rewrite_container(path: Path) -> None:
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".zip") as handle:
        temp = Path(handle.name)
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(
        temp, "w", zipfile.ZIP_DEFLATED
    ) as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == _CORE_PROPS:
                text = _DCTERMS.sub(rf"\g<1>{_STAMP}\g<3>", data.decode("utf-8"))
                data = text.encode("utf-8")
            stamped = zipfile.ZipInfo(info.filename, date_time=FIXED_INSTANT.timetuple()[:6])
            stamped.compress_type = info.compress_type
            stamped.external_attr = info.external_attr
            target.writestr(stamped, data)
    shutil.move(str(temp), str(path))
