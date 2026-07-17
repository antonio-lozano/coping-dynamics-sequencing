from __future__ import annotations

import zipfile
from pathlib import Path

ZIP_PATH = Path("H:/Downloads/Coping_data2.zip")

PREFIXES = (
    "COping/Code/Results/Frequency_metrics/",
    "COping/Code/Results/Bout_duration/",
    "COping/Code/6.",
    "COping/Code/7.",
)


def main() -> None:
    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = zf.namelist()
        for prefix in PREFIXES:
            print(f"\n[{prefix}]")
            hits = [name for name in names if name.startswith(prefix)]
            for name in hits[:80]:
                print(name)
            if len(hits) > 80:
                print(f"... {len(hits) - 80} more")


if __name__ == "__main__":
    main()
