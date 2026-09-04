"""Print the project page to ``docs/web/coping-dynamics-sequencing.pdf``.

The page is assembled exactly as ``.github/workflows/pages.yml`` assembles it,
served over loopback, and printed by headless Chromium through Playwright. The
print rules live in the page's own ``@media print`` block, so the PDF and the
site never drift apart.

This is an authoring tool, not part of the rebuild: the PDF is committed, and
``docs/`` sits outside the artifact directories hashed by ``MANIFEST.csv``.
Playwright is not a project dependency, so run it on demand:

    uv run --with playwright python scripts/build_page_pdf.py

It uses the system Chrome if one is installed, and otherwise the browser from
``playwright install chromium``.
"""

from __future__ import annotations

import functools
import http.server
import shutil
import socketserver
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "web" / "coping-dynamics-sequencing.pdf"
FIGURES = ("figure3.pdf", "figure4.pdf", "figure5.pdf", "figure6.pdf")
VIDEO = "Supplementary_Video_1_MoSeq_syllable_atlas_grid.mp4"


def assemble(site: Path) -> None:
    """Mirror the CI site layout into ``site``."""
    shutil.copytree(ROOT / "docs" / "web", site, dirs_exist_ok=True)
    (site / "media").mkdir(parents=True, exist_ok=True)
    (site / "figures").mkdir(parents=True, exist_ok=True)
    for asset in sorted((ROOT / "docs" / "media").glob("*.gif")):
        shutil.copy2(asset, site / "media" / asset.name)
    for asset in sorted((ROOT / "docs" / "media").glob("*.png")):
        shutil.copy2(asset, site / "media" / asset.name)
    for name in FIGURES:
        shutil.copy2(ROOT / "figures" / name, site / "figures" / name)
    video = ROOT / "supplementary_media" / VIDEO
    if video.exists():
        shutil.copy2(video, site / "media" / VIDEO)


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Serve the site without logging, and without shouting when the browser
    aborts a request it no longer needs (the video, which print hides)."""

    def log_message(self, fmt: str, *args: object) -> None:
        pass

    def handle_one_request(self) -> None:
        try:
            super().handle_one_request()
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            self.close_connection = True


def serve(site: Path) -> tuple[socketserver.TCPServer, int]:
    handler = functools.partial(QuietHandler, directory=str(site))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def print_page(url: str, output: Path) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as play:
        try:
            browser = play.chromium.launch(channel="chrome")
        except Exception:
            browser = play.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.goto(url, wait_until="networkidle")
        page.emulate_media(media="print")
        page.pdf(
            path=str(output),
            format="A4",
            print_background=True,
            prefer_css_page_size=False,
            margin={"top": "14mm", "bottom": "14mm", "left": "14mm", "right": "14mm"},
        )
        browser.close()


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        site = Path(tmp) / "_site"
        assemble(site)
        httpd, port = serve(site)
        try:
            print_page(f"http://127.0.0.1:{port}/index.html", OUTPUT)
        finally:
            httpd.shutdown()
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"wrote {OUTPUT.relative_to(ROOT).as_posix()} ({size_kb:.0f} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
