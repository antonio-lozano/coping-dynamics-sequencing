"""Opt-in real-Tk regressions requiring a desktop or an explicit virtual display."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("BEHAVIOR_STUDIO_NATIVE_TESTS") != "1",
    reason="Requires a display and BEHAVIOR_STUDIO_NATIVE_TESTS=1",
)


def test_small_window_reveals_keyboard_controls():
    from freezing_dlc.gui import FreezingDlcApp

    app = FreezingDlcApp()
    try:
        app.geometry("800x560")
        app.update()
        # Opening the app must retain the title and left-hand input labels.
        assert app._canvas.xview()[0] == 0
        assert app._canvas.yview()[0] == 0
        for page in range(4):
            app.notebook.select(page)
            app.update()
            pending = list(app.nametowidget(app.notebook.select()).winfo_children())
            while pending:
                widget = pending.pop(0)
                pending.extend(widget.winfo_children())
                if widget.winfo_class() not in (
                    "TEntry",
                    "TButton",
                    "TCheckbutton",
                    "TRadiobutton",
                ):
                    continue
                widget.focus_force()
                app.update()
                canvas = app._canvas
                x = widget.winfo_rootx() - canvas.winfo_rootx()
                y = widget.winfo_rooty() - canvas.winfo_rooty()
                # Oversized path entries retain an editable visible leading portion.
                assert 0 <= x <= canvas.winfo_width() - 32
                assert 0 <= y <= canvas.winfo_height() - widget.winfo_height()
                if widget.winfo_width() <= canvas.winfo_width() - 16:
                    assert x + widget.winfo_width() <= canvas.winfo_width()
        app.notebook.select(0)
        app.run_button.focus_force()
        app.update()
        app.run_button.event_generate("<Control-Tab>")
        app.update()
        assert app.notebook.index("current") == 1
    finally:
        app.destroy()


def test_results_focus_does_not_shift_visible_leading_edge():
    from freezing_dlc.gui import FreezingDlcApp

    app = FreezingDlcApp()
    try:
        app.geometry("1100x820")
        app.notebook.select(app.results_page)
        app.update()
        app._canvas.xview_moveto(0)
        app.update()
        app.results_tree.focus_force()
        app.update()
        assert app._canvas.xview()[0] == 0
    finally:
        app.destroy()


def test_analysis_button_completes_real_package_and_closes_when_idle(tmp_path, monkeypatch):
    """Real Tk + worker + inference + exports; no file-dialog or player claim."""
    import hashlib
    import time
    import tkinter as tk
    from pathlib import Path

    import cv2
    import pandas as pd
    from freezing_dlc.gui import FreezingDlcApp

    root = Path(__file__).resolve().parents[1]
    source = root / "data/raw/original_videos"
    tracking = sorted(source.glob("*filtered.csv"))[0]
    video = source / (tracking.name.split("DLC")[0] + ".mp4")
    model = root / "models/freezing_model.sav"
    originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (tracking, video, model)}
    project = tmp_path / "recordings with spaces"
    project.mkdir()
    # Match video and tracking frame counts, preserving the real tracking schema.
    frames = 50
    lines = tracking.read_text(encoding="utf-8").splitlines()[: frames + 3]
    assert len(lines) == frames + 3
    (project / tracking.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    capture = cv2.VideoCapture(str(video))
    writer = cv2.VideoWriter(
        str(project / video.name),
        cv2.VideoWriter_fourcc(*"mp4v"),
        25,
        (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    )
    try:
        assert capture.isOpened() and writer.isOpened()
        for _ in range(frames):
            ok, frame = capture.read()
            assert ok
            writer.write(frame)
    finally:
        capture.release()
        writer.release()

    errors = []
    monkeypatch.setattr(
        "freezing_dlc.gui.messagebox.showerror", lambda *args, **kwargs: errors.append(args)
    )
    app = FreezingDlcApp()
    closed = False
    try:
        app.project_dir.set(str(project))
        app.model_path.set(str(model))
        app.workflow.set("existing")
        app._apply_workflow()
        app.update()
        app.run_button.invoke()
        assert app._busy_count == 1
        assert str(app.run_button["state"]) == "disabled"
        deadline = time.monotonic() + 90
        while app._busy_count and time.monotonic() < deadline:
            app.update()
            time.sleep(0.02)
        assert not errors, errors
        assert app._busy_count == 0, app.log_text.get("1.0", "end")
        app.worker.join(timeout=5)
        assert not app.worker.is_alive()
        assert str(app.run_button["state"]) == "normal"
        assert app.notebook.select() == str(app.results_page)
        package = app.last_output_dir
        assert package is not None and package.is_dir()
        listed = {
            app.results_tree.item(item, "values")[0] for item in app.results_tree.get_children()
        }
        for name in ("freezing_FINAL_results.xlsx", "behavior_FINAL_results.xlsx"):
            assert f"06_results/{name}" in listed
            with pd.ExcelFile(package / "06_results" / name) as workbook:
                assert workbook.sheet_names
        exported = list(package.rglob("*.mp4"))
        assert len(exported) >= 3
        for path in exported:
            capture = cv2.VideoCapture(str(path))
            count = 0
            try:
                while capture.read()[0]:
                    count += 1
            finally:
                capture.release()
            assert count == frames, path
        assert "Analysis complete" in app.stage_text.get()
        assert all(
            hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in originals.items()
        )
        # Invoke the actual window-manager close handler, not destroy directly.
        app.tk.call(app.protocol("WM_DELETE_WINDOW"))
        closed = True
        with pytest.raises(tk.TclError, match="application has been destroyed"):
            app.winfo_exists()
    finally:
        if not closed:
            app.destroy()
