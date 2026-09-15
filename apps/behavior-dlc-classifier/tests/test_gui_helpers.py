from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from freezing_dlc.gui import FreezingDlcApp


class _Var:
    """Just enough of a tk.StringVar for the parser, without a Tk window."""

    def __init__(self, text: str) -> None:
        self._text = text

    def get(self) -> str:
        return self._text


def test_number_accepts_padded_numbers():
    assert FreezingDlcApp._number(_Var(" 25 "), "Frames per second") == 25.0


def test_number_error_names_the_field_and_the_value():
    with pytest.raises(ValueError, match="Frames per second must be a number; it is 'abc'"):
        FreezingDlcApp._number(_Var("abc"), "Frames per second")


def test_number_error_says_blank_for_empty_fields():
    with pytest.raises(ValueError, match="Time bin must be a number; it is blank"):
        FreezingDlcApp._number(_Var("   "), "Time bin")


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_number_rejects_nonfinite_settings(value):
    with pytest.raises(ValueError, match="finite"):
        FreezingDlcApp._number(_Var(value), "Frames per second")


@pytest.mark.parametrize("value", ["30.5", "0", "-1"])
def test_time_bin_does_not_silently_truncate(value):
    with pytest.raises(ValueError, match="positive whole"):
        FreezingDlcApp._time_bin(_Var(value))


def test_time_bin_accepts_integer_decimal():
    assert FreezingDlcApp._time_bin(_Var("30.0")) == 30


@pytest.mark.parametrize("active", ["worker", "manual_worker", "refine_training_worker"])
def test_close_keeps_active_work_alive(monkeypatch, active):
    app = SimpleNamespace(
        worker=None,
        manual_worker=None,
        refine_training_worker=None,
        _busy_count=0,
        destroy=Mock(),
    )
    setattr(app, active, SimpleNamespace(is_alive=lambda: True))
    notice = Mock()
    monkeypatch.setattr("freezing_dlc.gui.messagebox.showinfo", notice)
    FreezingDlcApp._request_close(app)
    app.destroy.assert_not_called()
    notice.assert_called_once()


def test_close_waits_for_pending_completion_event(monkeypatch):
    app = SimpleNamespace(
        worker=SimpleNamespace(is_alive=lambda: False),
        manual_worker=None,
        refine_training_worker=None,
        _busy_count=1,
        destroy=Mock(),
    )
    monkeypatch.setattr("freezing_dlc.gui.messagebox.showinfo", Mock())
    FreezingDlcApp._request_close(app)
    app.destroy.assert_not_called()


def test_close_exits_when_idle():
    app = SimpleNamespace(
        worker=None,
        manual_worker=None,
        refine_training_worker=None,
        _busy_count=0,
        destroy=Mock(),
    )
    FreezingDlcApp._request_close(app)
    app.destroy.assert_called_once()


@pytest.mark.parametrize("remaining", [0, 2])
def test_unsaved_annotation_cancels_refinement_without_training(monkeypatch, remaining):
    app = SimpleNamespace(
        refine_videos=["video"] * (remaining + 1),
        refine_next_index=1,
        _append_log=Mock(),
        _progress_end=Mock(),
        refinement_text=Mock(),
        refine_button=Mock(),
        _start_refine_training=Mock(),
        _start_next_refine_label=Mock(),
    )
    prompt = Mock()
    monkeypatch.setattr("freezing_dlc.gui.messagebox.askyesno", prompt)
    FreezingDlcApp._ask_continue_or_train(app, {"saved": False})
    app._start_refine_training.assert_not_called()
    app._start_next_refine_label.assert_not_called()
    prompt.assert_not_called()
    app.refine_button.configure.assert_called_once_with(state="normal")
    app._progress_end.assert_called_once()


def test_results_inventory_hides_working_files_and_preserves_relative_paths(tmp_path):
    from freezing_dlc.gui import result_files

    (tmp_path / "06_results").mkdir()
    (tmp_path / "06_results" / "results.xlsx").write_bytes(b"workbook")
    (tmp_path / "_working").mkdir()
    (tmp_path / "_working" / "temporary.csv").write_text("private work")
    (tmp_path / "package_summary.txt").write_text("summary")
    assert result_files(tmp_path) == [
        "06_results/results.xlsx",
        "package_summary.txt",
    ]


def test_results_inventory_does_not_follow_links_outside_package(tmp_path):
    from freezing_dlc.gui import result_files

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "unrelated.txt").write_text("unrelated")
    package = tmp_path / "package"
    package.mkdir()
    (package / "linked.txt").symlink_to(outside / "unrelated.txt")
    assert result_files(package) == []


def test_open_saved_package_does_not_replace_results_for_unrelated_folder(tmp_path, monkeypatch):
    app = SimpleNamespace(_show_results=Mock(), _append_log=Mock())
    monkeypatch.setattr("freezing_dlc.gui.filedialog.askdirectory", lambda **kw: str(tmp_path))
    notice = Mock()
    monkeypatch.setattr("freezing_dlc.gui.messagebox.showerror", notice)
    FreezingDlcApp._choose_results_package(app)
    app._show_results.assert_not_called()
    notice.assert_called_once()


def test_open_saved_package_browses_without_running_analysis(tmp_path, monkeypatch):
    (tmp_path / "package_summary.txt").write_text("Saved analysis")
    (tmp_path / "06_results").mkdir()
    app = SimpleNamespace(_show_results=Mock(), _append_log=Mock())
    monkeypatch.setattr("freezing_dlc.gui.filedialog.askdirectory", lambda **kw: str(tmp_path))
    FreezingDlcApp._choose_results_package(app)
    app._show_results.assert_called_once_with(tmp_path)


def test_refinement_saves_model_beside_user_labels_not_installed_package(tmp_path, monkeypatch):
    import queue

    labels = tmp_path / "manual_labels"
    labels.mkdir()
    app = SimpleNamespace(
        refine_dlc_dir=tmp_path, refine_labels_dir=labels, log_queue=queue.Queue()
    )
    train = Mock(return_value={"threshold": 0.4})
    monkeypatch.setattr("freezing_dlc.gui.train_model", train)
    FreezingDlcApp._refine_training_worker(app)
    assert train.call_args.kwargs["out_model"].parent == tmp_path / "refined_models"
    assert app.log_queue.get()[0] == "refine_model_done"


def test_saved_labels_can_retry_training_without_reopening_video(tmp_path, monkeypatch):
    labels = tmp_path / "manual_labels"
    labels.mkdir()
    (labels / "video_labels.csv").write_text("pred\n0\n1\n")
    app = SimpleNamespace(
        manual_worker=None,
        refine_training_worker=None,
        _find_filtered_dlc_dir=Mock(return_value=tmp_path),
        _start_refine_training=Mock(),
        _start_next_refine_label=Mock(),
    )
    monkeypatch.setattr("freezing_dlc.gui.filedialog.askdirectory", lambda **kw: str(tmp_path))
    FreezingDlcApp._train_saved_labels(app)
    assert app.refine_labels_dir == labels
    assert app.refine_dlc_dir == tmp_path
    app._start_refine_training.assert_called_once()
    app._start_next_refine_label.assert_not_called()


def test_saved_labels_refuses_missing_labels_without_training(tmp_path, monkeypatch):
    app = SimpleNamespace(
        manual_worker=None, refine_training_worker=None, _start_refine_training=Mock()
    )
    monkeypatch.setattr("freezing_dlc.gui.filedialog.askdirectory", lambda **kw: str(tmp_path))
    notice = Mock()
    monkeypatch.setattr("freezing_dlc.gui.messagebox.showerror", notice)
    FreezingDlcApp._train_saved_labels(app)
    app._start_refine_training.assert_not_called()
    notice.assert_called_once()
