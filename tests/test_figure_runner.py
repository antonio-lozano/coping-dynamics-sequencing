"""Missing figure entry points must fail the all-figures workflow."""

import pytest

from scripts import run_all_figures


def test_missing_generator_is_a_failed_workflow(tmp_path, monkeypatch):
    monkeypatch.setattr(run_all_figures, "SCRIPTS_DIR", tmp_path)
    monkeypatch.setattr(run_all_figures, "FIGURE_SCRIPTS", [("required figure", "missing.py")])
    with pytest.raises(SystemExit) as error:
        run_all_figures.main()
    assert error.value.code == 1


def test_reproduction_requests_fresh_prediction_models(monkeypatch):
    calls = []
    monkeypatch.setattr(
        run_all_figures.subprocess, "run", lambda command, **kwargs: calls.append(command)
    )
    script = run_all_figures.SCRIPTS_DIR / "figure_7_resilience_prediction.py"
    result, _ = run_all_figures.run_script("Figure 7", script, recompute=True)
    assert result == "SUCCESS"
    assert calls[0][-1] == "--recompute"


def test_fresh_recap_ignores_existing_prediction_tables(tmp_path, monkeypatch):
    import pandas as pd

    from scripts.generate_figures import figure_7_resilience_prediction as f7

    for name in ("figure7_predictor_catalog.csv", "figure7_predictor_timecourse.csv"):
        (tmp_path / name).write_text("stale cache must not be read")
    monkeypatch.setattr(f7, "SOURCE_OUT", tmp_path)
    monkeypatch.setattr(f7.recap, "predictor_color", lambda value: "black")
    called = []

    def cached(name, builder, *, fresh):
        called.append((name, fresh))
        if name == "predictors":
            return object(), object()
        if name == "horizon_features":
            return object()
        return pd.DataFrame({"predictor": ["fresh"], "auc": [0.75]})

    monkeypatch.setattr(f7.recap, "cached", cached)
    a, b = f7.recap_tables(fresh=True)
    assert a.predictor.tolist() == b.predictor.tolist() == ["fresh"]
    assert len(called) == 4 and all(fresh for _, fresh in called)
