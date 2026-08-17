import numpy as np

from freezing_dlc.summarize import summarize_predictions


def test_summarize_bins():
    pred = np.array([1, 0, 1, 1, 0, 0])
    totals, bins = summarize_predictions(pred, "vid1", fps=2, bin_sec=2)
    assert totals.loc[0, "freezing_pct"] == 100 * pred.mean()
    assert len(bins) == 2
    assert bins.loc[0, "bin"] == 0
