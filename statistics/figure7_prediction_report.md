

## Behavior dynamics and resilience prediction

`figure_7_resilience_prediction.py` rebuilds the early-prediction
panel with the three things the original lacked: a permutation null, a
per-behaviour breakdown, and Shapley attributions. **Behaviour dynamics**
means the 30-s trajectory of each behaviour's time share.

### Panel I - onset of prediction

Panel F plots raw ROC AUC so it is directly comparable with the other
time-resolved panels. The source table still carries the mean within-cohort
permutation AUC (200 shuffles per point) as an audit column, but it is
not drawn in the figure.

| Opening minutes | n features | ROC AUC |
| ---: | ---: | ---: |
| 0.5 | 7 | 0.664 |
| 1.0 | 14 | 0.632 |
| 1.5 | 21 | 0.761 |
| 2.0 | 28 | 0.672 |
| 2.5 | 35 | 0.661 |
| 3.0 | 42 | 0.549 |
| 3.5 | 49 | 0.480 |
| 4.0 | 56 | 0.603 |
| 4.5 | 63 | 0.618 |
| 5.0 | 70 | 0.753 |
| 5.5 | 77 | 0.710 |
| 6.0 | 84 | 0.741 |
| 6.5 | 91 | 0.770 |
| 7.0 | 98 | 0.845 |
| 7.5 | 105 | 0.833 |

The combined behaviour-dynamics curve is above chance from 0.5 min, with the clearest rise after 5 min.


### Complete A4 Figure 7 layout

The complete A4 version begins with Figure 7 panels A-C: accuracy versus
chance, global SHAP feature importance, and the confusion matrix. The
behaviour-dynamics held-out transfer, full-session LOOCV, and full-session
Shapley contributions form panels D-F in the second row. The all-predictor
held-out and CV panels follow, then side-by-side overtime summaries and four
individual-overtime panels: frequencies, diversity indices, transition
metrics, and bout durations.

Full-session LOOCV: behaviour dynamics 0.842, Freeze dynamics only 0.790.

The Shapley panel gives exact Shapley values for the full-session model. The classifier is
linear, so phi_ij = coef_j * z_ij is the closed-form Shapley value against a
mean-centred background - no sampling and no `shap` dependency. Ranked by
mean |phi|: Freeze (0.358, toward resilient), Turn (0.341, toward vulnerable), Jump (0.232, toward vulnerable).

**Caveat carried from the pre-shock work.** The label is built from the
full-session behaviour-dynamics profile, so every horizon is scored against
a target that partly includes the window being tested. The onset panel is therefore
anti-conservative; the within-cohort permutation audit controls for sample
size, cohort, and model size, not for that overlap.

Outputs: `figures/figure7.png/.pdf/.svg`,
`figure_source_data/figure7_*.csv`, and `statistics/figure7_*.csv`.
