# Equal-time ordering in bout calculations

Some source tables contain multiple syllable rows at the same animal/time bin. Bout calculations that sort only by time can depend on the order of tied rows. Alternative tie ordering changes bout lengths and derived contrasts; changing sort policy is therefore an analysis-method change, not a formatting adjustment.

When extending these analyses, record the row ordering and temporal resolution. Do not infer within-bin event order from a time-binned usage table. Compare regenerated outputs with the bundled references and retain differences for interpretation.

The read-only `scripts/diagnose_figure6_sorting.py` records sorting/runtime information without replacing reference results. See [reproduction instructions](../REPRODUCIBILITY.md).
