"""Shared timecourse figures follow the experimental tone protocol."""


def test_tone_shading_uses_protocol_time_not_bin_endpoints():
    from scripts.generate_figures.figure_5_resilience_dynamics import EVENT_SPANS

    # 180s baseline, three 30s tones, 60s between tones.
    assert EVENT_SPANS == [(3.0, 3.5), (4.5, 5.0), (6.0, 6.5)]


def test_all_timecourse_figures_share_one_protocol():
    from coping_dynamics.protocol import TONE_SPANS_MIN
    from scripts.generate_figures import figure_2_validation as f2
    from scripts.generate_figures import figure_3_behavior_clusters as f3
    from scripts.generate_figures import figure_5_resilience_dynamics as f5
    from scripts.generate_figures import figure_7_resilience_prediction as f7
    from scripts.generate_figures import supplementary_figure_1_tracking_clusters as s1

    assert list(TONE_SPANS_MIN) == [(3.0, 3.5), (4.5, 5.0), (6.0, 6.5)]
    assert f5.EVENT_SPANS == f7.EVENT_SPANS == list(TONE_SPANS_MIN)
    assert f2.EVENT_SPAN_STARTS_MIN == [a for a, b in TONE_SPANS_MIN]

    class Axis:
        def __init__(self):
            self.spans = []

        def axvspan(self, a, b, **kwargs):
            self.spans.append((a, b))

    for mod in (f3, s1):
        ax = Axis()
        mod.add_shock_shading(ax)
        assert ax.spans == list(TONE_SPANS_MIN)
