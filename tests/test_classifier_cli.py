"""Experimental training defaults must preserve the archived classifier."""

from coping_dynamics.behavior_classifier.cli import DEFAULT_MODEL, build_parser


def test_training_defaults_do_not_overwrite_archived_model_or_tables():
    parser = build_parser()
    for args in (["train"], ["train-from-dlc", "--dlc-dir", "tracks", "--labels-dir", "labels"]):
        parsed = parser.parse_args(args)
        assert parsed.model != DEFAULT_MODEL
        assert parsed.model.parent.name == "outputs"
        assert parsed.metrics.parent.name == "outputs"
        assert parsed.confusion.parent.name == "outputs"
    assert parser.parse_args(["predict"]).model == DEFAULT_MODEL
