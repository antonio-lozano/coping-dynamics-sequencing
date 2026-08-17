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
