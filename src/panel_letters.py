# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jeniffer Sanguino Gomez and Antonio Lozano
"""Panel-letter placement shared by the figure generators.

Kept out of ``src.plotting`` on purpose: that module calls
``sns.set(style="whitegrid")`` at import time, which would silently repaint
grid lines onto any figure that imported it just for this helper.  Nothing here
touches global matplotlib or seaborn state.
"""
from __future__ import annotations

import matplotlib.pyplot as plt


def align_panel_letters(
    fig: plt.Figure,
    letter_artists: list[tuple[plt.Axes, plt.Text]],
    y_pad: float = 0.012,
) -> None:
    """Put every panel letter in its own panel's y-axis title column.

    Panel letters are normally placed in axes coordinates, so a fixed offset
    lands in a different place on every panel: the gap between the axes and its
    y-label depends on how wide that panel's tick labels are.  This re-anchors
    each letter in figure coordinates to the left edge of its y-axis title, and
    to the top of its own axes, so the letters line up with each other and with
    the titles they belong to.

    ``letter_artists`` is a sequence of ``(axes, text)`` pairs.  Call this after
    every panel is drawn - it needs a rendered canvas to measure against.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_inv = fig.transFigure.inverted()
    for ax, text in letter_artists:
        if text is None:
            continue
        label_box = ax.yaxis.label.get_window_extent(renderer=renderer)
        axes_box = ax.get_window_extent(renderer=renderer)
        # An empty y-label has no width and collapses onto the axes spine, so
        # fall back to the tick labels to keep the letter clear of the panel.
        if label_box.width <= 0:
            boxes = [
                t.get_window_extent(renderer=renderer)
                for t in ax.get_yticklabels()
                if t.get_text()
            ]
            x_px = min((b.x0 for b in boxes), default=axes_box.x0)
        else:
            x_px = label_box.x0
        x_fig = fig_inv.transform((x_px, axes_box.y0))[0]
        y_fig = fig_inv.transform((axes_box.x0, axes_box.y1))[1] + y_pad
        text.set_transform(fig.transFigure)
        text.set_position((x_fig, y_fig))
        text.set_ha("left")
        text.set_va("top")
