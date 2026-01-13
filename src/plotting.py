"""Plotting helpers (placeholder for shared styles)."""
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def save(fig, path, dpi=300):
    """Save figure to both PNG and PDF."""
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
