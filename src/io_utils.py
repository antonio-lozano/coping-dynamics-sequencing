"""Lightweight I/O helpers for loading pickles and metadata."""
import pickle
import pandas as pd


def load_pickle(filepath):
    """Load a pickle file from disk."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def load_group_assignments(csv_path, name_col="name", group_col="group"):
    """Return mapping of recording name -> group label."""
    df = pd.read_csv(csv_path)
    return dict(zip(df[name_col], df[group_col]))


def normalize_name(name: str) -> str:
    """Normalize recording names for matching across sources."""
    return str(name).strip().replace(" ", "_")
