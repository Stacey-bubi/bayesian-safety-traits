from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "generation_id",
    "prompt_id",
    "domain",
    "outcome_family",
    "source",
    "steering_level",
    "steering_layer",
    "replicate_id",
    "refused_judge",
]


def load_data(path: str | Path, label_col: str = "refused_judge") -> pd.DataFrame:
    """Load a generations CSV and validate its schema.

    Args:
        path: CSV with one row per generation; must contain REQUIRED_COLUMNS.
        label_col: column copied into a canonical `refused` 0/1 label
            (e.g. "refused_judge" or "refused_rule").

    Returns:
        The loaded DataFrame with an added `refused` column.
    """
    df = pd.read_csv(path)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    assert not missing, f"missing required columns: {missing}"
    assert df["steering_level"].nunique() > 1, "steering_level does not vary"
    assert df["outcome_family"].notna().all(), "outcome_family missing for some rows"

    cell_counts = df.groupby(["prompt_id", "steering_level"]).size()
    assert (cell_counts >= 5).all(), "some (prompt_id, steering_level) cell has <5 replicates"

    alpha_sets = df.groupby("prompt_id")["steering_level"].apply(lambda s: frozenset(s.unique()))
    assert alpha_sets.nunique() == 1, "alpha levels are not identical across all prompts"

    df["refused"] = df[label_col]
    return df


def leave_prompts_out(df: pd.DataFrame, frac: float = 0.2, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by holding out a fraction of prompts within each domain.

    Args:
        df: output of load_data().
        frac: fraction of prompts per domain to hold out for testing.
        seed: RNG seed for which prompts get held out.

    Returns:
        (train_df, test_df).
    """
    rng = np.random.default_rng(seed)
    held: set[str] = set()
    for _, group in df.groupby("domain"):
        prompts = group["prompt_id"].unique()
        n_held = max(1, round(len(prompts) * frac))
        held.update(rng.choice(prompts, size=n_held, replace=False))
    test = df[df["prompt_id"].isin(held)]
    train = df[~df["prompt_id"].isin(held)]
    return train, test


def leave_one_domain_out(df: pd.DataFrame, domain: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by holding out one entire domain.

    Args:
        df: output of load_data().
        domain: value from the `domain` column to hold out for testing.

    Returns:
        (train_df, test_df).
    """
    test = df[df["domain"] == domain]
    train = df[df["domain"] != domain]
    return train, test
