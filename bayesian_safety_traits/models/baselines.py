from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .base import RefusalModel


class RawRate(RefusalModel):
    """No-model baseline: refusal fraction per (prompt, alpha) cell, falling back
    to (domain, alpha) then the global rate for rows unseen during fit."""

    def __init__(self) -> None:
        self.cell_rate: dict[tuple[str, float], float] = {}
        self.domain_rate: dict[tuple[str, float], float] = {}
        self.global_rate: float = 0.5

    def fit(self, train_df: pd.DataFrame) -> "RawRate":
        """Args: train_df must have `refused`, `prompt_id`, `domain`, `steering_level`."""
        y = train_df["refused"]
        self.cell_rate = y.groupby([train_df["prompt_id"], train_df["steering_level"]]).mean().to_dict()
        self.domain_rate = y.groupby([train_df["domain"], train_df["steering_level"]]).mean().to_dict()
        self.global_rate = float(y.mean())
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        """Args: rows must have `prompt_id`, `domain`, `steering_level`."""
        out = np.empty(len(rows))
        for i, row in enumerate(rows.itertuples()):
            cell_key = (row.prompt_id, row.steering_level) # ty: ignore
            domain_key = (row.domain, row.steering_level) # ty: ignore
            out[i] = self.cell_rate.get(cell_key, self.domain_rate.get(domain_key, self.global_rate))
        return out


class PlainLogistic(RefusalModel):
    """refused ~ steering_level + domain + outcome_family, with complete pooling
    over prompts (no per-prompt term: prompts don't matter, only category does)."""

    def __init__(self) -> None:
        self._clf = LogisticRegression(max_iter=1000)
        self._columns: list[str] = []

    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        # drop_first=False: an unseen domain/family at predict time gets all-zero
        # dummies, i.e. no domain signal, instead of silently aliasing to whatever
        # category drop_first would have picked as the reference
        design = pd.get_dummies(df[["domain", "outcome_family"]], drop_first=False)
        design.insert(0, "steering_level", df["steering_level"].to_numpy())
        return design

    def fit(self, train_df: pd.DataFrame) -> "PlainLogistic":
        """Args: train_df must have `refused`, `domain`, `outcome_family`, `steering_level`."""
        design = self._design(train_df)
        self._columns = design.columns.tolist()
        self._clf.fit(design, train_df["refused"])
        return self

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        """Args: rows must have `domain`, `outcome_family`, `steering_level`."""
        design = self._design(rows).reindex(columns=self._columns, fill_value=0)
        return self._clf.predict_proba(design)[:, 1]
