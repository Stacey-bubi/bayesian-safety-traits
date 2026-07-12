from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class RefusalModel(ABC):
    """Common fit/predict interface for every P(refuse) model, so baselines and
    Bayesian models are interchangeable and comparable on the same held-out rows."""

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "RefusalModel":
        """Fit on a long-format generations DataFrame (one row per generation,
        must include `refused`, `prompt_id`, `domain`, `outcome_family`,
        `steering_level`). Returns self."""
        ...

    @abstractmethod
    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        """Predict P(refuse) for each row of `rows` (same required columns as
        fit(), `refused` not needed). May include prompts/domains unseen in fit()."""
        ...

    def predict_interval(self, rows: pd.DataFrame, level: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
        """Predict a (lo, hi) credible interval per row at the given `level`
        (e.g. 0.9 for a 90% interval). Only models with a posterior implement this."""
        raise NotImplementedError(f"{type(self).__name__} does not provide posterior intervals")
