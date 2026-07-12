from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

from .base import RefusalModel


class HierarchicalLogit(RefusalModel):
    """Hierarchical logistic regression: random intercepts for prompt and domain,
    plus a steering_level x outcome_family interaction. Fit on Binomial-aggregated
    cells (equivalent to per-generation Bernoulli, faster to sample). For rows
    whose prompt/domain never appeared in fit(), draws a fresh random effect from
    the fitted hyper-prior instead of plugging in 0."""

    def __init__(self, draws: int = 1000, tune: int = 1000, chains: int = 4, seed: int = 0) -> None:
        """Args:
            draws, tune: NUTS post-warmup draws and warmup iterations per chain.
            chains: number of MCMC chains.
            seed: RNG seed for sampling and for fresh random-effect draws at predict time.
        """
        self._sample_kwargs = {"draws": draws, "tune": tune, "chains": chains, "random_seed": seed, "target_accept": 0.95}
        self.idata: az.InferenceData | None = None
        self._prompts: list[str] = []
        self._domains: list[str] = []
        self._rng = np.random.default_rng(seed)

    def fit(self, train_df: pd.DataFrame) -> "HierarchicalLogit":
        """Args: train_df must have `refused`, `prompt_id`, `domain`, `outcome_family`, `steering_level`."""
        cells = (
            train_df.assign(is_benign=(train_df["outcome_family"] == "over_refusal").astype(float))
            .groupby(["prompt_id", "domain", "steering_level", "is_benign"], as_index=False)
            .agg(n_trials=("refused", "size"), n_refuse=("refused", "sum"))
        )
        self._prompts = sorted(cells["prompt_id"].unique())
        self._domains = sorted(cells["domain"].unique())
        prompt_pos = {p: i for i, p in enumerate(self._prompts)}
        domain_pos = {d: i for i, d in enumerate(self._domains)}
        prompt_idx = cells["prompt_id"].map(prompt_pos).to_numpy()
        domain_idx = cells["domain"].map(domain_pos).to_numpy()

        with pm.Model():
            beta0 = pm.Normal("beta0", 0, 1.5)
            beta_steer = pm.Normal("beta_steer", 0, 1)
            beta_family = pm.Normal("beta_family", 0, 1)
            beta_interact = pm.Normal("beta_interact", 0, 1)
            sigma_prompt = pm.HalfNormal("sigma_prompt", 1)
            sigma_domain = pm.HalfNormal("sigma_domain", 1)
            # non-centered: raw ~ Normal(0,1) scaled by sigma, avoids the funnel
            # geometry that causes divergences when sigma is small (Neal's funnel)
            u_prompt_raw = pm.Normal("u_prompt_raw", 0, 1, shape=len(self._prompts))
            u_domain_raw = pm.Normal("u_domain_raw", 0, 1, shape=len(self._domains))
            u_prompt = pm.Deterministic("u_prompt", u_prompt_raw * sigma_prompt)
            u_domain = pm.Deterministic("u_domain", u_domain_raw * sigma_domain)

            alpha = cells["steering_level"].to_numpy()
            is_benign = cells["is_benign"].to_numpy()
            eta = (
                beta0
                + beta_steer * alpha
                + beta_family * is_benign
                + beta_interact * alpha * is_benign
                + u_prompt[prompt_idx]
                + u_domain[domain_idx]
            )
            pm.Binomial(
                "obs",
                n=cells["n_trials"].to_numpy(),
                p=pm.math.sigmoid(eta),
                observed=cells["n_refuse"].to_numpy(),
            )
            self.idata = pm.sample(**self._sample_kwargs, progressbar=False)
        return self

    def _posterior_p(self, rows: pd.DataFrame) -> np.ndarray:
        """P(refuse) per posterior draw per row, shape (n_draws, n_rows).

        Args: rows must have `prompt_id`, `domain`, `outcome_family`, `steering_level`.
        """
        assert self.idata is not None, "call fit() first"
        post = self.idata.posterior
        beta0 = post["beta0"].to_numpy().reshape(-1)
        beta_steer = post["beta_steer"].to_numpy().reshape(-1)
        beta_family = post["beta_family"].to_numpy().reshape(-1)
        beta_interact = post["beta_interact"].to_numpy().reshape(-1)
        sigma_prompt = post["sigma_prompt"].to_numpy().reshape(-1)
        sigma_domain = post["sigma_domain"].to_numpy().reshape(-1)
        u_prompt = post["u_prompt"].to_numpy().reshape(-1, len(self._prompts))
        u_domain = post["u_domain"].to_numpy().reshape(-1, len(self._domains))
        n_draws = beta0.shape[0]

        prompt_pos = {p: i for i, p in enumerate(self._prompts)}
        domain_pos = {d: i for i, d in enumerate(self._domains)}

        def group_effects(
            ids: np.ndarray, known_pos: dict[str, int], known_draws: np.ndarray, sigma: np.ndarray
        ) -> np.ndarray:
            fresh: dict[str, np.ndarray] = {}
            out = np.empty((n_draws, len(ids)))
            for j, gid in enumerate(ids):
                pos = known_pos.get(gid)
                if pos is not None:
                    out[:, j] = known_draws[:, pos]
                else:
                    if gid not in fresh:
                        fresh[gid] = self._rng.normal(0, sigma)
                    out[:, j] = fresh[gid]
            return out

        u_p = group_effects(rows["prompt_id"].to_numpy(), prompt_pos, u_prompt, sigma_prompt)
        u_d = group_effects(rows["domain"].to_numpy(), domain_pos, u_domain, sigma_domain)

        is_benign = (rows["outcome_family"] == "over_refusal").to_numpy(dtype=float)
        alpha = rows["steering_level"].to_numpy(dtype=float)

        eta = (
            beta0[:, None]
            + beta_steer[:, None] * alpha[None, :]
            + beta_family[:, None] * is_benign[None, :]
            + beta_interact[:, None] * (alpha * is_benign)[None, :]
            + u_p
            + u_d
        )
        return 1.0 / (1.0 + np.exp(-eta))

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        """Posterior predictive mean P(refuse) per row. Args: see _posterior_p."""
        return self._posterior_p(rows).mean(axis=0)

    def predict_interval(self, rows: pd.DataFrame, level: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
        """Posterior predictive (lo, hi) per row at the given credible `level`.

        Args: rows same as predict_proba(); level e.g. 0.9 for a 90% interval.
        """
        p = self._posterior_p(rows)
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        return np.quantile(p, lo, axis=0), np.quantile(p, hi, axis=0)
