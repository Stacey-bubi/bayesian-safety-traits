"""
Synthetic evaluation dataset generator for the Bayesian latent safety-trait project.

One-off dev helper that produces the table (one row = one generation) the pipeline expects.
Data is generated from a KNOWN hierarchical logistic process with
hand-set parameters (saved to ground_truth_params.json) so the hierarchical model's
recovery of those parameters can be checked.

Sign convention for steering_level (alpha):
    higher alpha  ->  MORE refusal   (adding the refusal direction)
    lower  alpha  ->  LESS refusal   (suppressing / ablating it)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DATA_DIR = Path("data")

# Ground-truth parameters the hierarchical model should recover.
GT: dict[str, Any] = {
    "beta0_intercept": 1.6,  # baseline logit(refuse) for harmful family at alpha=0 -> P~0.83
    "beta_steer": 1.2,  # slope of alpha for HARMFUL family
    "beta_family_benign": -3.0,  # shift for over_refusal family
    "beta_interaction": -0.6,  # benign slope = 1.2 - 0.6 = 0.6
    "sigma_prompt": 0.8,
    "sigma_domain": 0.5,
    "rule_sensitivity": 0.82,
    "rule_specificity": 0.80,
    "judge_sensitivity": 0.94,
    "judge_specificity": 0.93,
    "n_replicates": 8,  # per (prompt, alpha) cell;
    "alphas": [-2.0, -1.0, 0.0, 1.0, 2.0],
    "steering_layer": 13,
    "seed": 20260712,
}

HARMFUL_DOMAINS = ["Malware/Hacking", "Disinformation", "Fraud/Deception", "Physical harm"]
BENIGN_DOMAINS = ["benign_privacy", "benign_definitions", "benign_figurative"]


def build_prompts(rng: np.random.Generator, gt: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pid = 0
    for dom in HARMFUL_DOMAINS:
        for _ in range(10):
            pid += 1
            rows.append(
                {
                    "prompt_id": f"p{pid:03d}",
                    "domain": dom,
                    "outcome_family": "harmful_compliance",
                    "source": "JBB-Behaviors",
                    "text": f"[synthetic harmful prompt {pid} / {dom}]",
                }
            )
    for dom in BENIGN_DOMAINS:
        for _ in range(10):
            pid += 1
            rows.append(
                {
                    "prompt_id": f"p{pid:03d}",
                    "domain": dom,
                    "outcome_family": "over_refusal",
                    "source": "XSTest",
                    "text": f"[synthetic benign prompt {pid} / {dom}]",
                }
            )
    prompts = pd.DataFrame(rows)

    domains = prompts["domain"].unique()
    u_domain = {d: rng.normal(0, gt["sigma_domain"]) for d in domains}
    prompts["u_prompt"] = rng.normal(0, gt["sigma_prompt"], size=len(prompts))
    prompts["u_domain"] = prompts["domain"].map(u_domain)
    return prompts


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def logit_refuse(row: pd.Series, alpha: float, gt: dict[str, Any]) -> float:
    is_benign = 1.0 if row["outcome_family"] == "over_refusal" else 0.0
    return (
        gt["beta0_intercept"]
        + gt["beta_steer"] * alpha
        + gt["beta_family_benign"] * is_benign
        + gt["beta_interaction"] * alpha * is_benign
        + row["u_prompt"]
        + row["u_domain"]
    )


def observe(rng: np.random.Generator, true_label: int, sens: float, spec: float) -> int:
    """Corrupt a true 0/1 label through a noisy rater with given sens/spec."""
    if true_label == 1:
        return int(rng.random() < sens)
    return int(rng.random() < (1 - spec))


def generate_observations(rng: np.random.Generator, prompts: pd.DataFrame, gt: dict[str, Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    gen_counter = 0
    for _, row in prompts.iterrows():
        for alpha in gt["alphas"]:
            p = sigmoid(logit_refuse(row, alpha, gt))
            for r in range(gt["n_replicates"]):
                gen_counter += 1
                refused_true = int(rng.random() < p)
                base = 80 if refused_true else 20
                judge_score = float(np.clip(rng.normal(base, 12), 0, 100))

                records.append(
                    {
                        "generation_id": f"g{gen_counter:06d}",
                        "prompt_id": row["prompt_id"],
                        "domain": row["domain"],
                        "outcome_family": row["outcome_family"],
                        "source": row["source"],
                        "steering_level": alpha,
                        "steering_layer": gt["steering_layer"],
                        "replicate_id": r,
                        "response_text": f"[synthetic response; true_refuse={refused_true}]",
                        "refused_rule": observe(rng, refused_true, gt["rule_sensitivity"], gt["rule_specificity"]),
                        "refused_judge": observe(rng, refused_true, gt["judge_sensitivity"], gt["judge_specificity"]),
                        "judge_score": round(judge_score, 1),
                        "judge_id": "llm_judge_v1",
                        # not part of the real input contract; kept for parameter recovery checks only
                        "refused_true_HIDDEN": refused_true,
                        "p_true_HIDDEN": round(p, 4),
                    }
                )
    return pd.DataFrame(records)


def validate_contract(df: pd.DataFrame, gt: dict[str, Any]) -> None:
    assert df["steering_level"].nunique() > 1, "steering_level does not vary"

    cell_counts = df.groupby(["prompt_id", "steering_level"]).size()
    assert (cell_counts >= 5).all(), "some (prompt_id, steering_level) cell has <5 replicates"

    assert df["outcome_family"].notna().all(), "outcome_family missing for some rows"

    alpha_sets = df.groupby("prompt_id")["steering_level"].apply(lambda s: frozenset(s.unique()))
    assert alpha_sets.nunique() == 1, "alpha levels are not identical across all prompts"


def main() -> None:
    rng = np.random.default_rng(GT["seed"])
    prompts = build_prompts(rng, GT)
    df = generate_observations(rng, prompts, GT)
    validate_contract(df, GT)

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_DIR / "synthetic_eval_data.csv", index=False)
    with open(DATA_DIR / "ground_truth_params.json", "w") as f:
        json.dump(GT, f, indent=2)
    prompts[["prompt_id", "domain", "outcome_family", "u_prompt", "u_domain"]].to_csv(
        DATA_DIR / "ground_truth_random_effects.csv", index=False
    )

    n_cells = prompts.shape[0] * len(GT["alphas"])
    print(f"Saved synthetic_eval_data.csv, ground_truth_params.json, ground_truth_random_effects.csv to {DATA_DIR}")
    print("rows:", len(df), "| prompts:", prompts.shape[0], "| cells:", n_cells)


if __name__ == "__main__":
    main()
