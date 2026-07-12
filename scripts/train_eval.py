"""Fit RawRate, PlainLogistic, and HierarchicalLogit and compare them under
leave-prompts-out and leave-one-domain-out splits. Run as a script; reads
DATA_PATH, writes RESULTS_PATH."""

from __future__ import annotations

import pandas as pd

from bayesian_safety_traits.data import leave_one_domain_out, leave_prompts_out, load_data
from bayesian_safety_traits.metrics import coverage, score
from bayesian_safety_traits.models import HierarchicalLogit, PlainLogistic, RawRate, RefusalModel

DATA_PATH = "data/synthetic_eval_data.csv"
RESULTS_PATH = "data/eval_results.csv"
CELL_COLS = ["prompt_id", "domain", "outcome_family", "steering_level"]

MODELS: dict[str, type[RefusalModel]] = {
    "RawRate": RawRate,
    "PlainLogistic": PlainLogistic,
    "HierarchicalLogit": HierarchicalLogit,
}


def evaluate(model: RefusalModel, train: pd.DataFrame, test: pd.DataFrame, split: str, name: str) -> dict[str, float | str]:
    """Fit `model` on `train`, score it on `test`, return one results row.

    Args:
        model: fresh (unfit) RefusalModel instance.
        train, test: DataFrames from a data.py splitter.
        split: label for the split, stored in the results row (e.g. "leave_prompts_out").
        name: label for the model, stored in the results row (e.g. "RawRate").
    """
    model.fit(train)
    p_pred = model.predict_proba(test)
    metrics = score(test["refused"].to_numpy(), p_pred)

    cells = (
        test.assign(p_pred=p_pred)
        .groupby(CELL_COLS, as_index=False)
        .agg(rate=("refused", "mean"), p_pred=("p_pred", "first"))
    )
    try:
        lo, hi = model.predict_interval(cells[CELL_COLS])
        metrics["coverage"] = coverage(cells["rate"].to_numpy(), lo, hi)
    except NotImplementedError:
        metrics["coverage"] = float("nan")

    return {"model": name, "split": split, **metrics}


def main() -> None:
    """Load DATA_PATH, run every model over LPO and per-domain LODO splits,
    print the results table, and write it to RESULTS_PATH."""
    df = load_data(DATA_PATH)
    results: list[dict[str, float | str]] = []

    train, test = leave_prompts_out(df, frac=0.2, seed=0)
    for name, cls in MODELS.items():
        results.append(evaluate(cls(), train, test, "leave_prompts_out", name))

    harm_domains = df.loc[df["outcome_family"] == "harmful_compliance", "domain"].unique()
    for domain in sorted(harm_domains):
        train, test = leave_one_domain_out(df, domain)
        for name, cls in MODELS.items():
            results.append(evaluate(cls(), train, test, f"leave_one_domain_out:{domain}", name))

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    results_df.to_csv(RESULTS_PATH, index=False)


if __name__ == "__main__":
    main()
