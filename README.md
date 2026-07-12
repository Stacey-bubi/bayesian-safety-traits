# bayesian-safety-traits
Bayesian latent-trait inference for LLM safety evals, with held-out adversarial generalization testing

## Method

We take one open-weight model, apply activation steering along the refusal direction at graded strengths (`steering_level`), and model `P(refuse)` as a single latent trait rather than a raw pass/fail rate. The same `P(refuse)` axis covers two behavior families: refusing a harmful prompt is *good* (`harmful_compliance`), refusing a benign one is *bad* (`over_refusal`) — so one model, `refused ~ steering_level * outcome_family + (1 | prompt) + (1 | domain)`, gives both metrics instead of treating them as separate analyses.

Three models are compared on the same interface (`fit` / `predict_proba` / `predict_interval`):
- **RawRate** — empirical refusal fraction per `(prompt, steering_level)` cell, no generalization.
- **PlainLogistic** — logistic regression on `steering_level + domain + outcome_family`, complete pooling over prompts.
- **HierarchicalLogit** — Bayesian hierarchical logistic regression (PyMC) with random intercepts for prompt and domain; for prompts/domains unseen at fit time, it draws a fresh random effect from the fitted hyper-prior rather than assuming zero.

Models are scored with log loss, Brier score, and ECE under two held-out splits — leave-prompts-out (new prompts, same domains) and leave-one-domain-out (a whole harm domain never seen in training) — plus posterior interval coverage for the Bayesian model, which the other two cannot report at all.

## Getting started

Install dependencies into a local `.venv` (uv reads `uv.lock` + `.python-version`):

```bash
uv sync
```

Generate synthetic data with known ground-truth parameters into `data/` (useful for checking the pipeline recovers planted parameters before trusting it on real data):

```bash
uv run tools/generate_synthetic.py
```

Fit all three models and compare them on leave-prompts-out and leave-one-domain-out splits; writes `data/eval_results.csv`:

```bash
uv run scripts/train_eval.py
```
