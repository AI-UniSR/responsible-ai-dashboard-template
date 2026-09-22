# Model definition — Med-CLI 90-day mortality

## Estimator

- `lifelines.CoxPHFitter` (Cox proportional hazards), fit on the derivation cohort (Cohort 1).
- Inputs (4): `Age`, `Sex`, `adm_dysphagia`, `Braden Score`.
- The fitted `CoxPHFitter` was passed through a thin Med-CLI-specific adapter exposing
  `predict(X) = predict_partial_hazard(X)` and `feature_names_in_` from the fitted covariates,
  and the wrapper received that adapter.
- Through that adapter, `predict()` returns the partial hazard `r` (a risk score, not a
  probability or a survival probability), on the model's native positive-real scale.

## Cutoff derivation

- `τ_r = 1.871752447313276`, the 80th percentile of the partial hazard `r` on the derivation
  cohort (Cohort 1), fixed before evaluation on the external cohort (Cohort 2). No operating
  point was selected on evaluation data.
- The deployed `SkSurvRiskWrapper` is constructed with `log_transform=True`, so the sigmoid
  operates on `s = ln(r)` against `τ = ln(τ_r) ≈ 0.627`. Because `r = exp(s)`, the sigmoid
  mapping `p = expit(s - τ)` is algebraically identical to the old `p = r / (r + τ_r)` for every
  patient (see `tests/unit/test_risk_wrapper.py::test_old_vs_new_equivalence`).

## Naming

The deployed class is `SkSurvRiskWrapper` — the name is a historical artifact; the wrapper
accepts any estimator exposing `predict()`, and here it wraps a lifelines `CoxPHFitter`, not a
scikit-survival estimator.
