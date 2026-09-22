# Med-CLI model definition

This folder documents the model and pipeline configuration used for the Med-CLI 90-day
mortality analysis reported in the manuscript. It is documentation only: no weights, no
patient data, no trained artefact. It is not runnable without controlled-access cohort data
and does not regenerate the manuscript's figures or numbers.

- [model-definition.md](model-definition.md) — the estimator, its inputs, what `predict()`
  returns, and how the operating cutoff was derived.
- [pipeline.yml](pipeline.yml) — the `prepare_model_and_data_for_rai` configuration as used for
  the published run (horizon, extra features, missing-data handling, RAI component versions).
  It references workspace assets that do not exist outside the original subscription and is not
  submittable as-is.
- [test_assets/test_assets.json](test_assets/test_assets.json) — the locked cutoff, with its
  provenance (80th percentile of the partial hazard on the derivation cohort, fixed before
  evaluation on the external cohort — despite the folder's name, the value was never computed
  on evaluation data).

The deployed wrapper class is `SkSurvRiskWrapper` (`components/prepare_model_and_data_for_rai/risk_wrapper.py`),
which accepts any estimator exposing `predict()`; Med-CLI's own estimator is a lifelines
`CoxPHFitter`, not scikit-survival. `log_transform` is a constructor parameter of that wrapper,
not a pipeline or component input. In this template repository, `main.py` constructs the wrapper
in identity mode (`log_transform=False`). By contrast, the published Med-CLI run constructed the
wrapper with `log_transform=True` so the sigmoid saw the log-partial-hazard; that is the setting
under which a multiplicative positive risk score keeps the cutoff's meaning. Accordingly,
[`pipeline.yml`](pipeline.yml) records the published run configuration for documentation, but it is
not runnable as-is and should not be read as the current template integration.
