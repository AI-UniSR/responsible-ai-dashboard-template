# E2E Test Suite for RAI Survival Pipeline

## Overview

This test suite validates the end-to-end compatibility between survival models (scikit-survival) and Azure ML Responsible AI (RAI) components through synthetic data generation and dependency injection.

## Architecture

### Pipeline Flow
```
1. prepare_e2e_survival_assets
   ↓ (generates synthetic data + model + threshold)
2. prepare_model_and_data_for_rai
   ↓ (wraps survival model as classifier, applies threshold override)
3. create_rai_job → 4. error_analysis → 5. gather_insights
```

### Key Components

**[prepare_e2e_survival_assets/](prepare_e2e_survival_assets/)**
- Generates 500 synthetic patients with 6 features (age, bmi, sex, tumor, biomarker_1, biomarker_2)
- Trains `GradientBoostingSurvivalAnalysis` model
- Calculates **threshold as 66th percentile** of risk predictions on the training set (locked before evaluation)
- Outputs: train/test MLTables, registered model, `test_assets.json` with threshold

**[environment/](environment/)**
- `conda_dependencies.yml`: Package dependencies (scikit-survival, mlflow, mltable)
- `environment.yml`: Azure ML environment registration config
- Register with: `az ml environment create --file environment.yml`

**[e2e-test-surv-pipeline.yml](e2e-test-surv-pipeline.yml)**
- Full pipeline orchestration
- Passes `test_assets` output from step 1 to step 2 for threshold injection

## Threshold Override Mechanism

The `test_assets` folder contains `test_assets.json`:
```json
{
  "threshold": 1.2345,
  "percentile": 66,
  "description": "Threshold calculated as 66th percentile of risk predictions on training set"
}
```

When `prepare_model_and_data_for_rai` receives `test_assets`, it:
1. Loads `threshold` from JSON
2. Overrides default (1.0) in `SkSurvRiskWrapper(model, threshold=loaded_value)`
3. Sets the cutoff of the rank-preserving sigmoid mapping: `p = expit(s - threshold)`, a score
   transformation, not a calibrated probability

## Critical Dependencies

### Model Contract
Survival models **must**:
- Have `predict(X)` returning **risk scores** (not survival probabilities), valid over the whole
  real line by default; strictly positive scores are only required when the wrapper is
  constructed with `log_transform=True`
- Expose `feature_names_in_` attribute (sklearn convention)
- Be compatible with `scikit-survival` (e.g., `CoxPHSurvivalAnalysis`, `GradientBoostingSurvivalAnalysis`)
  or any other estimator exposing `predict()` — `SkSurvRiskWrapper` is not scikit-survival-specific

### Data Format
- **MLTable**: CSV + `MLTable` YAML file with `read_delimited` transformation
- **Categorical features**: Declared in pipeline as JSON string: `'["sex", "tumor"]'`
- **Binary labels**: Classes must be `'[false, true]'` for RAI components

### Environment
- `scikit-survival` is **critical** (not in default Azure ML images)
- Version pinning: `scikit-learn==1.5.1`, `mlflow==3.1.1`, `pandas==1.5.3`
- Use `python=3.9` (compatibility with scikit-survival)

## Extending the Test Suite

### Adding Features
1. Modify `generate_synthetic_survival_data()` in `main.py`
2. Update `feature_cols` list
3. Adjust `categorical_column_names` in pipeline YAML if adding categoricals

### Changing Model Type
Replace `GradientBoostingSurvivalAnalysis` with any scikit-survival estimator:
```python
from sksurv.linear_model import CoxPHSurvivalAnalysis
model = CoxPHSurvivalAnalysis()
```
**Caveat**: Ensure `predict()` returns risk scores (most do, but verify)

### Adjusting Sample Size
- Default: 500 patients (350 train, 150 test)
- Increase via pipeline input: `n_samples: 1000`
- **Warning**: Small samples (<100) may cause RAI components to fail

### Modifying Threshold Percentile
Currently hardcoded at 66th percentile, computed on the training split. To change:
1. Edit `calculate_threshold_percentile(percentile=66)` in `main.py`
2. Update `save_test_assets()` call with new percentile value

## Important Caveats

### 1. Model Registration Race Condition
`model_version: 1` is hardcoded in pipeline. If running multiple times:
- **Solution A**: Delete model before rerun: `az ml model delete --name <model_name>`
- **Solution B**: Parameterize version or use `@latest` (requires component modification)

### 2. Feature Name Consistency
The wrapper checks `model.feature_names_in_` against data columns. Mismatch causes:
```
ValueError: feature_names_in_ doesn't match DataFrame columns
```
**Fix**: Ensure synthetic data columns match exactly (order matters)

### 3. Survival Data Filtering
`prepare_binary_classification_dataset()` drops patients where:
- `event=False` AND `tte <= time_horizon` (censored before cutoff)

This reduces sample size. Ensure sufficient remaining samples (~70% retention expected).

### 4. RAI Component Requirements
- `use_model_dependency: true` is **mandatory** for scikit-survival models
- `task_type: classification` (even though underlying model is survival)
- Timeout: Error analysis may take >1 hour on large datasets

## Running the Test

```bash
# 1. Register environment (one-time)
cd tests/environment
az ml environment create --file environment.yml

# 2. Submit pipeline
cd ..
az ml job create --file e2e-test-surv-pipeline.yml
```

## Prediction Replay Tests

These tests cover `replay_predictions` (see the main README, section "Prediction replay"). Every
command takes the usual `--subscription/--resource-group/--workspace-name` arguments. The YAML files
name the `clusterprdwe-ogreif` compute; to use another one, add
`--set compute=azureml:<cluster>` (command jobs) or `--set settings.default_compute=azureml:<cluster>`
(pipelines).

### Environments (one-time)

```bash
az ml environment create --file tests/environment/environment.yml          # rai-survival-test:13
az ml environment create --file tests/environment_modern/environment.yml   # rai-replay-modern-test:1
```

`rai-replay-modern-test` is a model stack the RAI environment cannot load: Python 3.12, numpy 2,
pandas 2, scikit-learn 1.9, scikit-survival 0.28, MLflow 2.22.

### Unit tests (`unit/`)

`test_risk_wrapper.py`, `test_replay.py`, `test_main_modes.py` and `test_component_variants.py` need
no Azure resources (the model registry is faked). `test_main_modes.py` runs `main()` in every mode:
- with the flag off, it checks that registration and outputs are unchanged;
- with the flag on, it checks that only the registered model changes.

Run them in both environments:

```bash
az ml job create --file tests/run-unit-tests-job.yml
az ml job create --file tests/run-unit-tests-job.yml --set environment=azureml:rai-replay-modern-test:1
```

### Replay inside the RAI environment (`rai_env/`)

The RAI environment has no pytest, so `check_replay_in_rai_env.py` is a script. It builds replay
models with the component's code, loads them as the RAI components do, and runs the RAIInsights
steps of the pipeline: the constructor checks, error analysis, explanations, the gather and
score-card save/load cycles, and `get_data()`. The data includes missing values and an audit-only
extra feature. Run it for each RAI environment version your pipelines use:

```bash
az ml job create --file tests/rai_env/check-replay-in-rai-env-job.yml
az ml job create --file tests/rai_env/check-replay-in-rai-env-job.yml \
  --set environment=azureml://registries/azureml/environments/responsibleai-tabular/versions/39
```

### Replay end to end (`e2e-test-replay-pipeline.yml`)

1. `prepare_e2e_survival_assets` runs in `rai-replay-modern-test:1`, with `model_type: gbsa_impute`
   (median imputer + GBSA, so missing values are kept) and an audit-only `site` column.
2. `prepare_model_and_data_for_rai` runs in the same environment, with `replay_predictions: true`
   and `extra_features: '["site"]'`.

Both steps use the copies in `replay_assets/*.modern.yml`. A pipeline node cannot override a
component's environment (it is accepted but ignored), and `unit/test_component_variants.py` keeps
the copies in sync with the originals.
3. The unmodified registry components 0.22.0 run, with `use_model_dependency: false`: constructor,
   error analysis, explanation, gather, and score card (config in `replay_assets/`).
4. `check_replay_dashboard` runs in the RAI environment. It asserts that:
   - the dashboard holds the replay model;
   - predictions on the reloaded data equal those cached at construction;
   - the insights and the score card PDF exist;
   - the predictions were computed with Python 3.12.

```bash
az ml job create --file tests/e2e-test-replay-pipeline.yml
```

`e2e-test-surv-pipeline.yml` (above) is the regression test for the default mode. It does not set
the flag, so it must keep passing unchanged.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Import error: `sksurv` | Wrong environment | Verify `rai-survival-test:1` is used |
| `feature_names_in_` missing | Model not fitted | Ensure `model.fit()` is called |
| RAI timeout | Large dataset | Increase `limits.timeout` or reduce samples |
| Threshold not applied | `test_assets` not passed | Check pipeline job connections |
| Model version conflict | Model already exists | Delete existing model or increment version |

