# Azure ML Responsible AI Dashboard Template

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19889327.svg)](https://doi.org/10.5281/zenodo.19889327)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This repository provides a comprehensive template for integrating the Responsible AI Dashboard into Azure Machine Learning workflows. The template supports both **classification** and **survival analysis** models, with advanced features for model preparation and explainability analysis.

Additional information can be found in the Azure documentation: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-insights-sdk-cli?view=azureml-api-2&tabs=yaml

## Scope

This is a **template**, meant to be reused: it shows how to wire a model into the Responsible AI
Dashboard, not to reproduce any particular manuscript's results. No patient data is shipped
here, so nothing in this repository regenerates any published figures or numbers. The survival
use case under `tests/` is a **demo on synthetic data**, present so the pipeline can be run
end-to-end; it is not any specific analysis. [docs/med-cli/](docs/med-cli/) documents the model
definition used in one published analysis (Med-CLI) without weights, data, or a trained
artefact — it is documentation only and not runnable without controlled-access data.

## License

This project is licensed under the [MIT License](LICENSE).

## Citation

This repository is archived on Zenodo. Each release has its own DOI; the concept DOI
[10.5281/zenodo.19889327](https://doi.org/10.5281/zenodo.19889327) always resolves to the
latest version.

## Features

- **Dual-Mode Support**: Works with both classification models and survival analysis models (automatically adapted to binary classification)
- **Extra Features Integration**: Allows adding features to the RAI dashboard that weren't used in the original model training
- **Custom Model Preparation Component**: Handles model wrapping, feature expansion, and data filtering
- **Built-in Azure RAI Components**: Uses Microsoft AzureML registry components for error analysis, insight gathering, and more
- **Prediction Replay (opt-in)**: Runs the RAI dashboard for models the RAI environment cannot load (newer Python, numpy, pandas or scikit-learn) by replaying their predictions. See [Prediction replay](#prediction-replay-models-the-rai-environment-cannot-load)
- **Pipeline Automation**: Ready-to-use YAML pipelines for `az ml job create` submission

## Folder Structure

```plaintext
responsible-ai-dashboard-template/
├── components/                              # Custom component definitions
│   └── prepare_model_and_data_for_rai/     # Main preparation component
│       ├── component.yml                    # Component specification
│       ├── main.py                          # Main execution script
│       ├── risk_wrapper.py                  # Survival model wrapper
│       ├── utils.py                         # Utility functions
│       ├── replay.py                        # Prediction replay: builds and registers the replay model
│       └── rai_prediction_replay_loader.py  # Prediction replay: loader run inside the RAI environment
├── classification-pipeline.yml              # Pipeline for classification models
├── survival-pipeline.yml                    # Pipeline for survival analysis models
├── docs/med-cli/                             # Published-run documentation (not runnable)
├── tests/                                    # Synthetic demo pipeline and unit tests
└── README.md                                # This file
```
## How It Works

### The `prepare_model_and_data_for_rai` Component

This custom component is the core of the template. It:

1. **Loads a registered model** from Azure ML Model Registry
2. **Processes training and test data** (Input in folder,file or mltable - csv only at the moment)
3. **Handles two modes**:
   - **Classification Mode**: Works with standard binary classification models
   - **Survival Mode**: Converts survival models to binary classification using time horizons
4. **Adds extra features** (optional): Expands model inputs to include features for RAI analysis that weren't in the original training
5. **Registers the prepared model** for RAI dashboard consumption (or, with `replay_predictions: true`, a model that replays its predictions; see [Prediction replay](#prediction-replay-models-the-rai-environment-cannot-load))
6. **Outputs filtered MLTable datasets** with only relevant features

### Pipeline Flow

1. **prepare_model_and_data_for_rai**: Prepares model and data
2. **create_rai_job_dev**: Creates RAI insight constructor
3. **error_analysis_dev**: Performs error analysis
4. **gather_01**: Gathers all insights into dashboard

## Prerequisites

### Environment Setup

The component requires the `rai-survival-minimal` environment from the `env-registry` repository:
- This is a lightweight environment optimized for RAI dashboards
- **Important**: Add any additional libraries needed by your model (e.g., `xgboost`, `lightgbm`)
- Recommended for both model training and RAI pipeline execution
- With `replay_predictions: true` the component runs in the model's own environment instead (see [Prediction replay](#prediction-replay-models-the-rai-environment-cannot-load))

### Required Assets

- **Registered Model**: Your model must be registered in Azure ML Model Registry
- **Training Data**: MLTable format with all features (including extra features if needed)
- **Test Data**: MLTable format matching training data structure. It can be the training dataset itself if not test avalable.

## Quick Start

### 1. Choose Your Pipeline

- **`classification-pipeline.yml`**: For binary classification models
- **`survival-pipeline.yml`**: For survival analysis models

### 2. Configure Pipeline Parameters

Open the chosen YAML file and update these sections carefully:

#### Essential Parameters (MUST CHANGE)

```yaml
settings:
  default_compute: azureml:your-compute-cluster  # Your compute cluster name

inputs:
  # Model information
  model_name: "your-model-name"                   # Name of registered model
  model_version: 1                                # Version number
  binary_label_name: 'your_label_column'          # Target column name
  
  # Data paths
  training_data:
    type: mltable
    path: "azureml:your-training-data@latest"     # Training data asset
  
  test_data:
    type: mltable
    path: "azureml:your-test-data@latest"         # Test data asset
```

#### Classification-Specific Parameters

```yaml
inputs:
  # No additional parameters needed for pure classification
  extra_features: '[]'  # Optional: add features as JSON array
```

#### Survival-Specific Parameters

```yaml
inputs:
  time_column: 'time_to_event'       # Column with time-to-event values
  event_column: 'event_observed'     # Column indicating if event occurred
  time_horizon: 365                   # Cutoff time for binary classification
                                      # (context-dependent: 30, 90, 365 days, etc.)
  extra_features: '[]'                # Optional: add features as JSON array
```

#### RAI Dashboard Configuration

```yaml
jobs:
  create_rai_job_dev:
    inputs:
      title: "Your Dashboard Title"
      target_column_name: ${{parent.inputs.binary_label_name}}
      categorical_column_names: '["feature1", "feature2"]'  # List categorical features
      classes: '[false, true]'                               # For binary classification
```

### 3. Extra Features (Optional)

You can add features to the RAI dashboard that weren't used in the original model training:

```yaml
inputs:
  extra_features: '["age", "gender", "biomarker"]'  # JSON array format
```

**Use cases:**
- Analyzing fairness across demographic features not used in training
- Investigating feature importance for features excluded from the model
- Clinical context features for interpretability

**Important:** All extra features must exist in your training and test MLTable datasets.

### 4. Submit the Pipeline

```bash
az ml job create \
  --file classification-pipeline.yml \
  --subscription <your-subscription-id> \
  --resource-group <your-resource-group> \
  --workspace-name <your-workspace-name> \
  --stream
```

## Prediction replay (models the RAI environment cannot load)

The Microsoft RAI components load the registered model **inside their own environment**,
`responsibleai-tabular`:
- Python 3.9 for the 0.21.0 components, 3.10 for 0.22.0;
- numpy < 1.24, pandas < 2, scikit-learn 1.5.1 and responsibleai 0.36.

A model built with a newer stack cannot be loaded there. Examples: Python 3.12, numpy 2,
pandas 2, scikit-learn >= 1.6, scikit-survival >= 0.24. `use_model_dependency: true` does not
help: it installs the model's packages into the RAI environment and cannot change its Python.

`replay_predictions: true` removes that coupling:

1. **The component runs in the model's own environment.**
   - It prepares the data as usual: survival horizon, extra features, missing-value handling.
   - It builds the same wrapper or pipeline the default mode would register. One difference: in
     classification mode with extra features, the selector hands the model a DataFrame, so models
     that select their columns by name work.
   - It computes that model's `predict` and `predict_proba` on exactly the rows it writes for the
     RAI components.
2. **It registers a replay model instead of the pickled model.** The model is named
   `rai-<model_name>-replay`. It is an MLflow model that holds the predictions, a copy of the
   datasets and a small loader (`rai_prediction_replay_loader.py`).
3. **The unmodified RAI components load it like any MLflow model.** Inside the RAI environment the
   loader rebuilds an ml-wrappers `PredictionsModelWrapperClassification` from plain files.
   Nothing pickled in the model's environment is ever unpickled there.

The flag is off by default. Without it, the component behaves exactly as before.

### Enabling it

The component must run in the model's environment. An Azure ML pipeline node cannot override a
component's environment: a node-level `environment:` is accepted but ignored (we tried it). So make a
copy of `components/prepare_model_and_data_for_rai/component.yml` with:
- `environment:` pointing at the model's environment;
- its own `name:`;
- if the copy lives in another folder, `code:` pointing back at `components/prepare_model_and_data_for_rai`.

Use that copy in the pipeline, and keep `use_model_dependency` false:

```yaml
jobs:
  prepare_model_and_data_for_rai:
    type: command
    component: components/prepare_model_and_data_for_rai/component.<your-model-env>.yml  # the copy
    inputs:
      # ... the usual inputs ...
      replay_predictions: true

  create_rai_job_dev:
    type: command
    component: azureml://registries/azureml/components/rai_tabular_insight_constructor/versions/0.22.0
    inputs:
      # ... the usual inputs ...
      use_model_dependency: false
```

`tests/replay_assets/prepare_model_and_data_for_rai.modern.yml` is such a copy, and
`tests/unit/test_component_variants.py` keeps it in sync with the original.

Pipelines that use a copy by path submit it as an anonymous component. The own `name` matters if the
copy is ever registered: a copy registered as `prepare_model_and_data_for_rai` would replace the
default component for everyone who references it by name.

Instead of a file, you can register a variant once and reference it as `azureml:<name>:<version>`
(not exercised by the tests in this repository):

```bash
az ml component create --file components/prepare_model_and_data_for_rai/component.yml \
  --set name=prepare_model_and_data_for_rai_<env> \
  --set environment=azureml:<your-model-environment>:<version>
```

The model's environment needs the model's own libraries, plus what the component imports:
- `mlflow` and `azureml-mlflow`, to load and register models;
- `mltable`, `azure-identity`, `scikit-learn`, `scipy`, `pandas` and `pyyaml`.

`tests/environment_modern/` is a working example.

### What works

| RAI feature | With replay | Verified by |
|---|---|---|
| Constructor | yes | replay e2e, RAI-environment check |
| Error analysis (tree) | yes | replay e2e, RAI-environment check |
| Error analysis heatmap and cohort filters | yes (same dataset rows; inferred) | not run |
| Explanations (feature importance) | yes, numeric NaN included | replay e2e, RAI-environment check |
| Data balance | yes (never uses the model) | RAI-environment check (computed with categorical features) |
| Causal analysis | expected (never uses the model; refuses NaN) | not run |
| Score card | yes | replay e2e |
| Static dashboard (gather) | yes | replay e2e |
| Counterfactuals (`rai_tabular_counterfactual`) | no: do not add this component | not applicable |
| What-if and ICE plots (live predictions on new rows) | no | not applicable |

Replay can only answer rows it has seen: every row of the train and test sets, including the
copies RAI makes when it saves and reloads them. Anything that scores new or perturbed rows fails
with "The query data was not found in the combined dataset".

### Requirements and limits

- **Data the lookup supports.** It is checked when the replay model is built, on the data as written
  and again as the RAI components will read it (MLTable), with a clear error:
  - column names must be strings that do not look like numbers;
  - names must not be `target` or `target_<n>`;
  - pandas must be able to quote the names in a query: letters, digits, spaces and ordinary
    punctuation such as `( ) / - + > , .` are fine, while `#`, quotes, backslashes, backticks and
    symbols such as `²` or `≥` are not;
  - no date or time features, including numbers or strings in a date-like column (`*_at`, `*_time`,
    `date`, `timestamp*`) that the RAI JSON reload would turn into dates;
  - no `'`, `\` or line breaks inside string values;
  - at least two features.
- **`feature_metadata.dropped_features` is not supported.** With ml-wrappers 0.6.4
  (`responsibleai-tabular:39`) every lookup needs every stored column. Use `extra_features` instead:
  the template's selector already keeps audit-only columns away from the model.
- **Keep `use_model_dependency` false.** With true, the RAI components `pip install -U` the model's
  requirements into their own environment. The replay model declares none besides MLflow's pin,
  but its Python version still comes from the model's environment.
- **Size.** The 0.22.0 components run ml-wrappers 0.6.0 (in `responsibleai-tabular:28`), which
  looks up each row with a pandas query that scans the stored rows. Measured with 400 train and
  200 test rows: about 10 s for the constructor checks, which every RAI job repeats. Fine for
  thousands of rows, slow for hundreds of thousands. Lower `maximum_rows_for_test_dataset` if
  needed. `responsibleai-tabular:39` (ml-wrappers 0.6.4) hashes rows instead and is much faster.
- **Data governance.** The registered replay model contains every train and test row (features and
  predictions), like the dashboard itself. Apply the same access rules as to the data.
- **Binary classification only.** That is what both of the template's modes (survival and
  classification) produce.
- **Recent scikit-learn.** `SkSurvRiskWrapper` declares itself fitted, so selector → wrapper
  pipelines also predict with scikit-learn >= 1.8, which refuses unfitted pipelines.

### Missing values

Rows with missing values are replayed like any other row, so a model that handles NaN works. The
RAI components keep their own rules, independent of replay:

- Explanations refuse missing values in **categorical** features. Numeric NaN is fine: the replay
  e2e has 62 of 128 test rows with NaN.
- Causal analysis and counterfactuals refuse any missing value (read in responsibleai 0.36.0).
- Error analysis:
  - the tree does not place rows with NaN in a numeric split feature in either child (inferred from
    erroranalysis 0.5.5, not run);
  - importances are skipped when the test set has NaN;
  - the static dashboard shows NaN as 0.

### How it is tested

- **`tests/unit/`:**
  - the component in every mode with the flag off, where behaviour is unchanged;
  - the same modes with the flag on, where only the registered model changes and the RAI datasets
    are byte-identical;
  - the replay builder and loader.
- **`tests/rai_env/`:** the replay model loaded inside `responsibleai-tabular` versions 28 and 39,
  and run through the RAIInsights steps of the pipeline.
- **`tests/e2e-test-replay-pipeline.yml`:**
  - a scikit-survival model trained on Python 3.12, numpy 2, pandas 2 and scikit-learn 1.9, with
    missing values and an audit-only extra feature;
  - run through the unmodified RAI components: constructor, error analysis, explanation, gather and
    score card;
  - followed by a final assertion step.

See [tests/README.md](tests/README.md).


