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
- **Prediction Replay (opt-in; recommended when the model's library versions differ from the RAI environment's)**: Runs the RAI dashboard for models the RAI environment cannot load (newer Python, numpy, pandas, scikit-learn or scikit-survival) by replaying their predictions. See [Prediction replay](#prediction-replay-models-the-rai-environment-cannot-load)
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

The registered model puts a column selector in front of your model, which drops the extra features:
- in survival mode the selector hands the model a DataFrame with its own column names;
- in classification mode it hands the model a numpy array, as it always has. If your model cannot
  predict from that array (for example a scikit-learn `Pipeline` that selects its columns by name
  with a `ColumnTransformer`), the component notices on a few training rows and hands it a
  DataFrame with its own column names instead.

Models that work with the array keep it, because counterfactuals rely on it. dice-ml scores
object-dtype frames, which XGBoost and LightGBM accept as an array but reject as a DataFrame. With
dice-ml's `kdtree` method, plain scikit-learn models do the same. Counterfactuals for a model that
gets a DataFrame are not tested here (see
[What the RAI components hand your model](#what-the-rai-components-hand-your-model)).

The selector takes the column names from the model's `feature_names_in_` as plain strings, as the
survival mode already did. LightGBM (4.5 or later, which added `feature_names_in_`) and XGBoost store
them as `numpy.str_`, which scikit-learn's `ColumnTransformer` rejects ("No valid specification of
the columns"). This was seen with 1.5.1, and the same check is in every release up to 1.9.1. Before
this fix, those models failed in classification mode with extra features.

LightGBM and XGBoost models fitted on pandas `category` columns read the array's raw values as
category codes, because the template never rebuilds the `category` dtype. That is only right when
the categories are exactly 0..k-1. The component checks the model and the estimators it hands its
input to: Pipeline steps, a search's `best_estimator_`, calibration and ensemble members. It flags:
- LightGBM with categories other than 0..k-1;
- XGBoost with any categorical feature, because XGBoost before 3.1 does not store the category
  values, so the component cannot check them.

What happens then depends on whether the model ran before this change:
- **Refused with a clear error** when the model's feature names are a booster's `numpy.str_`. Those
  models failed before this change.
- **Only a warning** otherwise, so a model that ran before still runs. An example is a booster
  behind transformer steps.

Before training, recode such columns so that the categories are exactly 0..k-1 for LightGBM, or as
plain numeric columns for XGBoost.

The `responsibleai-tabular` images ship LightGBM but not XGBoost. An XGBoost model therefore also
needs `use_model_dependency: true` in the constructor, as the comment in
`classification-pipeline.yml` says. Either library must also be in the component's environment.

### 4. Submit the Pipeline

```bash
az ml job create \
  --file classification-pipeline.yml \
  --subscription <your-subscription-id> \
  --resource-group <your-resource-group> \
  --workspace-name <your-workspace-name> \
  --stream
```

## Interactive dashboard in Azure ML Studio

Without compute, Studio shows the dashboard that gather saved. Its interactive features need a
compute instance ("Connect compute"): error analysis with other settings or cohorts, the heatmap,
ICE plots and what-if counterfactuals.

**For the interactive dashboard with the 0.21.0 or 0.22.0 components, set
`rai_tabular_insight_gather` to 0.20.0.** The other steps keep their version. This was confirmed in
Studio on 2026-10-26, with a replay dashboard built with the 0.22.0 components and gather 0.20.0.
The template's pipelines already do this: their gather step is at 0.20.0, the other steps at 0.21.0.

```yaml
  gather_01:
    type: command
    component: azureml://registries/azureml/components/rai_tabular_insight_gather/versions/0.20.0
```

Why:
- **Observed**, in Studio's terminal output for a dashboard built with the 0.22.0 components (all
  steps): Studio started `responsibleai-tabular:28` and typed a startup script whose first line
  imports `azureml.core`, the v1 SDK. The import failed, and the dashboard never loaded.
- **Read**, in the image definitions: only the images up to `:25` contain azureml-core. Gather
  0.20.0 runs in `:24`, 0.21.0 in `:26` and 0.22.0 in `:28`.
- **Confirmed in Studio**: with gather at 0.20.0 and every other step at 0.22.0, which runs in
  `:28`, the dashboard opens interactively. So Studio takes the image from gather, the only step in
  `:24`, whose dashboard output its script downloads.
- Gather 0.20.0 reads the outputs of the 0.21.0 and 0.22.0 steps without problems, because all of
  them use responsibleai 0.36.0. These combinations were run end to end:
  - replay with the 0.22.0 steps, and with the 0.21.0 steps;
  - the default mode with the 0.21.0 steps, in survival and in classification with extra features.

The dashboard's model must also load in that image:
- **A replay dashboard always loads**, because it needs only ml-wrappers. Everything then works live
  except ICE plots, which predict on new rows. What-if needs counterfactuals, which replay cannot
  add.
- **A default-mode survival dashboard does not.** Its pickled `SkSurvRiskWrapper` needs the
  template's `risk_wrapper` module and scikit-survival, which the image does not have.
- **A default-mode classification dashboard loads if the image has the model's libraries.** Then
  everything works live, predictions on changed rows included. This was emulated in `:24` on the
  dashboards of the classification e2e, one with the numpy selector and one with the DataFrame
  fallback. The images ship scikit-learn 1.5.1 and LightGBM, not XGBoost (seen in `:24`, `:26`,
  `:28` and `:39`).

## What the RAI components hand your model

In the default mode the RAI components call your model on data they read and build themselves:
- **0/1 columns arrive as booleans.** The RAI components read the component's MLTable with type
  inference, which turns a column holding only 0 and 1 into `bool` (observed). A model that gets a
  DataFrame receives `bool` columns: classification without extra features, survival mode, or the
  DataFrame fallback of section 3. Preprocessing that rejects them, such as
  `SimpleImputer(strategy="most_frequent")` on a group of 0/1 columns, fails in the constructor
  (inferred).
- **Counterfactuals (dice-ml) hand the model categorical values as strings.**
  - A 0/1 categorical feature arrives as `'True'`/`'False'`, and models that expect numbers fail
    ("could not convert string to float"). This was observed with LightGBM in
    `responsibleai-tabular` 26, 28 and 39.
  - A model that encodes integer categories itself, for example with a `OneHotEncoder` fitted on
    0/1/2, sees `'1'` as an unknown category. It fails, or, with `handle_unknown="ignore"`, scores
    those rows as if the category were absent (inferred, not run).
- **What-if and ICE build their rows from the dashboard's JSON**, so the same type questions apply
  to them. A prediction on a row with a changed numeric value works (emulated); changed categorical
  values were not run.
- **No column arrives as pandas `category`.** The template never rebuilds that dtype: the component
  reads its inputs with `read_csv`, the RAI components with MLTable, and both give int64.
  - Without extra features, in the default mode and with replay alike, a LightGBM model fitted on
    `category` columns fails with "categorical_feature do not match".
  - An XGBoost model reads the raw values as category codes (inferred), which is right only for
    0..k-1. The component prints a warning.
  - With extra features, see section 3.

## Prediction replay (models the RAI environment cannot load)

**Recommended whenever the model's libraries differ from the RAI environment's: the RAI components
then never load them.** So any library version that can also run this component works. This was
verified with Python 3.12, numpy 2, pandas 2, scikit-learn 1.9 and scikit-survival 0.28. The
trade-offs:
- no counterfactuals, what-if or ICE;
- the limits under [Requirements and limits](#requirements-and-limits).

The Microsoft RAI components load the registered model **inside their own environment**,
`responsibleai-tabular`:
- Python 3.9 for the 0.20.0 and 0.21.0 components, 3.10 for 0.22.0;
- numpy < 1.24, pandas < 2, scikit-learn 1.5.1 and responsibleai 0.36.

A model built with a newer stack cannot be loaded there. Examples: Python 3.12, numpy 2,
pandas 2, scikit-learn >= 1.6, scikit-survival >= 0.24. `use_model_dependency: true` does not
help: it installs the model's packages into the RAI environment and cannot change its Python.

`replay_predictions: true` removes that coupling:

1. **The component runs in the model's own environment.**
   - It prepares the data as usual: survival horizon, extra features, missing-value handling.
   - It builds the same wrapper or pipeline the default mode would register.
   - It computes that model's `predict` and `predict_proba` on exactly the rows it writes for the
     RAI components.
2. **It registers a replay model instead of the pickled model.** The model is named
   `rai-<model_name>-replay`. It is an MLflow model that holds the predictions, a copy of the
   datasets and a small loader (`rai_prediction_replay_loader.py`).
3. **The unmodified RAI components load it like any MLflow model.** Inside the RAI environment the
   loader rebuilds an ml-wrappers `PredictionsModelWrapperClassification` from plain files.
   Nothing pickled in the model's environment is ever unpickled there.

The flag is off by default. Without it, the component registers what it did before for every model
that worked. The changes to the default mode only affect models that failed before:
- the DataFrame fallback and the plain-string column names (section 3), including the refusal of
  LightGBM models whose categories are not 0..k-1 and of XGBoost models with categorical features,
  when their names show that they failed before;
- `SkSurvRiskWrapper` declaring itself fitted, which matters only with scikit-learn >= 1.8.

### Enabling it

The component must run in the model's environment. An Azure ML pipeline node cannot override a
component's environment: a node-level `environment:` is accepted but ignored (we tried it). So make a
copy of `components/prepare_model_and_data_for_rai/component.yml` with:
- `environment:` pointing at the model's environment;
- its own `name:` (a lowercase letter, then lowercase letters, digits or underscores, for example
  `prepare_model_and_data_for_rai_modern`);
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
    # the same version as your other rai_tabular_* steps (the template pins 0.21.0, gather 0.20.0)
    component: azureml://registries/azureml/components/rai_tabular_insight_constructor/versions/0.22.0
    inputs:
      # ... the usual inputs ...
      use_model_dependency: false
```

Use the same version for every `rai_tabular_*` step, except gather at 0.20.0 if you want the
interactive dashboard (confirmed in Studio; see
[Interactive dashboard in Azure ML Studio](#interactive-dashboard-in-azure-ml-studio)).
- **End to end.** Replay was verified with every step at 0.22.0
  (`tests/e2e-test-replay-pipeline.yml`), with every step at 0.20.0, and with 0.22.0 plus gather
  0.20.0. The last two are the same pipeline with the versions changed (commands in
  [tests/README.md](tests/README.md)).
- **Inside the images.** The replay model was also checked inside the images of gather 0.20.0
  (`responsibleai-tabular:24`), 0.21.0 (`:26`), 0.22.0 (`:28`) and `:39` (`tests/rai_env/`).

`tests/replay_assets/prepare_model_and_data_for_rai.modern.yml` is such a copy, and
`tests/unit/test_component_variants.py` keeps it in sync with the original.

Pipelines that use a copy by path submit it as an anonymous component. The own `name` matters if the
copy is ever registered: a copy registered as `prepare_model_and_data_for_rai` would replace the
default component for everyone who references it by name.

Instead of a file, you can register a variant once and reference it as `azureml:<name>:<version>`
(not exercised by the tests in this repository):

```bash
az ml component create --file components/prepare_model_and_data_for_rai/component.yml \
  --set name=prepare_model_and_data_for_rai_<suffix> \
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
| Error analysis heatmap and cohort filters | yes for the heatmap; cohort filters expected (they only select stored rows; not run) | heatmap: an emulation of Studio's backend calls in `responsibleai-tabular` 24, 26, 27, 28 and 39 (not in this repository) |
| Interactive dashboard in Studio (connect compute) | yes with gather 0.20.0, except ICE (it predicts on new rows) | Studio (0.22.0 components with gather 0.20.0), and an emulation of the startup in `:24` |
| Explanations (feature importance) | yes, numeric NaN included | replay e2e, RAI-environment check |
| Data balance | not affected (never uses the model) | RAI-environment check (asserts the measures are computed) |
| Causal analysis | expected (never uses the model; refuses NaN) | not run |
| Score card | yes | replay e2e |
| Static dashboard (gather) | yes | replay e2e |
| Counterfactuals (`rai_tabular_counterfactual`) | no: do not add this component | not applicable |
| What-if and ICE plots (live predictions on new rows) | no | not applicable |

Replay can only answer rows it has seen: every row of the train and test sets, including the
copies RAI makes when it saves and reloads them. Anything that scores new or perturbed rows fails
with "The query data was not found in the combined dataset".

Data balance does not reach the dashboard with the registry components, with or without replay.
The gathered dashboards of both e2e tests (gather 0.21.0 in the default mode, 0.22.0 with replay)
hold its configuration but no measures (`measures.json`).

### Requirements and limits

- **Data the lookup supports.** It is checked when the replay model is built, on the data as written
  and again as the RAI components will read it (MLTable), with a clear error:
  - column names must be strings that do not look like numbers;
  - names must not be `target` or `target_<n>`;
  - pandas must be able to quote the names in a query: letters, digits, spaces and ordinary
    punctuation such as `( ) / - + > , .` are fine, while `#`, quotes, backslashes, backticks and
    symbols such as `²` or `≥` are not;
  - no date or time features, including numbers or strings in a date-like column (`*_at`, `*_time`,
    `date`, `datetime`, `modified`, `timestamp*`) that the RAI JSON reload would turn into dates;
  - no `'`, `\` or line breaks inside string values;
  - at least two features.
- **`feature_metadata.dropped_features` is not supported.** With ml-wrappers 0.6.4
  (`responsibleai-tabular:39`) every lookup needs every stored column. Use `extra_features` instead:
  the template's selector already keeps audit-only columns away from the model.
- **Keep `use_model_dependency` false.** With true, the RAI components `pip install -U` the model's
  requirements into their own environment. The replay model declares none besides MLflow's pin,
  but its Python version still comes from the model's environment.
- **Size.** The registry components 0.20.0, 0.21.0 and 0.22.0 (`responsibleai-tabular` 24, 26 and
  28) run ml-wrappers 0.6.0, which finds each row with a pandas query over the stored rows (up to
  three copies of every train and test row).
  - Every job that builds RAIInsights (constructor, each tool, gather) looks up each train row 3
    times and each test row at least twice. The cost should grow roughly with the square of the
    row count (inferred, not measured).
  - Measured with 400 train and 200 test rows on `:26` and `:28`: about 10 s for two constructions,
    so about 5 s per job for the construction, plus the tool's own predictions. Larger sizes are not
    measured; `:24` was not timed.
  - To reduce the cost, subsample before this component. `maximum_rows_for_test_dataset` saves only
    part of the test lookups and limits the analysis to the first test rows.
  - ml-wrappers 0.6.4 (`responsibleai-tabular:39`) hashes rows instead (0.1 s for the same
    constructions), but no registry component uses `:39`.
- **Data governance.** The registered replay model contains every train and test row (features and
  predictions), like the dashboard itself. Apply the same access rules as to the data.
- **Binary classification only.** That is what both of the template's modes (survival and
  classification) produce.
- **Recent scikit-learn.** `SkSurvRiskWrapper` declares itself fitted, so selector → wrapper
  pipelines also predict with scikit-learn >= 1.8, which refuses unfitted pipelines.
- **Classification with extra features in an environment with scikit-learn 1.6 or later.** Your model
  runs inside a scikit-learn `Pipeline`, which checks it before predicting. It must have `fit`
  (from 1.6, inferred) and inherit `BaseEstimator` (from 1.8). Otherwise the component fails with
  scikit-learn's error, printed for both selectors. This does not apply without extra features.

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
  - the component in every mode with the flag off, where it registers what it did before;
  - in classification mode with extra features, the selector:
    - the numpy array for models that accept it, also on hand-built frames shaped like dice-ml's;
    - a DataFrame for a model that selects its columns by name;
    - main's selector when neither works;
    - the refusal of XGBoost fitted on `category` columns, on its own and inside a Pipeline,
      CalibratedClassifierCV or GridSearchCV;
    - only a warning when the model ran before (plain-string names), or has no extra features;
  - the same modes with the flag on, where only the registered model changes and the RAI datasets
    are byte-identical;
  - the replay builder and loader.
- **`tests/rai_env/`:**
  - the replay model loaded inside `responsibleai-tabular` versions 24, 26, 28 and 39, and run
    through the RAIInsights steps of the pipeline;
  - the same for the default mode's classification selector, including real dice-ml counterfactuals
    for LightGBM, and the refusal of a LightGBM model whose categories are not 0..k-1, on its own,
    in a Pipeline and inside CalibratedClassifierCV.
- **`tests/e2e-test-replay-pipeline.yml`:**
  - a scikit-survival model trained on Python 3.12, numpy 2, pandas 2 and scikit-learn 1.9, with
    missing values and an audit-only extra feature;
  - run through the unmodified RAI components: constructor, error analysis, explanation, gather and
    score card;
  - followed by a final assertion step.
- **`tests/e2e-test-classification-pipeline.yml`:** the default mode in classification with an
  extra feature, through the components pinned in `classification-pipeline.yml` (0.21.0, gather
  0.20.0). It uses a model
  that selects its columns by position, and one that selects them by name.

See [tests/README.md](tests/README.md).


