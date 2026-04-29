# Azure ML Responsible AI Dashboard Template

This repository provides a comprehensive template for integrating the Responsible AI Dashboard into Azure Machine Learning workflows. The template supports both **classification** and **survival analysis** models, with advanced features for model preparation and explainability analysis.

Additional information can be found in the Azure documentation: https://learn.microsoft.com/en-us/azure/machine-learning/how-to-responsible-ai-insights-sdk-cli?view=azureml-api-2&tabs=yaml

## Features

- **Dual-Mode Support**: Works with both classification models and survival analysis models (automatically adapted to binary classification)
- **Extra Features Integration**: Allows adding features to the RAI dashboard that weren't used in the original model training
- **Custom Model Preparation Component**: Handles model wrapping, feature expansion, and data filtering
- **Built-in Azure RAI Components**: Uses Microsoft AzureML registry components for error analysis, insight gathering, and more
- **Pipeline Automation**: Ready-to-use YAML pipelines for `az ml job create` submission

## Folder Structure

```plaintext
responsible-ai-dashboard-template/
├── components/                              # Custom component definitions
│   └── prepare_model_and_data_for_rai/     # Main preparation component
│       ├── component.yml                    # Component specification
│       ├── main.py                          # Main execution script
│       ├── risk_wrapper.py                  # Survival model wrapper
│       └── utils.py                         # Utility functions
├── classification-pipeline.yml              # Pipeline for classification models
├── survival-pipeline.yml                    # Pipeline for survival analysis models
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
5. **Registers the prepared model** for RAI dashboard consumption
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


