import os
import pandas as pd
from pathlib import Path
import yaml
import json
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import mlflow.sklearn


def validate_and_clean_data(
    df: pd.DataFrame, 
    remove_missing: list = None, 
    stage_name: str = "dataset"
) -> pd.DataFrame:
    """
    Validate dataset and remove rows with missing values based on specified strategy.
    
    Args:
        df: DataFrame to validate
        remove_missing: Strategy for removing missing values:
            - None or []: Preserve all rows (no removal)
            - ["*"] or ["all"]: Remove rows with any missing value
            - List of column names: Remove rows with missing values only in those columns
        stage_name: Name for logging (e.g., "training", "test")
    
    Returns:
        Cleaned DataFrame
    
    Raises:
        ValueError: If all rows would be removed or if specified columns don't exist
    """
    initial_rows = len(df)
    
    # Parse removal strategy
    if not remove_missing or len(remove_missing) == 0:
        print(f"[{stage_name}] Preserving all rows (no missing value removal)")
        return df
    
    # Check for wildcard indicators
    if "*" in remove_missing or "all" in remove_missing:
        print(f"[{stage_name}] Removing rows with ANY missing value...")
        dropna_cols = None  # Will check all columns
    else:
        # Check if specified columns exist
        missing_cols = [col for col in remove_missing if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in {stage_name} data: {missing_cols}")
        dropna_cols = remove_missing
        print(f"[{stage_name}] Removing rows with missing values in: {dropna_cols}")
    
    # Report missing values before cleaning
    if dropna_cols:
        subset_missing = df[dropna_cols].isna().sum()
        if subset_missing.sum() > 0:
            print(f"  Missing values detected:")
            for col in dropna_cols:
                if subset_missing[col] > 0:
                    print(f"    - {col}: {subset_missing[col]} ({subset_missing[col]/initial_rows:.1%})")
    else:
        all_missing = df.isna().sum()
        if all_missing.sum() > 0:
            print(f"  Missing values detected:")
            for col, count in all_missing[all_missing > 0].items():
                print(f"    - {col}: {count} ({count/initial_rows:.1%})")
    
    # Remove rows with NaN
    if dropna_cols:
        df_clean = df.dropna(subset=dropna_cols)
    else:
        df_clean = df.dropna()
    
    rows_removed = initial_rows - len(df_clean)
    
    # Validate we still have data
    if len(df_clean) == 0:
        raise ValueError(
            f"All rows removed from {stage_name} set after dropping NaN. "
            f"Original: {initial_rows} rows. "
            f"Consider adjusting remove_missing parameters."
        )
    
    # Log results
    if rows_removed > 0:
        print(f"  Removed {rows_removed} rows ({rows_removed/initial_rows:.1%})")
        print(f"  Remaining: {len(df_clean)} rows")
        
        if rows_removed / initial_rows > 0.5:
            print(f"  ⚠️  WARNING: More than 50% of {stage_name} data removed due to missing values")
    else:
        print(f"  No missing values found. All {len(df_clean)} rows retained.")
    
    return df_clean


def load_data(path):
    """
    Load data from path into a pandas DataFrame. Supports Folder, File and MLTable formats.
    """
    if path.endswith("mltable"):
        import mltable
        mltable_data = mltable.load(path)
        df = mltable_data.to_pandas_dataframe()
    elif os.path.isdir(path):
        csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in folder: {path}")
        csv_path = os.path.join(path, csv_files[0])
        df = pd.read_csv(csv_path)
    elif os.path.isfile(path) and path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported data format or path does not exist: {path}")
    return df


def generate_model_info_json(model_name, model_version, output_dir):
    """
    Create JSON file with model name and version inside the given output folder.
    """
    model_info = {"id": f"{model_name}:{model_version}"}
    print(f"Generating model info JSON for model {model_info['id']}")

    # Ensure the folder exists (AzureML doesn't create it automatically)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "model_info.json")
    with open(output_file, "w") as f:
        json.dump(model_info, f)

    print(f"Model info saved to {output_file}")


def write_filtered_mltable(df, features, target_col, output_dir):
    """
    Save a minimal MLTable with selected features and target column.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Keep only relevant columns
    cols = [c for c in features + [target_col] if c in df.columns]
    df_filtered = df[cols]
    df_filtered.to_csv(output_path / "data.csv", index=False)

    # Minimal MLTable definition (official structure)
    mltable_def = {
        "paths": [{"file": "data.csv"}],
        "transformations": [{"read_delimited": {"delimiter": ","}}],
    }

    with open(output_path / "MLTable", "w") as f:
        yaml.safe_dump(mltable_def, f, sort_keys=False)

    return output_path



def build_selector_pipeline(model_name, model_version, reference_df, target_col):
    """Return a sklearn Pipeline that filters extra columns before model prediction.

    The selector hands the model a numpy array. A model that cannot predict from that array (for
    example a pipeline that selects its columns by name) gets a DataFrame with its own column names
    instead, as in the survival branch. Every model that can predict from the array keeps it: other
    RAI tools rely on that (dice-ml's object-dtype frames, which XGBoost and LightGBM reject as a
    DataFrame, are fine as an array). LightGBM and XGBoost models whose categories the array would
    misread (see category_codes_error) are refused when their names are a booster's numpy.str_, which
    failed before; otherwise, as before, they only get a warning.
    """
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)

    if not hasattr(model, "feature_names_in_"):
        raise AttributeError("Model has no attribute 'feature_names_in_'.")

    category_error = category_codes_error(model)
    if category_error:
        if any(type(name) is not str for name in model.feature_names_in_):
            # numpy.str_ names (a booster's, see below) failed before plain-string names: refuse
            # these models rather than let them through with wrong scores
            raise ValueError(category_error)
        print(f"Warning: {category_error}")

    # str(), as in the survival branch: LightGBM (>= 4.5) and XGBoost name their features with
    # numpy.str_, which scikit-learn's ColumnTransformer rejects ("No valid specification").
    used_features = [str(f) for f in model.feature_names_in_]
    X_ref = reference_df.drop(columns=[target_col], errors="ignore")
    selector = ColumnTransformer(
        [("keep", "passthrough", used_features)],
        remainder="drop"
    )

    selector.fit(X_ref)

    pipeline = Pipeline([
        ("selector", selector),
        ("model", model)
    ])
    error = _prediction_error(pipeline, X_ref)
    if error is None:
        return pipeline

    by_name = ColumnTransformer(
        [("keep", "passthrough", used_features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    by_name.set_output(transform="pandas")
    by_name.fit(X_ref)
    by_name_pipeline = Pipeline([
        ("selector", by_name),
        ("model", model)
    ])
    by_name_error = _prediction_error(by_name_pipeline, X_ref)
    if by_name_error is not None:
        # Neither works: keep the original selector, so the model fails as it did before (in the RAI
        # constructor, or here when replay computes the predictions).
        print(f"Warning: the model cannot predict on the training rows "
              f"(numpy array: {error}; DataFrame: {by_name_error}).")
        return pipeline
    print(f"The model cannot predict from a numpy array ({error}); "
          "the selector hands it a DataFrame with its own column names.")
    return by_name_pipeline


def _prediction_error(pipeline, X, n_rows=5):
    """Return the error the pipeline raises on the first rows of X, or None if it predicts."""
    rows = X.head(n_rows)
    try:
        pipeline.predict(rows)
        if hasattr(pipeline, "predict_proba"):
            pipeline.predict_proba(rows)
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def category_codes_error(model):
    """Describe a LightGBM or XGBoost estimator in model that reads raw values as category codes
    without a check, or return None.

    The template never rebuilds pandas 'category' dtypes, so these boosters get the raw values and read
    them as category codes, which is right only when the categories are exactly 0..k-1. LightGBM keeps
    its categories, so 0..k-1 passes; XGBoost before 3.1 does not, so any categorical feature is flagged.
    """
    for booster in _boosters(model):
        library = _booster_library(booster)
        try:
            if library == "lightgbm":
                for categories in getattr(booster.booster_, "pandas_categorical", None) or []:
                    if list(categories) != list(range(len(categories))):
                        return (f"LightGBM model fitted on pandas 'category' columns whose categories are "
                                f"{list(categories)[:5]}, not 0..k-1: it reads the raw values it gets as "
                                "category codes, so it would score wrongly. Before training, recode those "
                                "columns so that the categories are exactly 0..k-1.")
            elif "c" in (booster.get_booster().feature_types or []):
                return ("XGBoost model with categorical features (pandas 'category' columns or feature_types "
                        "'c'): it reads the raw values it gets as category codes, which the component cannot "
                        "check. Before training, encode those columns as plain numeric columns.")
        except Exception:
            continue  # not fitted, e.g. the template a search object keeps next to best_estimator_
    return None


def _booster_library(estimator):
    """'lightgbm' or 'xgboost' if the estimator's class comes from that library, else None."""
    for cls in type(estimator).__mro__:
        library = cls.__module__.split(".")[0]
        if library in ("lightgbm", "xgboost"):
            return library
    return None


def _boosters(model, depth=0):
    """The LightGBM and XGBoost estimators in model: the model itself, or estimators it calls (Pipeline
    steps, a search's best_estimator_, calibration and ensemble members)."""
    if _booster_library(model):
        return [model]
    if depth > 5:
        return []
    found = []
    for attribute in ("steps", "best_estimator_", "calibrated_classifiers_", "estimator", "estimators_",
                      "final_estimator_"):
        try:
            value = getattr(model, attribute, None)
        except Exception:
            value = None
        for item in value if isinstance(value, (list, tuple)) else [value]:
            if isinstance(item, tuple):  # Pipeline steps are (name, estimator)
                item = item[-1]
            if item is not None and not isinstance(item, str):
                found += _boosters(item, depth + 1)
    return found

def register_model(mlflow_model, model_name, input_example=None, code_paths=None):
    """
    Register the given MLflow model under the specified name.
    """
    mlflow.sklearn.log_model(
        sk_model=mlflow_model,
        artifact_path='model',
        registered_model_name=model_name,
        input_example=input_example,
        code_paths=code_paths
    )
    # Get model registration info
    run = mlflow.active_run()
    client = mlflow.tracking.MlflowClient()
    latest_model = client.get_latest_versions(model_name, stages=["None"])[0]
    print(f"✅ Model registered: {latest_model.name} (version {latest_model.version})")
    return latest_model.name, latest_model.version