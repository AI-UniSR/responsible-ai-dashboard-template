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
    """Return a sklearn Pipeline that filters extra columns before model prediction."""
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)

    if not hasattr(model, "feature_names_in_"):
        raise AttributeError("Model has no attribute 'feature_names_in_'.")

    used_features = list(model.feature_names_in_)
    selector = ColumnTransformer(
        [("keep", "passthrough", used_features)],
        remainder="drop"
    )

    selector.fit(reference_df.drop(columns=[target_col], errors="ignore"))

    return Pipeline([
        ("selector", selector),
        ("model", model)
    ])

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