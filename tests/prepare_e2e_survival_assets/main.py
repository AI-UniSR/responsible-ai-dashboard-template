"""
E2E Test Component: Generate synthetic survival data, train model, and prepare assets for RAI testing.

This component generates:
1. Synthetic survival data with ~500 patients
2. A trained GradientBoostingSurvivalAnalysis model (scikit-survival)
3. Train/Test MLTable datasets
4. test_assets folder with threshold JSON (66th percentile of predictions on test set)
"""
import argparse
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import mlflow
from sklearn.model_selection import train_test_split
from sksurv.ensemble import GradientBoostingSurvivalAnalysis


# ============================================================================
# Data Generation
# ============================================================================
def generate_synthetic_survival_data(
    n_samples: int = 500, 
    random_state: int = 42,
    missing_value_columns: list = None,
    missing_value_fraction: float = 0.0
) -> pd.DataFrame:
    """
    Generate synthetic survival data with realistic feature distributions.
    
    Features:
    - age: Patient age (continuous, 18-90)
    - bmi: Body mass index (continuous, 15-45)
    - sex: Binary categorical (0=Female, 1=Male)
    - tumor: Categorical tumor type (0=Benign, 1=Malignant_A, 2=Malignant_B)
    - biomarker_1, biomarker_2: Continuous biomarkers
    - tte: Time-to-event (derived from features + noise)
    - event: Event indicator (boolean)
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        missing_value_columns: List of columns to inject NaN values (for testing)
        missing_value_fraction: Fraction of rows to set as NaN in specified columns (0.0-1.0)
    
    Returns:
        pd.DataFrame with columns: age, bmi, sex, tumor, biomarker_1, biomarker_2, tte, event
    """
    np.random.seed(random_state)
    
    # Generate features
    age = np.random.uniform(18, 90, n_samples)
    bmi = np.random.normal(25, 5, n_samples).clip(15, 45)
    sex = np.random.choice([0, 1], n_samples, p=[0.45, 0.55])
    tumor = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.35, 0.25])
    biomarker_1 = np.random.exponential(2, n_samples)
    biomarker_2 = np.random.normal(0, 1, n_samples)
    
    # Generate survival times (log-linear hazard model)
    log_hazard = (
        0.02 * age 
        + 0.03 * bmi 
        + 0.5 * sex 
        + 0.8 * (tumor == 2).astype(float)
        + 0.3 * (tumor == 1).astype(float)
        + 0.2 * biomarker_1 
        + 0.1 * biomarker_2
    )
    
    # Generate time-to-event using exponential distribution scaled by hazard
    baseline_time = 24  # months
    tte = np.random.exponential(baseline_time / np.exp(log_hazard - log_hazard.mean()))
    tte = np.clip(tte, 0.1, 60)  # Clip to realistic range (0.1 to 60 months)
    
    # Generate censoring (administrative censoring + random censoring)
    max_followup = 36  # months
    censoring_time = np.random.uniform(6, max_followup, n_samples)
    
    # Event indicator: True if event happened before censoring
    event = tte <= censoring_time
    tte = np.minimum(tte, censoring_time)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age.round(1),
        'bmi': bmi.round(1),
        'sex': sex,
        'tumor': tumor,
        'biomarker_1': biomarker_1.round(3),
        'biomarker_2': biomarker_2.round(3),
        'tte': tte.round(2),
        'event': event
    })
    
    # Inject missing values if requested
    if missing_value_columns and missing_value_fraction > 0:
        np.random.seed(random_state)  # Ensure reproducibility
        for col in missing_value_columns:
            if col not in df.columns:
                print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
                continue
            n_missing = int(n_samples * missing_value_fraction)
            missing_indices = np.random.choice(df.index, size=n_missing, replace=False)
            df.loc[missing_indices, col] = np.nan
            print(f"Injected {n_missing} missing values ({missing_value_fraction:.1%}) in column '{col}'")
    
    return df


# ============================================================================
# MLTable Utilities
# ============================================================================
def save_as_mltable(df: pd.DataFrame, output_path: str) -> None:
    """Save DataFrame as MLTable format (CSV + MLTable YAML)."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_path = output_path / "data.csv"
    df.to_csv(csv_path, index=False)
    
    # Create MLTable YAML
    mltable_yaml = {
        "paths": [{"file": "./data.csv"}],
        "transformations": [{"read_delimited": {"delimiter": ","}}]
    }
    
    with open(output_path / "MLTable", "w") as f:
        yaml.dump(mltable_yaml, f)
    
    print(f"Saved MLTable to {output_path} with {len(df)} rows")


# ============================================================================
# Data Validation and Cleaning
# ============================================================================
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
            f"Consider adjusting missing_value_fraction or remove_missing parameters."
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


# ============================================================================
# Model Training
# ============================================================================
def train_survival_model(df_train: pd.DataFrame, feature_cols: list, time_col: str, event_col: str):
    """
    Train a GradientBoostingSurvivalAnalysis model.
    
    Args:
        df_train: Training DataFrame
        feature_cols: List of feature column names
        time_col: Name of time-to-event column
        event_col: Name of event indicator column
    
    Returns:
        Trained scikit-survival model
    """
    X_train = df_train[feature_cols].values
    
    # Create structured array for survival target (required by scikit-survival)
    y_train = np.array(
        [(e, t) for e, t in zip(df_train[event_col], df_train[time_col])],
        dtype=[('event', bool), ('time', float)]
    )
    
    # Train model with reasonable hyperparameters for small dataset
    model = GradientBoostingSurvivalAnalysis(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Store feature names for compatibility with RAI wrapper
    model.feature_names_in_ = np.array(feature_cols)
    
    return model


def calculate_threshold_percentile(model, df_test: pd.DataFrame, feature_cols: list, percentile: float = 66) -> float:
    """
    Calculate the threshold based on percentile of risk predictions on test set.
    
    Args:
        model: Trained survival model
        df_test: Test DataFrame
        feature_cols: List of feature column names
        percentile: Percentile to use (default 66)
    
    Returns:
        Threshold value at the specified percentile
    """
    X_test = df_test[feature_cols].values
    risk_scores = model.predict(X_test)
    threshold = np.percentile(risk_scores, percentile)
    return float(threshold)


def register_model_mlflow(model, model_name: str, input_example: pd.DataFrame) -> int:
    """
    Register model with MLflow.
    
    Args:
        model: Trained sklearn-compatible model
        model_name: Name for registered model
        input_example: Example input for model signature
    
    Returns:
        Model version number
    """
    with mlflow.start_run():
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='model',
            registered_model_name=model_name,
            input_example=input_example
        )
        
        # Get the version from the registered model
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        latest_version = max([int(v.version) for v in versions])
        
    print(f"Registered model '{model_name}' version {latest_version}")
    return latest_version


def save_test_assets(output_path: str, threshold: float, percentile: float = 66) -> None:
    """
    Save test assets JSON with threshold information.
    
    Args:
        output_path: Path to output folder
        threshold: Calculated threshold value
        percentile: Percentile used for threshold calculation
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    test_assets = {
        "threshold": threshold,
        "percentile": percentile,
        "description": f"Threshold calculated as {percentile}th percentile of risk predictions on test set"
    }
    
    with open(output_path / "test_assets.json", "w") as f:
        json.dump(test_assets, f, indent=2)
    
    print(f"Saved test_assets.json to {output_path}")


def generate_model_info_json(model_name: str, model_version: str, output_dir: str) -> None:
    """
    Create JSON file with model name and version.
    """
    model_info = {"id": f"{model_name}:{model_version}"}
    print(f"Generating model info JSON for model {model_info['id']}")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "model_info.json")
    
    with open(output_file, "w") as f:
        json.dump(model_info, f)

    print(f"Model info saved to {output_file}")


# ============================================================================
# Main
# ============================================================================
def parse_args():
    # DEBUG: Print exactly what arguments are being received
    import sys
    print("=" * 80)
    print("DEBUG: sys.argv received:")
    for i, arg in enumerate(sys.argv):
        print(f"  [{i}]: {repr(arg)}")
    print("=" * 80)
    
    parser = argparse.ArgumentParser(description="Generate synthetic survival data and train model for E2E testing")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, required=True, help="Name for the registered model")
    parser.add_argument("--time_horizon", type=int, default=12, help="Time horizon for binary classification (months)")
    parser.add_argument("--n_samples", type=int, default=500, help="Number of synthetic samples to generate")
    parser.add_argument("--test_size", type=float, default=0.3, help="Proportion of data for test set")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    
    # Missing value handling
    parser.add_argument("--missing_value_columns", type=str, default=None, 
                        help="Comma-separated list of columns to inject missing values (e.g., 'age,bmi')")
    parser.add_argument("--missing_value_fraction", type=float, default=0.0, 
                        help="Fraction of rows to set as NaN in specified columns (0.0-1.0)")
    parser.add_argument("--remove_missing", type=str, default=None, 
                        help="Comma-separated list of columns to check for NaN before dropping rows. Use '*' or 'all' to check all columns. Empty list preserves all rows.")
    
    # Outputs
    parser.add_argument("--train_data", type=str, required=True, help="Output path for training MLTable")
    parser.add_argument("--test_data", type=str, required=True, help="Output path for test MLTable")
    parser.add_argument("--model_info_path", type=str, required=True, help="Output path for model info JSON")
    parser.add_argument("--test_assets", type=str, required=True, help="Output path for test assets (threshold JSON)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("E2E Test: Generate Synthetic Survival Assets")
    print("=" * 80)
    
    # Define feature columns and survival columns
    feature_cols = ['age', 'bmi', 'sex', 'tumor', 'biomarker_1', 'biomarker_2']
    time_col = 'tte'
    event_col = 'event'
    
    # Parse missing value parameters
    missing_cols = None
    if args.missing_value_columns:
        missing_cols = [col.strip() for col in args.missing_value_columns.split(',')]
        print(f"Missing value injection enabled for columns: {missing_cols}")
        print(f"Missing value fraction: {args.missing_value_fraction}")
    
    # Step 1: Generate synthetic data
    print("\n[Step 1] Generating synthetic survival data...")
    df = generate_synthetic_survival_data(
        n_samples=args.n_samples, 
        random_state=args.random_state,
        missing_value_columns=missing_cols,
        missing_value_fraction=args.missing_value_fraction
    )
    print(f"Generated {len(df)} samples")
    print(f"Event rate: {df[event_col].mean():.2%}")
    print(f"Median time-to-event: {df[time_col].median():.1f} months")
    
    # Step 2: Split into train/test
    print("\n[Step 2] Splitting into train/test sets...")
    df_train, df_test = train_test_split(
        df, 
        test_size=args.test_size, 
        random_state=args.random_state,
        stratify=df[event_col]  # Stratify by event to maintain event rate
    )
    print(f"Train set: {len(df_train)} samples (event rate: {df_train[event_col].mean():.2%})")
    print(f"Test set: {len(df_test)} samples (event rate: {df_test[event_col].mean():.2%})")
    
    # Step 2b: Clean datasets (remove rows with missing values based on strategy)
    remove_missing_list = None
    if args.remove_missing:
        remove_missing_list = [col.strip() for col in args.remove_missing.split(',')]
    
    print("\n[Step 2b] Validating and cleaning datasets...")
    df_train = validate_and_clean_data(df_train, remove_missing=remove_missing_list, stage_name="training")
    df_test = validate_and_clean_data(df_test, remove_missing=remove_missing_list, stage_name="test")
    
    # Step 3: Train survival model
    print("\n[Step 3] Training GradientBoostingSurvivalAnalysis model...")
    model = train_survival_model(df_train, feature_cols, time_col, event_col)
    print("Model trained successfully")
    
    # Step 4: Calculate threshold (66th percentile on test set)
    print("\n[Step 4] Calculating threshold (66th percentile on test set)...")
    threshold = calculate_threshold_percentile(model, df_test, feature_cols, percentile=66)
    print(f"Threshold (66th percentile): {threshold:.4f}")
    
    # Step 5: Register model with MLflow
    print("\n[Step 5] Registering model with MLflow...")
    input_example = df_train[feature_cols].iloc[:5]
    model_version = register_model_mlflow(model, args.model_name, input_example)
    
    # Step 6: Save outputs
    print("\n[Step 6] Saving outputs...")
    
    # Save train/test MLTables
    save_as_mltable(df_train, args.train_data)
    save_as_mltable(df_test, args.test_data)
    
    # Save model info JSON
    generate_model_info_json(args.model_name, str(model_version), args.model_info_path)
    
    # Save test assets (threshold JSON)
    save_test_assets(args.test_assets, threshold, percentile=66)
    
    print("\n" + "=" * 80)
    print("E2E Test Assets Generation Complete!")
    print("=" * 80)
    print(f"Model: {args.model_name}:{model_version}")
    print(f"Train data: {args.train_data}")
    print(f"Test data: {args.test_data}")
    print(f"Model info: {args.model_info_path}")
    print(f"Test assets (threshold): {args.test_assets}")


if __name__ == "__main__":
    main()
