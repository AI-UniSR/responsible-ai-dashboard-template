"""
This component is used to convert a survival analysis model into a classification model.
It takes as input a registered survival model (model name and version), training data and test data (optional, training data will be used if test data is not provided).
It outputs train and test data converted into classification format, and a registered classification model.
"""
import argparse
import os
import mlflow
import pandas as pd
from azure.identity import DefaultAzureCredential
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import mltable
from utils import load_data, generate_model_info_json, write_filtered_mltable, build_selector_pipeline, register_model, validate_and_clean_data
from risk_wrapper import SkSurvRiskWrapper, prepare_binary_classification_dataset
import json

# --- Edit and process dataframe if needed ---
# --- Edit and process dataframe if needed ---
# --- Edit and process dataframe if needed ---
def process_df(df, id_column="id"):
    # Placeholder for any preprocessing steps if needed
    # drop id column if exists
    if id_column in df.columns:
        df = df.drop(columns=[id_column])
    return df

def parse_json_list(s):
    try:
        lst = json.loads(s)
        if not isinstance(lst, list):
            raise ValueError("Expected a JSON list.")
        return lst
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON list: {e.msg} (line {e.lineno}, col {e.colno})")

def main():
    parser = argparse.ArgumentParser()
    # --- Model inputs ---
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_version", type=str, required=False, default=None)
    parser.add_argument("--model_info_path_input", type=str, required=False, default=None,
                        help="Alternative to model_version - reads version from model_info.json")
    parser.add_argument("--binary_label_name", type=str, required=True,
                         help="Name of the binary label column in the classification dataset or name of the binary label column for survival analysis will have.")
    # --- Data inputs/outputs ---
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--train_data_out", type=str, required=True)
    parser.add_argument("--test_data_out", type=str, required=True)
    parser.add_argument("--model_info_path", type=str, required=True)
    # --- Survival to classification params ---
    parser.add_argument("--time_column", type=str, required=False, default=None)
    parser.add_argument("--event_column", type=str, required=False, default=None)
    parser.add_argument("--time_horizon", type=float, required=False, default=None)
    # --- Optional extra features ---
    parser.add_argument(
        "--extra_features", type=str, help="Optional[List[str]]", default=None, required=False
    )
    # --- Optional test_assets for threshold override ---
    parser.add_argument(
        "--test_assets", type=str, help="Path to folder with test_assets.json containing threshold", default=None, required=False
    )
    # --- Missing value handling ---
    parser.add_argument(
        "--remove_missing", type=str, help="Comma-separated list of columns to check for NaN before dropping rows. Use '*' or 'all' for all columns.", default=None, required=False
    )
    args = parser.parse_args()

    # --- Load model_version from model_info_path_input if provided ---
    if args.model_info_path_input:
        model_info_path = os.path.join(args.model_info_path_input, "model_info.json")
        if os.path.exists(model_info_path):
            with open(model_info_path, "r") as f:
                model_info = json.load(f)
            # Parse "model_name:version" format
            _, version_str = model_info["id"].split(":")
            args.model_version = version_str
            print(f"Loaded model_version from model_info_path_input: {args.model_version}")
        else:
            raise FileNotFoundError(f"model_info.json not found at {model_info_path}")
    elif not args.model_version:
        raise ValueError("Must provide either --model_version or --model_info_path_input")

    if args.extra_features:
        args.extra_features = parse_json_list(args.extra_features)

    # --- Load threshold from test_assets if provided ---
    threshold_override = None
    if args.test_assets:
        test_assets_path = os.path.join(args.test_assets, "test_assets.json")
        if os.path.exists(test_assets_path):
            with open(test_assets_path, "r") as f:
                test_assets_data = json.load(f)
            threshold_override = test_assets_data.get("threshold")
            print(f"Loaded threshold override from test_assets: {threshold_override}")
        else:
            print(f"Warning: test_assets provided but test_assets.json not found at {test_assets_path}")

    print("Loading input MLTable datasets...")
    # Load the training and test data
    df_train = load_data(args.train_data)
    df_train = process_df(df_train)
    df_test = load_data(args.test_data)
    df_test = process_df(df_test)

    # Handle missing values if remove_missing is specified
    remove_missing_list = None
    if args.remove_missing:
        remove_missing_list = [col.strip() for col in args.remove_missing.split(',')]
        print(f"\nMissing value handling enabled: {remove_missing_list}")
        df_train = validate_and_clean_data(df_train, remove_missing=remove_missing_list, stage_name="training")
        df_test = validate_and_clean_data(df_test, remove_missing=remove_missing_list, stage_name="test")
    else:
        print("\nNo missing value removal specified. Preserving all rows.")

    model_path = f"models:/{args.model_name}/{args.model_version}"
    out_model_name = args.model_name
    out_model_version = args.model_version

    # --- Determine mode and process accordingly ---
    if args.time_column is not None and args.event_column is not None and args.time_horizon is not None:
        # Survival mode
        print("Survival mode detected. Loading survival model and preparing classification datasets...")
        surv_model = mlflow.sklearn.load_model(model_path)
        # --- Prepare binary classification datasets ---
        df_train = prepare_binary_classification_dataset(df_train, args.time_column, args.event_column, args.time_horizon, args.binary_label_name)
        df_test = prepare_binary_classification_dataset(df_test, args.time_column, args.event_column, args.time_horizon, args.binary_label_name)
        used_features = list(surv_model.feature_names_in_)
        features = [f for f in used_features if f != args.binary_label_name]
        # update
        out_model_name = "rai-" + out_model_name + "-clf"
        # Use threshold from test_assets if provided, otherwise default to 1.0
        wrapper_threshold = threshold_override if threshold_override is not None else 1.0
        print(f"Using threshold for SkSurvRiskWrapper: {wrapper_threshold}")
        model = SkSurvRiskWrapper(surv_model, threshold=wrapper_threshold, decision_threshold=0.5)
        if args.extra_features:
            print("Survival mode with extra features detected.")
            print("Determining used features from model...")
            used_features = list(model.feature_names_in_)
            print(f"Model features ({len(used_features)}): {used_features}")

            print(f"Adding extra features: {args.extra_features}")
            features += [f for f in args.extra_features if f not in features]

            print(f"Final feature list ({len(features)}): {features}")
            # Survival mode with extra features
            # --- Expand model input features using ColumnTransformer ---
            model = build_selector_pipeline(
                model_name=args.model_name,
                model_version=args.model_version,
                reference_df=df_train,
                target_col=args.binary_label_name
            )
            _ , out_model_version = register_model(model, out_model_name, 
                                                           code_paths=["./risk_wrapper.py"],
                                                           input_example=df_train[features].iloc[:5,:])
        else:
            # Survival mode without extra features
            _ , out_model_version = register_model(model, out_model_name, 
                                                           code_paths=["./risk_wrapper.py"],
                                                           input_example=df_train[features].iloc[:5,:])
        
    elif args.time_column is None and args.event_column is None and args.time_horizon is None:
        # Classification mode
        print("Classification mode detected. Loading classification model...")
        model = mlflow.sklearn.load_model(model_path)  # caricato dopo!
        used_features = list(model.feature_names_in_)
        features = [f for f in used_features if f != args.binary_label_name]
        # if there are features in extra_features, expand the model input features
        if args.extra_features:
            # Classification mode with extra features
            print("Classification mode with extra features detected.")
            print("Determining used features from model...")
            print(f"Model features ({len(used_features)}): {used_features}")

            print(f"Adding extra features: {args.extra_features}")
            features += [f for f in args.extra_features if f not in features]

            print(f"Final feature list ({len(features)}): {features}")
            # --- Expand model input features using ColumnTransformer ---
            model = build_selector_pipeline(
                model_name=args.model_name,
                model_version=args.model_version,
                reference_df=df_train,
                target_col=args.binary_label_name
            )
            _ , out_model_version = register_model(model, out_model_name, input_example=df_train[features].iloc[:5,:])
    else:
        raise ValueError("Either provide all of time_column, event_column, time_horizon for survival mode, or none of them for classification mode.")
    
    # --- Generate model info JSON ---
    generate_model_info_json(
        model_name=out_model_name,
        model_version=str(out_model_version),
        output_dir=args.model_info_path
    )

    # --- Generate filtered MLTable outputs ---
    write_filtered_mltable(df_train, features=features, target_col=args.binary_label_name, output_dir=args.train_data_out)
    write_filtered_mltable(df_test, features=features, target_col=args.binary_label_name, output_dir=args.test_data_out)
    print("Filtered MLTable datasets created for train/test.")
    print("Component completed successfully.")

if __name__ == "__main__":
    main()