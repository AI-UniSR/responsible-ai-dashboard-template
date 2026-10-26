"""
Prediction replay mode of prepare_model_and_data_for_rai (opt-in: --replay_predictions true).

By default the Azure ML RAI components load the registered model inside their own environment
(responsibleai-tabular: Python 3.9 or 3.10, pandas 1.5, scikit-learn 1.5), so the model must be
built with compatible libraries.

In replay mode this component, running in the model's own environment:
  1. computes predict/predict_proba on exactly the rows it writes for the RAI components;
  2. registers an MLflow model that only replays those predictions
     (see rai_prediction_replay_loader.py).

The RAI components then work with any model library for everything that predicts on those
rows: error analysis, explanations, causal analysis, data balance, the score card and the
dashboard. Counterfactuals, what-if and ICE predict on new rows, so they are not available.
"""
import json
import os
import platform
import re
import tempfile
import warnings

import numpy as np
import pandas as pd

from rai_prediction_replay_loader import FORMAT_VERSION, META_FILE, PREDICTIONS_FILE, SPLITS, read_rai_dataset
from utils import write_filtered_mltable

LOADER_MODULE = "rai_prediction_replay_loader"
LOADER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOADER_MODULE + ".py")
# Nothing needs installing: everything the loader imports is in the RAI environment. Listing
# packages would let use_model_dependency (which must stay false) upgrade them in that environment.
REPLAY_PIP_REQUIREMENTS = []
# ml-wrappers stores the predictions in columns named target, target_0, target_1, ...
_RESERVED_NAME = re.compile(r"^target(_\d+)?$")
# Characters pandas 1.5.3 (the RAI environment) can turn into a backtick-quoted query name: Python's
# one-character operator tokens plus the extras it maps itself (create_valid_python_identifier).
_QUERY_NAME_PUNCTUATION = set("%&()*+,-./:;<=>@[]^{|}~ ?!$€°")
_FORBIDDEN_NAME_CHARACTERS = set("#'\"\\`\r\n")
# pandas.read_json, used by RAIInsights to reload data, turns date-like columns whose values are all
# above this stamp into datetimes (keep_default_dates).
_MIN_JSON_DATE_STAMP = 31536000


def _looks_numeric(name):
    try:
        float(name)
        return True
    except ValueError:
        return False


def _query_name_ok(name):
    if set(name) & _FORBIDDEN_NAME_CHARACTERS:
        return False
    mapped = "".join("_" if character in _QUERY_NAME_PUNCTUATION else character for character in name)
    return ("BACKTICK_QUOTED_STRING_" + mapped).isidentifier()


def _date_like_name(name):
    lowered = name.lower()
    return (lowered.endswith("_at") or lowered.endswith("_time") or lowered in ("modified", "date", "datetime")
            or lowered.startswith("timestamp"))


def _parse_as_dates(text):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(text, errors="coerce").notna().all()


def validate_replay_data(df_train, df_test, features):
    """Reject data the RAI-side lookup cannot replay, before anything is registered.

    The lookup (ml-wrappers PredictionsModelWrapper; 0.6.0 in responsibleai-tabular:28) finds a row
    by turning its values into a pandas query, and RAIInsights reloads the data through JSON.
    The names and values rejected here would make the RAI components fail later, with obscure
    errors.
    """
    problems = []
    if len(features) < 2:
        problems.append("at least two feature columns are needed (one-column rows cannot be looked up)")
    for name in features:
        if not isinstance(name, str):
            problems.append("column name {!r} is not a string".format(name))
            continue
        if _RESERVED_NAME.match(name):
            problems.append("column name '{}' is reserved by ml-wrappers for the predictions".format(name))
        if _looks_numeric(name):
            problems.append("column name '{}' looks like a number (the RAI JSON reload converts it)".format(name))
        if not _query_name_ok(name):
            problems.append("column name '{}' cannot be used in a pandas query (avoid #, quotes, backslashes, "
                            "backticks, line breaks and symbols such as ² or ≥)".format(name))
    for split, df in (("train", df_train), ("test", df_test)):
        for name in features:
            column = df[name]
            if pd.api.types.is_datetime64_any_dtype(column) or pd.api.types.is_timedelta64_dtype(column):
                problems.append("{} column '{}' holds dates or durations".format(split, name))
            elif pd.api.types.is_bool_dtype(column):
                continue
            elif pd.api.types.is_numeric_dtype(column):
                values = column.dropna()
                if isinstance(name, str) and _date_like_name(name) and len(values) and (values > _MIN_JSON_DATE_STAMP).all():
                    problems.append("{} column '{}' has a date-like name and numbers that the RAI JSON reload "
                                    "turns into dates".format(split, name))
            else:
                # text, whatever its dtype (object, string, category)
                text = column.dropna().astype(str)
                if text.str.contains("'", regex=False).any():
                    problems.append("{} column '{}' has values containing a single quote".format(split, name))
                if text.str.contains("\\", regex=False).any():
                    problems.append("{} column '{}' has values containing a backslash".format(split, name))
                if text.str.contains(r"[\r\n]").any():
                    problems.append("{} column '{}' has values containing line breaks".format(split, name))
                if isinstance(name, str) and _date_like_name(name) and len(text) and _parse_as_dates(text):
                    problems.append("{} column '{}' has a date-like name and values that the RAI JSON reload "
                                    "turns into dates".format(split, name))
    if problems:
        raise ValueError(
            "replay_predictions cannot be used with this data:\n  - " + "\n  - ".join(dict.fromkeys(problems))
        )


def check_data_as_rai_reads_it(data_dir, n_rows, features):
    """Re-read the written replay data the way the RAI components will (MLTable) and validate that.

    MLTable may parse values differently from the in-memory frame (for example dates), and line
    breaks inside values split records.
    """
    try:
        import mltable  # noqa: F401
    except ImportError:
        print("mltable is not installed here: skipping the check of the data as the RAI components read it.")
        return
    frames = {}
    for split in SPLITS:
        frame = read_rai_dataset(os.path.join(data_dir, split))
        if len(frame) != n_rows[split]:
            raise ValueError(
                "replay_predictions: {} {} rows were written but {} are read back as MLTable "
                "(line breaks inside values?)".format(n_rows[split], split, len(frame))
            )
        frames[split] = frame
    validate_replay_data(frames["train"], frames["test"], features)


def compute_predictions(model, df, features):
    """Return (labels, probabilities) of model on df[features], the rows the RAI components see."""
    X = df[features]
    y_pred = np.asarray(model.predict(X))
    y_proba = np.asarray(model.predict_proba(X), dtype=float)
    if y_pred.ndim != 1 or len(y_pred) != len(X):
        raise ValueError("predict returned shape {} for {} rows".format(y_pred.shape, len(X)))
    if y_proba.ndim != 2 or len(y_proba) != len(X):
        raise ValueError("predict_proba returned shape {} for {} rows".format(y_proba.shape, len(X)))
    if not np.all(np.isfinite(y_proba)):
        raise ValueError("predict_proba returned non-finite values")
    return y_pred, y_proba


def _json_values(values):
    """Plain Python values (json-serializable) from a 1-D numpy array."""
    return [value.item() if isinstance(value, np.generic) else value for value in np.asarray(values).tolist()]


def _library_versions():
    import mlflow
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "mlflow": mlflow.__version__,
    }


def build_replay_model(model, df_train, df_test, features, target_col, model_dir, source_model):
    """Save, at model_dir, an MLflow pyfunc model that replays the predictions of model on
    df_train and df_test. Return the metadata stored with it."""
    import mlflow.pyfunc

    features = list(features)
    validate_replay_data(df_train, df_test, features)

    predictions, label_dtypes, n_classes = {}, set(), set()
    for split, df in zip(SPLITS, (df_train, df_test)):
        y_pred, y_proba = compute_predictions(model, df, features)
        predictions[split] = {"y_pred": _json_values(y_pred), "y_proba": y_proba.tolist()}
        label_dtypes.add(str(y_pred.dtype))
        n_classes.add(y_proba.shape[1])
    if len(n_classes) != 1:
        raise ValueError("predict_proba returned a different number of columns on train and test")

    meta = {
        "format_version": FORMAT_VERSION,
        "task_type": "classification",
        "target_column": target_col,
        "features": features,
        # Train and test labels share a dtype unless one split predicts a single, narrower type.
        "label_dtype": label_dtypes.pop() if len(label_dtypes) == 1 else "object",
        "n_classes": n_classes.pop(),
        "n_rows": {"train": len(df_train), "test": len(df_test)},
        "source_model": source_model,
        "created_with": _library_versions(),
    }

    with tempfile.TemporaryDirectory() as staging:
        data_dir = os.path.join(staging, "replay_data")
        for split, df in zip(SPLITS, (df_train, df_test)):
            # The same writer as the component's train/test outputs, so the files are identical.
            write_filtered_mltable(df, features=features, target_col=target_col, output_dir=os.path.join(data_dir, split))
        check_data_as_rai_reads_it(data_dir, meta["n_rows"], features)
        with open(os.path.join(data_dir, PREDICTIONS_FILE), "w") as file:
            json.dump(predictions, file)
        with open(os.path.join(data_dir, META_FILE), "w") as file:
            json.dump(meta, file, indent=2)
        mlflow.pyfunc.save_model(
            path=model_dir,
            loader_module=LOADER_MODULE,
            data_path=data_dir,
            code_paths=[LOADER_FILE],
            pip_requirements=REPLAY_PIP_REQUIREMENTS,
        )
    return meta


def register_replay_model(model_dir, model_name):
    """Register the saved replay model and return (name, version).

    It uses save_model + create_model_version(file://...), as Microsoft's own RAI e2e does: the
    azureml-mlflow plugin uploads the folder, with MLflow 2.x and 3.x alike. If that fails, it
    falls back to logging the folder to the job's run and registering that copy. The registration
    goes directly from the artifact URI, because MLflow 3 register_model would look for a logged
    model.
    """
    import mlflow
    from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    try:
        client.create_registered_model(model_name)
        print("Created registered model {}".format(model_name))
    except Exception as error:
        print("Registered model {} not created ({}); adding a new version.".format(model_name, error))
    try:
        version = client.create_model_version(name=model_name, source="file://" + os.path.abspath(model_dir))
    except Exception as error:
        print("Registering from the local folder failed ({}); registering through the job's run.".format(error))
        mlflow.log_artifacts(model_dir, artifact_path="replay_model")
        run_id = mlflow.active_run().info.run_id
        source = RunsArtifactRepository.get_underlying_uri("runs:/{}/replay_model".format(run_id))
        version = client.create_model_version(name=model_name, source=source, run_id=run_id)
    print("✅ Replay model registered: {} (version {})".format(model_name, version.version))
    return model_name, version.version


def build_and_register_replay_model(model, df_train, df_test, features, target_col, model_name, source_model):
    """Build the replay model of model on df_train/df_test and register it as model_name."""
    with tempfile.TemporaryDirectory() as workdir:
        model_dir = os.path.join(workdir, "replay_model")
        meta = build_replay_model(model, df_train, df_test, features, target_col, model_dir, source_model)
        print(
            "Prediction replay: stored predictions for {} train and {} test rows, {} features, "
            "label dtype {} (Python {}, scikit-learn {}).".format(
                meta["n_rows"]["train"],
                meta["n_rows"]["test"],
                len(meta["features"]),
                meta["label_dtype"],
                meta["created_with"]["python"],
                meta["created_with"]["scikit-learn"],
            )
        )
        return register_replay_model(model_dir, model_name)
