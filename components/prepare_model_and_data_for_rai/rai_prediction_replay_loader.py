"""
Prediction replay loader for the Azure ML Responsible AI components.

This module ships inside the replay model that prepare_model_and_data_for_rai registers when
replay_predictions is true. MLflow imports it inside the RAI component environment
(responsibleai-tabular) and calls _load_pyfunc. That call rebuilds an ml-wrappers
PredictionsModelWrapperClassification from plain files:
  - the datasets the component wrote for RAI;
  - the predictions it computed in the model's own environment.

Nothing pickled in the model's environment is unpickled here. The model.pkl that RAI writes
afterwards references only ml-wrappers, pandas and numpy, all of which are installed in the RAI
environment.

Keep this file loadable in the RAI environment: Python 3.8+, pandas 1.x, numpy 1.x,
ml-wrappers 0.5/0.6. Do not import anything from the component (utils, replay, ...).
"""
import io
import json
import os

import numpy as np
import pandas as pd

FORMAT_VERSION = 1
META_FILE = "replay_meta.json"
PREDICTIONS_FILE = "predictions.json"
SPLITS = ("train", "test")


def read_rai_dataset(folder):
    """Read a dataset folder the way the RAI components do: as MLTable first, then as parquet
    (rai_component_utilities.load_dataset)."""
    try:
        import mltable

        return mltable.load(folder).to_pandas_dataframe()
    except Exception as mltable_error:
        try:
            return pd.read_parquet(folder)
        except Exception:
            raise mltable_error


def rai_json_round_trip(df):
    """Return df as RAIInsights reloads it after save().

    RAIInsights writes train/test with to_json(orient='split'), together with their dtypes, and
    reads them back with read_json(dtype=dtypes, orient='split') (responsibleai
    rai_base_insights._save_data/_load_data). Three things change on the way:
      - to_json rounds floats to 10 decimal places (values below 5e-11 become 0.0);
      - pandas then decodes JSON floats inexactly ("0.3" comes back as 0.30000000000000004);
      - object columns come back with None for missing values.
    So the rows gather, the score card and the dashboard predict on are not the rows the
    component wrote. Doing the same round trip here, with the same pandas, reproduces them
    exactly.
    """
    dtypes = df.dtypes.astype(str).to_dict()
    return pd.read_json(io.StringIO(df.to_json(orient="split")), dtype=dtypes, orient="split")


def build_lookup_table(path, meta, predictions):
    """Return (features, labels, probabilities) for every row the RAI components may ask about.

    Each split is stored three times, with identical copies skipped:
      - as written (constructor and tool components);
      - after one RAIInsights save/load (gather);
      - after two (score card and dashboard, which load what gather saved).
    """
    target = meta["target_column"]
    frames, labels, probas = [], [], []
    for split in SPLITS:
        data = read_rai_dataset(os.path.join(path, split))
        if target not in data.columns:
            raise ValueError("Replay data for '{}' has no target column '{}'.".format(split, target))
        y_pred = predictions[split]["y_pred"]
        y_proba = predictions[split].get("y_proba") or []
        if len(data) != len(y_pred) or (y_proba and len(y_proba) != len(y_pred)):
            raise ValueError(
                "Replay data for '{}' has {} rows but {} predictions.".format(split, len(data), len(y_pred))
            )
        variants = [data]
        once = rai_json_round_trip(data)
        for candidate in (once, rai_json_round_trip(once)):
            if not any(candidate.equals(seen) for seen in variants):
                variants.append(candidate)
        for variant in variants:
            frames.append(variant.drop(columns=[target]))
            labels.extend(y_pred)
            probas.extend(y_proba)

    features = pd.concat(frames, ignore_index=True)
    if list(features.columns) != list(meta["features"]):
        raise ValueError(
            "Replay data columns {} do not match the recorded features {}.".format(
                list(features.columns), list(meta["features"])
            )
        )
    return features, labels, probas


def _as_array(values, dtype_name):
    """Restore predicted labels with the dtype the model returned (int, bool, str, ...)."""
    try:
        return np.asarray(values, dtype=np.dtype(dtype_name))
    except (TypeError, ValueError):
        return np.asarray(values, dtype=object)


def _self_check(model, features, labels, n_checks=6):
    """Look up a few stored rows, so a lookup problem fails here with a clear message instead of
    deep inside a RAI component."""
    positions = sorted(set(np.linspace(0, len(features) - 1, num=min(n_checks, len(features))).astype(int)))
    for position in positions:
        try:
            found = model.predict(features.iloc[[position]])
        except Exception as error:
            raise RuntimeError(
                "Prediction replay could not look up stored row {}: {}. See 'Prediction replay' "
                "in the template README for the data the lookup supports.".format(position, error)
            ) from error
        if found.tolist() != [labels[position]]:
            raise RuntimeError(
                "Prediction replay returned {} for stored row {}, expected {}.".format(
                    found.tolist(), position, labels[position]
                )
            )


def _load_pyfunc(path, model_config=None):
    """MLflow loader_module entry point: return the replay model built from the files at path."""
    with open(os.path.join(path, META_FILE), "r") as file:
        meta = json.load(file)
    if meta.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            "Unsupported replay format {} (this loader reads version {}).".format(
                meta.get("format_version"), FORMAT_VERSION
            )
        )
    with open(os.path.join(path, PREDICTIONS_FILE), "r") as file:
        predictions = json.load(file)

    features, labels, probas = build_lookup_table(path, meta, predictions)
    y_pred = _as_array(labels, meta["label_dtype"])
    if meta["task_type"] == "classification":
        from ml_wrappers.model.predictions_wrapper import PredictionsModelWrapperClassification

        model = PredictionsModelWrapperClassification(features, y_pred, np.asarray(probas, dtype=float))
    elif meta["task_type"] == "regression":
        from ml_wrappers.model.predictions_wrapper import PredictionsModelWrapperRegression

        model = PredictionsModelWrapperRegression(features, y_pred)
    else:
        raise ValueError("Unsupported replay task type '{}'.".format(meta["task_type"]))

    _self_check(model, features, y_pred.tolist())

    try:
        import ml_wrappers

        ml_wrappers_version = getattr(ml_wrappers, "__version__", "unknown")
    except Exception:
        ml_wrappers_version = "unknown"
    print(
        "Prediction replay model loaded: {} stored rows ({} train, {} test as written), source model {}, "
        "predictions computed with Python {}; ml-wrappers {}, pandas {}.".format(
            len(features),
            meta["n_rows"]["train"],
            meta["n_rows"]["test"],
            meta.get("source_model"),
            meta.get("created_with", {}).get("python"),
            ml_wrappers_version,
            pd.__version__,
        )
    )
    return model
