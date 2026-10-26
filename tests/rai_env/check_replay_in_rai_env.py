"""
Checks prediction replay inside the curated RAI environment (responsibleai-tabular).

Run it as an Azure ML command job (check-replay-in-rai-env-job.yml); the RAI environment has no
pytest. Steps:
  1. builds replay models with the component's own code (replay.py);
  2. loads them exactly like the RAI components do: mlflow.pyfunc.load_model(...)._model_impl;
  3. runs the RAIInsights steps of the pipeline: constructor checks, error analysis, explanations,
     save/load as in gather, a second save/load as in the score card and dashboard, get_data(),
     and error analysis again on the reloaded data.

Exits with status 1 if any check fails.
"""
import os
import pickle
import platform
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "components", "prepare_model_and_data_for_rai"))

import mlflow  # noqa: E402
import mltable  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

import replay  # noqa: E402
from risk_wrapper import SkSurvRiskWrapper  # noqa: E402
from utils import write_filtered_mltable  # noqa: E402

TARGET = "label"
CATEGORICAL = ["sex", "tumor", "site"]


def versions():
    import ml_wrappers
    import responsibleai
    import sklearn

    return {
        "python": platform.python_version(),
        "responsibleai": responsibleai.__version__,
        "ml-wrappers": getattr(ml_wrappers, "__version__", "?"),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "mlflow": mlflow.__version__,
        "mltable": getattr(mltable, "__version__", "?"),
    }


def synthetic(n, seed, nan_fraction):
    """Values that stress the replay: one-decimal floats (inexact after the JSON reload), long
    floats, NaN, integer-coded categoricals and an audit-only string column."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.uniform(18, 90, n).round(1),
        "bmi": rng.normal(25, 5, n).clip(15, 45).round(1),
        "ratio": rng.uniform(0, 1, n) / 7.0,
        "biomarker": rng.exponential(2, n).round(3),
        "sex": rng.integers(0, 2, n),
        "tumor": rng.integers(0, 3, n),
        "site": rng.choice(["north", "south", "centre"], n),
    })
    risk = 0.03 * df["age"] + 0.05 * df["bmi"] + 0.4 * df["tumor"] + 0.3 * df["biomarker"] + rng.normal(0, 0.5, n)
    df[TARGET] = (risk > np.median(risk)).astype(int)
    for column in ("age", "biomarker"):
        df.loc[rng.random(n) < nan_fraction, column] = np.nan
    return df


def classifier_with_extras(train, model_features):
    """A NaN-native classifier behind the template's audit-only selector (classification mode with extras)."""
    clf = HistGradientBoostingClassifier(max_iter=60, random_state=0).fit(train[model_features], train[TARGET])
    selector = ColumnTransformer([("keep", "passthrough", model_features)], remainder="drop",
                                 verbose_feature_names_out=False)
    selector.set_output(transform="pandas")
    selector.fit(train[model_features + ["site"]])
    return Pipeline([("selector", selector), ("model", clf)])


def survival_like(train, model_features):
    """SkSurvRiskWrapper on a NaN-native risk regressor (survival mode, cutoff at the 66th percentile)."""
    risk_model = HistGradientBoostingRegressor(max_iter=60, random_state=0).fit(
        train[model_features], train["age"].fillna(50) / 10 + train["tumor"])
    threshold = float(np.percentile(risk_model.predict(train[model_features]), 66))
    return SkSurvRiskWrapper(risk_model, threshold=threshold, decision_threshold=0.5)


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def run_scenario(name, model, train, test, features, workdir):
    from responsibleai import RAIInsights
    from responsibleai.feature_metadata import FeatureMetadata

    timings = {}
    started = time.time()
    model_dir = os.path.join(workdir, name, "replay_model")
    replay.build_replay_model(model, train, test, features, TARGET, model_dir, "check:" + name)
    # The datasets the RAI components read (the component's own train/test outputs).
    data = {}
    for split, df in (("train", train), ("test", test)):
        folder = write_filtered_mltable(df, features, TARGET, os.path.join(workdir, name, split + "_out"))
        data[split] = mltable.load(str(folder)).to_pandas_dataframe()
    timings["build"] = time.time() - started

    started = time.time()
    loaded = mlflow.pyfunc.load_model(model_dir)._model_impl  # what the RAI components do
    timings["mlflow load"] = time.time() - started
    check(type(loaded).__module__.startswith("ml_wrappers") and type(loaded).__name__ == "PredictionsModelWrapperClassification",
          "unexpected model class {}.{}".format(type(loaded).__module__, type(loaded).__name__))
    check(type(pickle.loads(pickle.dumps(loaded))) is type(loaded), "replay model does not pickle")

    expected = {split: np.asarray(model.predict(df[features])) for split, df in (("train", train), ("test", test))}
    for split in ("train", "test"):
        got = loaded.predict(data[split].drop(columns=[TARGET]))
        check(np.array_equal(got, expected[split]), "{}: replayed predictions differ from the model's".format(split))

    feature_metadata = FeatureMetadata()
    feature_metadata.categorical_features = [c for c in CATEGORICAL if c in features]
    constructor_args = dict(target_column=TARGET, task_type="classification", classes=[False, True],
                            feature_metadata=feature_metadata, maximum_rows_for_test=5000)

    started = time.time()
    # constructor component: forecasting_enabled=True for every task, built only as a check
    RAIInsights(model=loaded, train=data["train"], test=data["test"], forecasting_enabled=True, **constructor_args)
    rai = RAIInsights(model=loaded, train=data["train"], test=data["test"], forecasting_enabled=False, **constructor_args)
    timings["constructor"] = time.time() - started

    started = time.time()
    rai.error_analysis.add(max_depth=3, num_leaves=31, min_child_samples=20, filter_features=None)
    rai.explainer.add()
    rai.compute()
    timings["error analysis + explanation"] = time.time() - started

    # gather: save, load, save again; score card and dashboard: load what gather saved
    started = time.time()
    first, second = os.path.join(workdir, name, "saved_1"), os.path.join(workdir, name, "saved_2")
    rai.save(first)
    gathered = RAIInsights.load(first)
    gathered.save(second)
    gathered.get_data()
    dashboard = RAIInsights.load(second)
    dashboard.get_data()
    timings["save/load/get_data"] = time.time() - started

    for label, reloaded in (("after one reload", gathered), ("after two reloads", dashboard)):
        got = reloaded.model.predict(reloaded.test.drop(columns=[TARGET]))
        check(np.array_equal(got, expected["test"]), "test predictions differ " + label)
        proba = reloaded.model.predict_proba(reloaded.test.drop(columns=[TARGET]))
        check(np.allclose(proba, model.predict_proba(test[features]), rtol=0, atol=1e-12), "probabilities differ " + label)

    started = time.time()
    # the score card recomputes error analysis on the reloaded dashboard with its own metric
    # (create_score_card -> _rai_insight_data.get_error_analysis_data)
    ea = dashboard.error_analysis
    ea._analyzer._metric = "accuracy_score"
    for config in ea._ea_config_list:
        config.is_computed = False
    ea._ea_report_list = []
    ea.compute()
    check(len(ea.get()) >= 1, "no error analysis report after the score-card recompute")
    timings["error analysis after reload"] = time.time() - started

    check(len(rai.error_analysis.get()) >= 1, "no error analysis report")
    check(len(rai.explainer.get()) >= 1, "no explanation")
    n_nan = int(data["test"].isna().any(axis=1).sum())
    print("  {}: OK ({} train, {} test rows, {} test rows with NaN); timings (s): {}".format(
        name, len(train), len(test), n_nan, {k: round(v, 1) for k, v in timings.items()}))


def main():
    print("Environment:", versions())
    workdir = tempfile.mkdtemp(prefix="replay_check_")
    model_features = ["age", "bmi", "ratio", "biomarker", "sex", "tumor"]
    train, test = synthetic(400, 1, 0.1), synthetic(200, 2, 0.1)
    scenarios = [
        ("classification_extras_nan", classifier_with_extras(train, model_features), model_features + ["site"]),
        ("survival_wrapper_nan", survival_like(train, model_features), model_features),
    ]
    failures = 0
    for name, model, features in scenarios:
        try:
            run_scenario(name, model, train, test, features, workdir)
        except Exception:
            failures += 1
            print("  {}: FAILED".format(name))
            traceback.print_exc()
    shutil.rmtree(workdir, ignore_errors=True)
    if failures:
        print("{} scenario(s) failed".format(failures))
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
