"""
Checks prediction replay inside the curated RAI environment (responsibleai-tabular).

Run it as an Azure ML command job (check-replay-in-rai-env-job.yml); the RAI environment has no
pytest. Steps:
  1. builds replay models with the component's own code (replay.py);
  2. loads them exactly like the RAI components do: mlflow.pyfunc.load_model(...)._model_impl;
  3. runs the RAIInsights steps of the pipeline: constructor checks, error analysis, explanations,
     data balance, save/load as in gather, a second save/load as in the score card and dashboard,
     get_data(), and error analysis again on the reloaded data.

The last scenarios use the default mode (no replay): the pipeline it registers in classification
mode with extra features (utils.build_selector_pipeline), saved and loaded as the RAI components do,
then constructor, error analysis, explanations, data balance and two save/load cycles. Models:
  - one that selects its columns by name, so the selector must hand it a DataFrame;
  - LightGBM (plain, and fitted on pandas 'category' columns) and XGBoost if the environment has it:
    they must keep the original numpy selector and also run counterfactuals (dice-ml), whose
    object-dtype frames they reject as a DataFrame;
  - LightGBM fitted on 'category' columns whose categories are not 0..k-1, on its own, in a one-step
    Pipeline and inside CalibratedClassifierCV: the numpy array would score it wrongly, so
    build_selector_pipeline must refuse it.
Expected predictions come from the rows as the component writes them, before the RAI components
read them back with their own types.

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
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

import replay  # noqa: E402
import utils  # noqa: E402
from risk_wrapper import SkSurvRiskWrapper  # noqa: E402
from utils import write_filtered_mltable  # noqa: E402

TARGET = "label"
CATEGORICAL = ["sex", "tumor", "site"]


def versions():
    import ml_wrappers
    import responsibleai
    import sklearn

    found = {
        "python": platform.python_version(),
        "responsibleai": responsibleai.__version__,
        "ml-wrappers": getattr(ml_wrappers, "__version__", "?"),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "mlflow": mlflow.__version__,
        "mltable": getattr(mltable, "__version__", "?"),
    }
    for package in ("dice_ml", "lightgbm", "xgboost"):
        try:
            found[package] = getattr(__import__(package), "__version__", "?")
        except ImportError:
            found[package] = "not installed"
    return found


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
    # DataBalanceManager.compute only warns on failure, leaving get() empty
    check(bool(rai.data_balance.get()), "no data balance measures")

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


def name_based_classifier(train, model_features):
    """A NaN-native classifier behind preprocessing that selects its columns by name."""
    numeric = [c for c in model_features if c not in ("sex", "tumor")]
    pre = ColumnTransformer([("num", "passthrough", numeric),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), ["sex", "tumor"])])
    model = Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(max_iter=60, random_state=0))])
    return model.fit(train[model_features], train[TARGET])


def as_categories(X):
    return X.astype({c: "category" for c in ("sex", "tumor") if c in X.columns})


def lightgbm_classifier(train, model_features, categories=False):
    """LightGBM. With categories=True it is fitted as its documentation suggests, with the integer
    codes stored as pandas 'category' (codes 0..k-1, so the numpy array scores them as the DataFrame would)."""
    import lightgbm
    X = as_categories(train[model_features]) if categories else train[model_features]
    model = lightgbm.LGBMClassifier(n_estimators=40, random_state=0, verbose=-1)
    return model.fit(X, train[TARGET])


def xgboost_classifier(train, model_features):
    import xgboost
    model = xgboost.XGBClassifier(n_estimators=40, max_depth=3, random_state=0)
    return model.fit(train[model_features], train[TARGET])


def run_default_selector_scenario(name, raw_model, train, test, model_features, workdir,
                                  expect_dataframe, counterfactuals=False, model_input=None):
    """Default mode, classification with extras: the pipeline build_selector_pipeline registers,
    saved as an MLflow sklearn model, loaded and analysed as the RAI components do."""
    from responsibleai import RAIInsights
    from responsibleai.feature_metadata import FeatureMetadata

    features = model_features + ["site"]
    original_load = mlflow.sklearn.load_model
    mlflow.sklearn.load_model = lambda uri: raw_model  # build_selector_pipeline loads the registered model
    try:
        pipeline = utils.build_selector_pipeline("model", 1, train, TARGET)
    finally:
        mlflow.sklearn.load_model = original_load
    model_dir = os.path.join(workdir, name, "sklearn_model")
    # cloudpickle, as the component registers it (MLflow 2.22); MLflow >= 3.14 would default to skops
    mlflow.sklearn.save_model(pipeline, model_dir, serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)

    data = {}
    for split, df in (("train", train), ("test", test)):
        folder = write_filtered_mltable(df, features, TARGET, os.path.join(workdir, name, split + "_out"))
        data[split] = mltable.load(str(folder)).to_pandas_dataframe()

    # what the constructor does: SNAP/rai_component_utilities.py:434 and create_rai_insights.py:202-204
    loaded = mlflow.pyfunc.load_model(model_dir)._model_impl
    if loaded.__class__.__name__ == "_SklearnModelWrapper":
        loaded = loaded.sklearn_model
    X_test = data["test"].drop(columns=[TARGET])
    print("  {}: dtypes as the RAI components read them: {}".format(name, X_test.dtypes.astype(str).to_dict()))
    selected = loaded.named_steps["selector"].transform(X_test)
    check(isinstance(selected, pd.DataFrame) == expect_dataframe,
          "the selector hands the model a {}".format(type(selected).__name__))
    raw_input = test[model_features]  # as the component wrote them; RAI reads 0/1 columns back as bool
    if model_input is not None:
        raw_input = model_input(raw_input)
    expected = raw_model.predict(raw_input)
    check(np.array_equal(loaded.predict(X_test), expected),
          "loaded pipeline predictions differ from the raw model's")
    check(np.allclose(loaded.predict_proba(X_test), raw_model.predict_proba(raw_input), rtol=0, atol=1e-6),
          "loaded pipeline probabilities differ from the raw model's")

    feature_metadata = FeatureMetadata()
    feature_metadata.categorical_features = [c for c in CATEGORICAL if c in features]
    constructor_args = dict(target_column=TARGET, task_type="classification", classes=[False, True],
                            feature_metadata=feature_metadata, maximum_rows_for_test=5000)
    RAIInsights(model=loaded, train=data["train"], test=data["test"], forecasting_enabled=True, **constructor_args)
    rai = RAIInsights(model=loaded, train=data["train"], test=data["test"], forecasting_enabled=False, **constructor_args)
    rai.error_analysis.add(max_depth=3, num_leaves=31, min_child_samples=20, filter_features=None)
    rai.explainer.add()
    if counterfactuals:
        # the rai_tabular_counterfactual defaults: dice-ml method random, with feature importance
        rai.counterfactual.add(total_CFs=10, method="random", desired_class="opposite")
    rai.compute()
    check(bool(rai.data_balance.get()), "no data balance measures")
    first, second = os.path.join(workdir, name, "saved_1"), os.path.join(workdir, name, "saved_2")
    rai.save(first)
    gathered = RAIInsights.load(first)
    gathered.save(second)
    gathered.get_data()
    dashboard = RAIInsights.load(second)
    dashboard.get_data()
    check(np.array_equal(dashboard.model.predict(dashboard.test.drop(columns=[TARGET])), expected),
          "predictions after two reloads differ")
    check(len(rai.error_analysis.get()) >= 1 and len(rai.explainer.get()) >= 1, "missing insights")
    if counterfactuals:
        check(len(rai.counterfactual.get()) >= 1, "no counterfactuals")
    print("  {}: OK ({} train, {} test rows; selector output {}{})".format(
        name, len(train), len(test), type(selected).__name__, "; counterfactuals" if counterfactuals else ""))


def run_category_guard_scenario(name, wrap, train, test, model_features, workdir):
    """LightGBM fitted on 'category' columns whose categories are not 0..k-1 (tumor 1..3), on its own or
    inside a wrapper that hands it the input unchanged. The numpy array would hand it raw values as
    codes, so build_selector_pipeline must refuse it. Without that refusal the scenario goes on to the
    prediction checks, which then show the wrong scores."""
    import lightgbm
    shifted_train, shifted_test = (df.assign(tumor=df["tumor"] + 1) for df in (train, test))
    raw_model = wrap(lightgbm.LGBMClassifier(n_estimators=40, random_state=0, verbose=-1))
    raw_model.fit(as_categories(shifted_train[model_features]), shifted_train[TARGET])
    try:
        run_default_selector_scenario(name, raw_model, shifted_train, shifted_test, model_features, workdir,
                                      expect_dataframe=False, model_input=as_categories)
    except ValueError as error:
        if "not 0..k-1" not in str(error):
            raise
        print("  {}: OK (refused: {})".format(name, str(error)[:110]))
        return
    raise AssertionError("a LightGBM model with categories other than 0..k-1 was accepted")


def main():
    print("Environment:", versions())
    workdir = tempfile.mkdtemp(prefix="replay_check_")
    model_features = ["age", "bmi", "ratio", "biomarker", "sex", "tumor"]
    train, test = synthetic(400, 1, 0.1), synthetic(200, 2, 0.1)
    scenarios = [
        ("classification_extras_nan", classifier_with_extras(train, model_features), model_features + ["site"]),
        ("survival_wrapper_nan", survival_like(train, model_features), model_features),
        ("classification_extras_by_name", name_based_classifier(train, model_features), model_features + ["site"]),
    ]
    failures = 0
    for name, model, features in scenarios:
        try:
            run_scenario(name, model, train, test, features, workdir)
        except Exception:
            failures += 1
            print("  {}: FAILED".format(name))
            traceback.print_exc()
    # Default mode. Counterfactuals refuse missing values, hence the NaN-free sets (24 test rows: dice-ml
    # feature importance needs at least 10). Their models leave out sex: the MLTable reads a 0/1
    # column as boolean, and dice-ml then hands any model 'True'/'False' strings, whatever the selector.
    cf_train, cf_test = synthetic(400, 3, 0.0), synthetic(24, 4, 0.0)
    cf_features = [c for c in model_features if c != "sex"]
    default_scenarios = [
        ("default_mode_extras_by_name", lambda: name_based_classifier(train, model_features), train, test,
         model_features, dict(expect_dataframe=True)),
        ("default_mode_extras_lightgbm", lambda: lightgbm_classifier(cf_train, cf_features),
         cf_train, cf_test, cf_features, dict(expect_dataframe=False, counterfactuals=True)),
        ("default_mode_extras_lightgbm_categories", lambda: lightgbm_classifier(cf_train, cf_features, True),
         cf_train, cf_test, cf_features, dict(expect_dataframe=False, counterfactuals=True, model_input=as_categories)),
        ("default_mode_extras_xgboost", lambda: xgboost_classifier(cf_train, cf_features),
         cf_train, cf_test, cf_features, dict(expect_dataframe=False, counterfactuals=True)),
    ]
    skipped = []
    for name, make_model, scenario_train, scenario_test, scenario_features, options in default_scenarios:
        try:
            raw_model = make_model()
        except ImportError as error:
            skipped.append(name)
            print("  {}: skipped ({})".format(name, error))
            continue
        try:
            run_default_selector_scenario(name, raw_model, scenario_train, scenario_test, scenario_features,
                                          workdir, **options)
        except Exception:
            failures += 1
            print("  {}: FAILED".format(name))
            traceback.print_exc()
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import make_pipeline
    wrappers = [("", lambda model: model), ("_pipeline", make_pipeline),
                ("_calibrated", lambda model: CalibratedClassifierCV(model, cv=2))]
    for suffix, wrap in wrappers:
        name = "default_mode_extras_lightgbm_categories_not_0_to_k" + suffix
        try:
            run_category_guard_scenario(name, wrap, cf_train, cf_test, cf_features, workdir)
        except Exception:
            failures += 1
            print("  {}: FAILED".format(name))
            traceback.print_exc()
    shutil.rmtree(workdir, ignore_errors=True)
    if failures:
        print("{} scenario(s) failed".format(failures))
        sys.exit(1)
    print("ALL CHECKS PASSED" + (" (skipped: {})".format(", ".join(skipped)) if skipped else ""))


if __name__ == "__main__":
    main()
