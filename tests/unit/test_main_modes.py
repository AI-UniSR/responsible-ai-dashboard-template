"""
Runs main() of prepare_model_and_data_for_rai in every mode (survival or classification, with or
without extra features), with replay_predictions off and on.

The model registry is faked; no Azure is needed. The tests check two things:
  - with the flag off (the default), the component behaves exactly as before replay existed, for
    the models used here. The deliberate changes, in classification mode with extras: a model that
    cannot predict from the selector's numpy array (one that selects its columns by name) now gets a
    DataFrame instead; models with numpy.str_ feature names (XGBoost, LightGBM, on their own or in
    a wrapper) no longer fail, unless they were fitted on categories the array would misread
    (LightGBM categories other than 0..k-1, any XGBoost categorical feature), which are refused;
  - with the flag on, only the registered model changes: the RAI datasets are byte-identical and
    the replay model holds the model's own predictions.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mltable")
pytest.importorskip("azure.identity")

import mlflow  # noqa: E402
import yaml  # noqa: E402
from sklearn.base import BaseEstimator, ClassifierMixin  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402
from sklearn.pipeline import Pipeline, make_pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder  # noqa: E402

import main  # noqa: E402
import replay  # noqa: E402
from risk_wrapper import SkSurvRiskWrapper  # noqa: E402

MODEL_FEATURES = ["age", "bmi", "sex", "tumor"]
EXTRA = ["site"]


class FakeSurvivalModel:
    """Stands in for a fitted scikit-survival model: predict() returns a risk score."""

    def __init__(self, feature_names):
        self.feature_names_in_ = np.array(feature_names)

    def predict(self, X):
        X = X[list(self.feature_names_in_)]
        return np.asarray(X, dtype=float).sum(axis=1) / 40.0


def _frame(n, seed):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.uniform(20, 90, n).round(1),
        "bmi": rng.normal(25, 4, n).round(1),
        "sex": rng.integers(0, 2, n),
        "tumor": rng.integers(0, 3, n),
        "site": rng.choice(["north", "south", "centre"], n),
        "tte": rng.uniform(1, 36, n).round(2),
        "event": rng.random(n) < 0.6,
    })


def _write_csv_folder(df, folder):
    os.makedirs(folder, exist_ok=True)
    df.to_csv(os.path.join(folder, "data.csv"), index=False)
    return folder


def _classification_model(train):
    label = (train["tte"] <= 12) & train["event"]
    return LogisticRegression(max_iter=500).fit(train[MODEL_FEATURES], label.astype(int))


def _name_based_classifier(train):
    """A pipeline that selects its columns by name, as most real preprocessing does."""
    pre = ColumnTransformer([("num", "passthrough", ["age", "bmi"]),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), ["sex", "tumor"])])
    label = ((train["tte"] <= 12) & train["event"]).astype(int)
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=500))]).fit(train[MODEL_FEATURES], label)


def _xgboost_classifier(train):
    xgboost = pytest.importorskip("xgboost")
    label = ((train["tte"] <= 12) & train["event"]).astype(int)
    return xgboost.XGBClassifier(n_estimators=20, max_depth=3).fit(train[MODEL_FEATURES], label)


def _xgboost_on_categories(train, wrap=lambda model: model):
    xgboost = pytest.importorskip("xgboost")
    label = ((train["tte"] <= 12) & train["event"]).astype(int)
    model = wrap(xgboost.XGBClassifier(n_estimators=20, max_depth=3, enable_categorical=True, tree_method="hist"))
    return model.fit(train[MODEL_FEATURES].astype({"tumor": "category"}), label)


# Wrappers that hand the booster their input unchanged and take over its numpy.str_ feature names
BOOSTER_WRAPPERS = [
    pytest.param(lambda model: model, id="bare"),
    pytest.param(make_pipeline, id="pipeline"),
    pytest.param(lambda model: CalibratedClassifierCV(model, cv=2), id="calibrated"),
    pytest.param(lambda model: GridSearchCV(model, {"max_depth": [2, 3]}, cv=2), id="search"),
]


class _CannotPredict(ClassifierMixin, BaseEstimator):
    """A fitted model whose every prediction raises (e.g. a broken custom model). It is a scikit-learn
    estimator because recent Pipelines (scikit-learn 1.9.1 in the modern env) check their final step
    through the estimator API before predicting."""

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X):
        raise RuntimeError("this model cannot predict")


def _main_branch_selector():
    """The selector main registered before this branch (classification mode with extras)."""
    return ColumnTransformer([("keep", "passthrough", MODEL_FEATURES)], remainder="drop")


def _dice_frames(rows):
    """Hand-built frames shaped like the ones dice-ml 0.11 hands the model for counterfactuals (dice-ml
    itself only runs in tests/rai_env). Method 'random' makes every column object and turns the
    categorical values it samples into strings (here all of them); 'kdtree' passes the categoricals as
    category of str. 0/1 columns read from CSV give '0'/'1' here, not the 'True'/'False' that dice-ml
    produces from the boolean columns the RAI components read."""
    random, kdtree = rows.astype(object), rows.copy()
    for column in ("sex", "tumor", "site"):
        random[column] = random[column].astype(str)
        kdtree[column] = kdtree[column].astype(str).astype("category")
    return {"random": random, "kdtree": kdtree}


def _run(tmp_path, monkeypatch, run_name, survival, extras, replay_flag, model_factory=None):
    train, test = _frame(80, 1), _frame(40, 2)
    if not survival:
        # classification mode reads an existing binary label
        for df in (train, test):
            df["label"] = ((df["tte"] <= 12) & df["event"]).astype(int)
    if model_factory is not None:
        model = model_factory(train)
    else:
        model = FakeSurvivalModel(MODEL_FEATURES) if survival else _classification_model(train)
    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda uri: model)

    registered, replayed = [], []

    def fake_register_model(mlflow_model, model_name, input_example=None, code_paths=None):
        registered.append({"model": mlflow_model, "name": model_name, "input_example": input_example,
                           "code_paths": code_paths})
        return model_name, "7"

    def fake_register_replay_model(model_dir, model_name):
        with open(os.path.join(model_dir, "MLmodel")) as f:
            mlmodel = yaml.safe_load(f)
        data_dir = os.path.join(model_dir, mlmodel["flavors"]["python_function"]["data"])
        with open(os.path.join(data_dir, "replay_meta.json")) as f:
            meta = json.load(f)
        with open(os.path.join(data_dir, "predictions.json")) as f:
            predictions = json.load(f)
        replayed.append({"name": model_name, "mlmodel": mlmodel, "meta": meta, "predictions": predictions,
                         "code": sorted(os.listdir(os.path.join(model_dir, "code")))})
        return model_name, "3"

    monkeypatch.setattr(main, "register_model", fake_register_model)
    monkeypatch.setattr(replay, "register_replay_model", fake_register_replay_model)

    base = tmp_path / run_name
    out = {name: str(base / name) for name in ("train_out", "test_out", "info")}
    argv = [
        "main.py", "--model_name", "m", "--model_version", "2", "--binary_label_name", "label",
        "--train_data", _write_csv_folder(train, str(base / "train_in")),
        "--test_data", _write_csv_folder(test, str(base / "test_in")),
        "--train_data_out", out["train_out"], "--test_data_out", out["test_out"],
        "--model_info_path", out["info"],
    ]
    if survival:
        argv += ["--time_column", "tte", "--event_column", "event", "--time_horizon", "12"]
    if extras:
        argv += ["--extra_features", json.dumps(EXTRA)]
    if replay_flag is not None:
        argv += ["--replay_predictions", replay_flag]
    monkeypatch.setattr(sys, "argv", argv)
    main.main()

    with open(os.path.join(out["info"], "model_info.json")) as f:
        model_info = json.load(f)
    return {"registered": registered, "replayed": replayed, "model_info": model_info, "out": out, "model": model}


MODES = [
    pytest.param(True, False, id="survival"),
    pytest.param(True, True, id="survival+extras"),
    pytest.param(False, False, id="classification"),
    pytest.param(False, True, id="classification+extras"),
]


@pytest.mark.parametrize("survival,extras", MODES)
@pytest.mark.parametrize("flag", [None, "False", "false"])
def test_default_behaviour_is_unchanged(tmp_path, monkeypatch, survival, extras, flag):
    result = _run(tmp_path, monkeypatch, "run", survival, extras, flag)
    features = MODEL_FEATURES + (EXTRA if extras else [])
    assert result["replayed"] == []

    if survival:
        (call,) = result["registered"]
        assert call["name"] == "rai-m-clf"
        assert call["code_paths"] == ["./risk_wrapper.py"]
        assert list(call["input_example"].columns) == features and len(call["input_example"]) == 5
        assert isinstance(call["model"], Pipeline if extras else SkSurvRiskWrapper)
        assert result["model_info"] == {"id": "rai-m-clf:7"}
    elif extras:
        (call,) = result["registered"]
        assert call["name"] == "m" and call["code_paths"] is None
        assert list(call["input_example"].columns) == features
        assert isinstance(call["model"], Pipeline)
        assert result["model_info"] == {"id": "m:7"}
        # main's selector, which hands the model a numpy array
        selector = call["model"].named_steps["selector"]
        assert selector.get_params(deep=False) == _main_branch_selector().get_params(deep=False)
        assert isinstance(selector.transform(call["input_example"]), np.ndarray)
    else:
        assert result["registered"] == []
        assert result["model_info"] == {"id": "m:2"}

    written = pd.read_csv(os.path.join(result["out"]["train_out"], "data.csv"))
    assert list(written.columns) == features + ["label"]


@pytest.mark.parametrize("survival,extras", MODES)
def test_replay_changes_only_the_registered_model(tmp_path, monkeypatch, survival, extras):
    default = _run(tmp_path, monkeypatch, "default", survival, extras, None)
    replayed = _run(tmp_path, monkeypatch, "replay", survival, extras, "True")
    features = MODEL_FEATURES + (EXTRA if extras else [])

    # Nothing else is registered, and model_info.json points at the replay model.
    assert replayed["registered"] == []
    (entry,) = replayed["replayed"]
    assert entry["name"] == "rai-m-replay"
    assert replayed["model_info"] == {"id": "rai-m-replay:3"}

    # The datasets the RAI components read are byte-identical to the default run's.
    for key in ("train_out", "test_out"):
        for name in ("data.csv", "MLTable"):
            with open(os.path.join(default["out"][key], name), "rb") as a, \
                    open(os.path.join(replayed["out"][key], name), "rb") as b:
                assert a.read() == b.read(), (key, name)

    # The replay model is a loader_module pyfunc whose loader ships in code/.
    pyfunc = entry["mlmodel"]["flavors"]["python_function"]
    assert pyfunc["loader_module"] == "rai_prediction_replay_loader"
    assert entry["code"] == ["rai_prediction_replay_loader.py"]
    meta = entry["meta"]
    assert meta["features"] == features and meta["target_column"] == "label"
    assert meta["task_type"] == "classification" and meta["source_model"] == "m:2"

    # Stored predictions are exactly the RAI-facing model's outputs on the rows written for RAI.
    for split, key in (("train", "train_out"), ("test", "test_out")):
        rows = pd.read_csv(os.path.join(default["out"][key], "data.csv"))
        reference = default["registered"][0]["model"] if default["registered"] else default["model"]
        expected_proba = np.asarray(reference.predict_proba(rows[features]), dtype=float)
        expected_pred = np.asarray(reference.predict(rows[features]))
        stored = entry["predictions"][split]
        assert np.array_equal(np.asarray(stored["y_pred"]), expected_pred)
        assert np.allclose(np.asarray(stored["y_proba"]), expected_proba, rtol=0, atol=1e-12)


@pytest.mark.parametrize("flag", [None, "True"], ids=["default", "replay"])
@pytest.mark.parametrize("factory", [_classification_model, _xgboost_classifier], ids=["sklearn", "xgboost"])
def test_extras_selector_keeps_the_numpy_array_for_models_that_accept_it(tmp_path, monkeypatch, factory, flag):
    # A model that predicts from the numpy array keeps main's selector, so it keeps accepting frames
    # shaped like dice-ml's counterfactual frames (XGBoost and LightGBM reject them as a DataFrame).
    result = _run(tmp_path, monkeypatch, "numpy", False, True, flag, model_factory=factory)
    features = MODEL_FEATURES + EXTRA
    rows = pd.read_csv(os.path.join(result["out"]["test_out"], "data.csv"))[features]
    expected = result["model"].predict_proba(rows[MODEL_FEATURES])
    if flag is None:
        (call,) = result["registered"]
        for name, frame in _dice_frames(rows).items():
            assert np.allclose(call["model"].predict_proba(frame), expected, rtol=0, atol=1e-6), name
        assert isinstance(call["model"].named_steps["selector"].transform(rows), np.ndarray)
    else:
        # replay stores the same pipeline's predictions
        (entry,) = result["replayed"]
        assert entry["meta"]["features"] == features
        assert np.allclose(np.asarray(entry["predictions"]["test"]["y_proba"]), expected, rtol=0, atol=1e-6)


@pytest.mark.parametrize("wrap", BOOSTER_WRAPPERS)
@pytest.mark.parametrize("flag", [None, "True"], ids=["default", "replay"])
def test_extras_selector_refuses_xgboost_fitted_on_categories(tmp_path, monkeypatch, flag, wrap):
    # Through the numpy array XGBoost reads raw values as category codes, which cannot be checked. Its
    # numpy.str_ names failed before, so it is refused, on its own or inside a wrapper.
    with pytest.raises(ValueError, match="XGBoost model with categorical features"):
        _run(tmp_path, monkeypatch, "xgb_categories", False, True, flag,
             model_factory=lambda train: _xgboost_on_categories(train, wrap))


def test_extras_selector_only_warns_for_models_that_ran_before(tmp_path, monkeypatch, capsys):
    # Behind a transformer the names are plain str, so the model ran before this branch: it keeps
    # main's selector and gets a warning, not a refusal.
    behind_a_transformer = lambda model: make_pipeline(FunctionTransformer(), model)  # noqa: E731
    result = _run(tmp_path, monkeypatch, "warn", False, True, None,
                  model_factory=lambda train: _xgboost_on_categories(train, behind_a_transformer))
    (call,) = result["registered"]
    assert call["model"].named_steps["selector"].get_params(deep=False) == _main_branch_selector().get_params(deep=False)
    assert "Warning: XGBoost model with categorical features" in capsys.readouterr().out


def test_classification_without_extras_warns_about_category_codes(tmp_path, monkeypatch, capsys):
    # Without extra features nothing is registered, as before; the component only warns.
    result = _run(tmp_path, monkeypatch, "no_extras", False, False, None, model_factory=_xgboost_on_categories)
    assert result["registered"] == []
    assert "Warning: XGBoost model with categorical features" in capsys.readouterr().out


def test_extras_selector_keeps_main_selector_when_both_fail(tmp_path, monkeypatch, capsys):
    # Neither selector works: main's selector is registered, as before, and both errors are printed.
    result = _run(tmp_path, monkeypatch, "fails", False, True, None,
                  model_factory=lambda train: _CannotPredict().fit(train[MODEL_FEATURES]))
    (call,) = result["registered"]
    assert call["model"].named_steps["selector"].get_params(deep=False) == _main_branch_selector().get_params(deep=False)
    out = capsys.readouterr().out
    assert "numpy array: RuntimeError: this model cannot predict" in out
    assert "DataFrame: RuntimeError: this model cannot predict" in out


@pytest.mark.parametrize("flag", [None, "True"], ids=["default", "replay"])
def test_extras_selector_hands_name_based_models_a_dataframe(tmp_path, monkeypatch, flag):
    # A model that cannot predict from the numpy array (it selects its columns by name) gets a
    # DataFrame with its own columns instead, with or without replay.
    result = _run(tmp_path, monkeypatch, "names", False, True, flag, model_factory=_name_based_classifier)
    features = MODEL_FEATURES + EXTRA
    for split, key in (("train", "train_out"), ("test", "test_out")):
        rows = pd.read_csv(os.path.join(result["out"][key], "data.csv"))
        expected = result["model"].predict(rows[MODEL_FEATURES])
        if flag is None:
            # what the RAI components call: the registered pipeline on the RAI dataset's columns
            (call,) = result["registered"]
            selected = call["model"].named_steps["selector"].transform(rows[features])
            assert isinstance(selected, pd.DataFrame) and list(selected.columns) == MODEL_FEATURES
            assert np.array_equal(call["model"].predict(rows[features]), expected)
            proba = call["model"].predict_proba(rows[features])
            assert np.allclose(proba, result["model"].predict_proba(rows[MODEL_FEATURES]), rtol=0, atol=0)
        else:
            (entry,) = result["replayed"]
            assert entry["predictions"][split]["y_pred"] == expected.tolist()


def test_replay_flag_rejects_garbage(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _run(tmp_path, monkeypatch, "bad", True, False, "maybe")
