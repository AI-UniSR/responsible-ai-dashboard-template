"""
Runs main() of prepare_model_and_data_for_rai in every mode (survival or classification, with or
without extra features), with replay_predictions off and on.

The model registry is faked; no Azure is needed. The tests check two things:
  - with the flag off (the default), the component behaves exactly as before replay existed;
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
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

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


def test_replay_with_extras_hands_name_based_models_a_dataframe(tmp_path, monkeypatch):
    # In replay mode the classification selector outputs a DataFrame, so a model that selects its
    # columns by name works (the default mode keeps its numpy selector unchanged).
    result = _run(tmp_path, monkeypatch, "names", False, True, "True", model_factory=_name_based_classifier)
    (entry,) = result["replayed"]
    for split, key in (("train", "train_out"), ("test", "test_out")):
        rows = pd.read_csv(os.path.join(result["out"][key], "data.csv"))
        assert entry["predictions"][split]["y_pred"] == result["model"].predict(rows[MODEL_FEATURES]).tolist()


def test_replay_flag_rejects_garbage(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _run(tmp_path, monkeypatch, "bad", True, False, "maybe")
