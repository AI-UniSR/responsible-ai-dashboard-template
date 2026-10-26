"""
Unit tests for prediction replay (replay.py and rai_prediction_replay_loader.py). They run with
pytest alone, in the component's environment (no Azure, no ml-wrappers).

The full load inside the RAI environment, where ml-wrappers and responsibleai are installed, is
covered by tests/rai_env/check_replay_in_rai_env.py.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest
import yaml

import rai_prediction_replay_loader as loader
import replay
from utils import write_filtered_mltable

FEATURES = ["age", "ratio", "site"]


class ThresholdModel:
    """Deterministic classifier on 'age' with predict/predict_proba, no fitting needed."""

    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-(X["age"].to_numpy(dtype=float) - 50.0) / 10.0))
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _frame(n, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        # one-decimal values such as 0.3 do not survive the RAI JSON reload unchanged
        "age": rng.uniform(20, 90, n).round(1),
        "ratio": rng.uniform(0, 1, n) / 3.0,
        "site": rng.choice(["north", "south", "centre"], n),
    })
    df["label"] = (df["age"] > 50).astype(int)
    return df


def _build(tmp_path, train=None, test=None, model=None):
    train = _frame(30, 1) if train is None else train
    test = _frame(20, 2) if test is None else test
    model_dir = str(tmp_path / "replay_model")
    meta = replay.build_replay_model(model or ThresholdModel(), train, test, FEATURES, "label", model_dir, "m:1")
    with open(os.path.join(model_dir, "MLmodel")) as f:
        mlmodel = yaml.safe_load(f)
    data_dir = os.path.join(model_dir, mlmodel["flavors"]["python_function"]["data"])
    return train, test, model_dir, mlmodel, data_dir, meta


def test_replay_model_layout(tmp_path):
    _, _, model_dir, mlmodel, data_dir, meta = _build(tmp_path)
    pyfunc = mlmodel["flavors"]["python_function"]
    assert pyfunc["loader_module"] == replay.LOADER_MODULE
    assert os.listdir(os.path.join(model_dir, pyfunc["code"])) == [replay.LOADER_MODULE + ".py"]
    assert sorted(os.listdir(data_dir)) == ["predictions.json", "replay_meta.json", "test", "train"]
    assert meta["format_version"] == loader.FORMAT_VERSION
    assert meta["features"] == FEATURES and meta["target_column"] == "label"
    assert meta["n_rows"] == {"train": 30, "test": 20} and meta["n_classes"] == 2
    assert meta["label_dtype"] == "int64"
    with open(os.path.join(data_dir, "replay_meta.json")) as f:
        assert json.load(f) == meta
    # nothing for use_model_dependency to install into the RAI environment but MLflow's own pin
    with open(os.path.join(model_dir, "requirements.txt")) as f:
        requirements = [line.strip() for line in f if line.strip()]
    assert all(line.startswith("mlflow") for line in requirements), requirements


def test_replay_data_is_byte_identical_to_component_outputs(tmp_path):
    train, test, _, _, data_dir, _ = _build(tmp_path)
    for split, df in (("train", train), ("test", test)):
        expected = write_filtered_mltable(df, FEATURES, "label", str(tmp_path / ("out_" + split)))
        for name in ("data.csv", "MLTable"):
            with open(os.path.join(data_dir, split, name), "rb") as a, open(os.path.join(str(expected), name), "rb") as b:
                assert a.read() == b.read()


def test_stored_predictions_are_the_model_outputs(tmp_path):
    train, test, _, _, data_dir, _ = _build(tmp_path)
    with open(os.path.join(data_dir, "predictions.json")) as f:
        stored = json.load(f)
    model = ThresholdModel()
    for split, df in (("train", train), ("test", test)):
        assert stored[split]["y_pred"] == model.predict(df[FEATURES]).tolist()
        assert np.allclose(stored[split]["y_proba"], model.predict_proba(df[FEATURES]), rtol=0, atol=0)


def test_boolean_and_string_labels_keep_their_type(tmp_path):
    class BoolModel(ThresholdModel):
        def predict(self, X):
            return self.predict_proba(X)[:, 1] >= 0.5

    _, _, _, _, data_dir, meta = _build(tmp_path, model=BoolModel())
    assert meta["label_dtype"] == "bool"
    with open(os.path.join(data_dir, "predictions.json")) as f:
        assert all(isinstance(v, bool) for v in json.load(f)["test"]["y_pred"])
    assert loader._as_array([True, False], "bool").dtype == np.bool_
    assert loader._as_array(["a", "b"], "<U1").tolist() == ["a", "b"]
    assert loader._as_array([1, 2], "not-a-dtype").dtype == object


@pytest.mark.parametrize("mutate,message", [
    (lambda df: df.rename(columns={"ratio": "target"}), "reserved"),
    (lambda df: df.rename(columns={"ratio": "target_1"}), "reserved"),
    (lambda df: df.rename(columns={"ratio": "2.5"}), "looks like a number"),
    (lambda df: df.rename(columns={"ratio": "a`b"}), "backtick"),
    (lambda df: df.rename(columns={"ratio": "n#lesions"}), "pandas query"),
    (lambda df: df.rename(columns={"ratio": "eGFR (mL/min/1.73 m²)"}), "pandas query"),
    (lambda df: df.rename(columns={"ratio": "AFP≥400"}), "pandas query"),
    (lambda df: df.rename(columns={"ratio": "patient's ratio"}), "pandas query"),
    (lambda df: df.assign(site=df["site"].replace("north", "Crohn's")), "single quote"),
    (lambda df: df.assign(site=df["site"].replace("north", "a\\b")), "backslash"),
    (lambda df: df.assign(site=df["site"].replace("north", "first\nsecond")), "line breaks"),
    (lambda df: df.assign(ratio=pd.Timestamp("2020-01-01")), "dates"),
    (lambda df: df.rename(columns={"ratio": "created_at"}).assign(created_at=1.6e9), "dates"),
    (lambda df: df.rename(columns={"site": "date"}).assign(date="2020-01-01"), "dates"),
])
def test_validation_rejects_data_the_lookup_cannot_replay(mutate, message):
    train = mutate(_frame(10, 1))
    features = [c for c in train.columns if c != "label"]
    with pytest.raises(ValueError, match=message):
        replay.validate_replay_data(train, train, features)


def test_validation_needs_two_features():
    train = _frame(10, 1)
    with pytest.raises(ValueError, match="at least two"):
        replay.validate_replay_data(train, train, ["age"])


def test_validation_accepts_nan_and_regular_strings():
    train = _frame(10, 1)
    train.loc[0, "age"] = np.nan
    train.loc[1, "site"] = None
    replay.validate_replay_data(train, train, FEATURES)


def test_validation_accepts_names_with_ordinary_punctuation():
    train = _frame(10, 1).rename(columns={"age": "age (years)", "ratio": "hbv/hcv", "site": "any toxicity grade>2"})
    replay.validate_replay_data(train, train, ["age (years)", "hbv/hcv", "any toxicity grade>2"])
    # pandas.read_json converts only *_at, *_time, date, datetime, modified and timestamp* columns
    dated = _frame(10, 1).rename(columns={"site": "visit_date"}).assign(visit_date="2020-01-01")
    replay.validate_replay_data(dated, dated, ["age", "ratio", "visit_date"])


def test_check_as_rai_reads_it_catches_records_split_by_line_breaks(tmp_path):
    pytest.importorskip("mltable")
    train = _frame(10, 1)
    train.loc[3, "site"] = "first\nsecond"  # bypasses validation on purpose
    write_filtered_mltable(train, FEATURES, "label", str(tmp_path / "train"))
    write_filtered_mltable(train, FEATURES, "label", str(tmp_path / "test"))
    try:
        replay.check_data_as_rai_reads_it(str(tmp_path), {"train": 10, "test": 10}, FEATURES)
    except ValueError as error:
        assert "read back" in str(error) or "line breaks" in str(error)
    else:
        pytest.fail("a value with a line break was not caught")


def test_lookup_table_holds_written_rows_and_their_json_reloads(tmp_path):
    pytest.importorskip("mltable")
    train, test, _, _, data_dir, meta = _build(tmp_path)
    with open(os.path.join(data_dir, "predictions.json")) as f:
        predictions = json.load(f)
    features, labels, probas = loader.build_lookup_table(data_dir, meta, predictions)

    assert list(features.columns) == FEATURES
    assert len(features) == len(labels) == len(probas)
    # 1 to 3 copies of each split: as written, after one and after two JSON reloads
    assert len(train) + len(test) <= len(features) <= 3 * (len(train) + len(test))

    # the rows as RAI reads them come first, with their own predictions
    written = loader.read_rai_dataset(os.path.join(data_dir, "train")).drop(columns=["label"])
    pd.testing.assert_frame_equal(features.iloc[:len(train)].reset_index(drop=True), written)
    assert labels[:len(train)] == predictions["train"]["y_pred"]

    # every JSON reload of the test rows is present, paired with the same predictions
    once = loader.rai_json_round_trip(loader.read_rai_dataset(os.path.join(data_dir, "test")))
    twice = loader.rai_json_round_trip(once)
    for reloaded in (once, twice):
        rows = reloaded.drop(columns=["label"])
        for i in (0, len(rows) - 1):
            match = (features == rows.iloc[i]).all(axis=1) | (features.isna() & rows.iloc[i].isna()).all(axis=1)
            assert match.any()
            assert {labels[j] for j in np.flatnonzero(match.to_numpy())} == {predictions["test"]["y_pred"][i]}


def test_lookup_table_rejects_mismatched_predictions(tmp_path):
    pytest.importorskip("mltable")
    _, _, _, _, data_dir, meta = _build(tmp_path)
    with open(os.path.join(data_dir, "predictions.json")) as f:
        predictions = json.load(f)
    predictions["test"]["y_pred"] = predictions["test"]["y_pred"][:-1]
    with pytest.raises(ValueError, match="predictions"):
        loader.build_lookup_table(data_dir, meta, predictions)


def test_loader_rejects_unknown_format(tmp_path):
    _, _, _, _, data_dir, meta = _build(tmp_path)
    meta["format_version"] = 99
    with open(os.path.join(data_dir, "replay_meta.json"), "w") as f:
        json.dump(meta, f)
    with pytest.raises(ValueError, match="Unsupported replay format"):
        loader._load_pyfunc(data_dir)


def test_loader_module_needs_nothing_from_the_component():
    source = open(loader.__file__).read()
    for name in ("utils", "replay", "risk_wrapper", "main"):
        assert "import {}".format(name) not in source and "from {} ".format(name) not in source
