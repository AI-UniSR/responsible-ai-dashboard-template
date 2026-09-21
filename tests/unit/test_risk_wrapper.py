"""
Unit tests for SkSurvRiskWrapper (R-1, R-2, R-3). Runs with pytest alone: no Azure, no
registry, no cohort data. All scores below are synthetic values chosen to span the observed
Med-CLI range (0.179-7.452), not real patient data.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from risk_wrapper import SkSurvRiskWrapper

CUTOFF = 1.871752447313276
OBSERVED_RANGE = np.array([0.179, 0.5, 1.0, 1.871752447313276, 3.0, 7.452])


class FakeRiskModel:
    """Minimal estimator exposing predict() + feature_names_in_, like CoxSurvivalWrapper."""

    def __init__(self, feature_names):
        self.feature_names_in_ = np.array(feature_names)

    def predict(self, X):
        X = X[list(self.feature_names_in_)] if hasattr(X, "columns") else X
        return np.asarray(X).sum(axis=1).astype(float)


def test_old_vs_new_equivalence():
    # log_transform=True recovers the old r/(r+threshold) mapping exactly for r = predict().
    model = FakeRiskModel(["x"])
    wrapper = SkSurvRiskWrapper(model, threshold=CUTOFF, log_transform=True)
    X = pd.DataFrame({"x": OBSERVED_RANGE})

    p_new = wrapper.predict_proba(X)[:, 1]
    p_old = OBSERVED_RANGE / (OBSERVED_RANGE + CUTOFF)

    # Agree to floating-point precision, not bitwise (the letter's word is "identical").
    assert np.allclose(p_new, p_old, rtol=1e-12)
    assert np.array_equal(wrapper.predict(X), (p_old >= 0.5).astype(int))


def test_extreme_scores_stay_valid_and_warning_free():
    model = FakeRiskModel(["x"])
    wrapper = SkSurvRiskWrapper(model, threshold=0.0)
    scores = np.array([-1e6, -1.0, 0.0, 1.0, 1e6])
    X = pd.DataFrame({"x": scores})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = wrapper.predict_proba(X)[:, 1]

    assert np.all(np.isfinite(p))
    assert np.all((p >= 0.0) & (p <= 1.0))
    assert np.all(np.diff(p) >= 0)

    moderate = np.abs(scores) <= 30
    assert np.all((p[moderate] > 0.0) & (p[moderate] < 1.0))


def test_p_is_half_at_cutoff():
    model = FakeRiskModel(["x"])
    wrapper = SkSurvRiskWrapper(model, threshold=2.5)
    X = pd.DataFrame({"x": [2.5]})
    assert wrapper.predict_proba(X)[0, 1] == pytest.approx(0.5)


def test_rank_preservation():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=50)
    model = FakeRiskModel(["x"])
    wrapper = SkSurvRiskWrapper(model, threshold=0.3)
    X = pd.DataFrame({"x": scores})
    p = wrapper.predict_proba(X)[:, 1]
    assert np.array_equal(np.argsort(p), np.argsort(scores))


def test_rejects_non_finite_scores():
    model = FakeRiskModel(["x"])
    wrapper = SkSurvRiskWrapper(model, threshold=1.0)
    with pytest.raises(ValueError, match="finite"):
        wrapper.predict_proba(pd.DataFrame({"x": [np.nan]}))
    with pytest.raises(ValueError, match="finite"):
        wrapper.predict_proba(pd.DataFrame({"x": [np.inf]}))


def test_rejects_non_positive_scores_with_log_transform():
    model = FakeRiskModel(["x"])
    wrapper = SkSurvRiskWrapper(model, threshold=1.0, log_transform=True)
    with pytest.raises(ValueError, match="positive"):
        wrapper.predict_proba(pd.DataFrame({"x": [-1.0]}))


def test_classifier_contract():
    model = FakeRiskModel(["x"])
    wrapper = SkSurvRiskWrapper(model, threshold=1.0)
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    proba = wrapper.predict_proba(X)
    assert proba.shape == (3, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    # consistent with classes: [false, true] declared in survival-pipeline.yml
    assert set(np.unique(wrapper.predict(X))).issubset({0, 1})


def _build_survival_extra_features_pipeline():
    """Mirrors the fixed main.py survival + extra_features branch: selector (model features
    only) -> SkSurvRiskWrapper(raw model), with audit-only extras dropped before predict()."""
    model_features = ["age", "bmi"]
    extra_features = ["audit_only"]
    df_train = pd.DataFrame({
        "age": [50.0, 60.0, 70.0, 80.0],
        "bmi": [20.0, 22.0, 24.0, 26.0],
        "audit_only": [1, 2, 3, 4],
    })
    surv_model = FakeRiskModel(model_features)
    wrapped_model = SkSurvRiskWrapper(surv_model, threshold=100.0)

    selector = ColumnTransformer(
        [("keep", "passthrough", model_features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    selector.set_output(transform="pandas")
    selector.fit(df_train[model_features + extra_features])

    pipeline = Pipeline([("selector", selector), ("model", wrapped_model)])
    return pipeline, df_train, model_features, extra_features


def test_registered_object_exposes_valid_classifier_outputs(tmp_path):
    pipeline, df_train, model_features, extra_features = _build_survival_extra_features_pipeline()
    X = df_train[model_features + extra_features]

    model_path = tmp_path / "model"
    mlflow.sklearn.save_model(
        pipeline, str(model_path), serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
    )
    loaded = mlflow.sklearn.load_model(str(model_path))

    assert hasattr(loaded, "predict") and hasattr(loaded, "predict_proba")
    proba = loaded.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_audit_only_columns_do_not_alter_predictions():
    pipeline, df_train, model_features, extra_features = _build_survival_extra_features_pipeline()
    X = df_train[model_features + extra_features]

    X_perturbed = X.copy()
    X_perturbed["audit_only"] = X_perturbed["audit_only"].values[::-1]

    assert np.array_equal(pipeline.predict(X), pipeline.predict(X_perturbed))
    assert np.allclose(pipeline.predict_proba(X), pipeline.predict_proba(X_perturbed))


def test_wrapped_and_raw_rankings_are_identical():
    pipeline, df_train, model_features, extra_features = _build_survival_extra_features_pipeline()
    X = df_train[model_features + extra_features]

    raw_model = pipeline.named_steps["model"].model
    raw_scores = raw_model.predict(df_train[model_features])
    wrapped_proba = pipeline.predict_proba(X)[:, 1]

    assert np.array_equal(np.argsort(raw_scores), np.argsort(wrapped_proba))
