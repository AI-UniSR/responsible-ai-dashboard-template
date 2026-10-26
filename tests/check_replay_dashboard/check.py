"""
Last step of the prediction replay e2e test: checks the gathered RAI dashboard and the replay model.

Runs in the curated RAI environment (responsibleai-tabular). The pipeline fails if:
  - the dashboard's model is not the ml-wrappers replay object;
  - predictions on the reloaded test data differ from those RAI cached when it built the dashboard;
  - get_data(), error analysis or explanations are missing or fail;
  - the replay predictions were not computed with the expected Python (the model's environment);
  - the score card PDF is missing (when given).
"""
import argparse
import glob
import json
import os
import tempfile

import numpy as np


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("OK  " + message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard", type=str, required=True)
    parser.add_argument("--score_card", type=str, default=None)
    parser.add_argument("--expected_model_python", type=str, default=None)
    args = parser.parse_args()

    from responsibleai import RAIInsights

    rai = RAIInsights.load(args.dashboard)
    model = rai.model
    check(type(model).__module__.startswith("ml_wrappers") and type(model).__name__ == "PredictionsModelWrapperClassification",
          "dashboard model is the replay object ({}.{})".format(type(model).__module__, type(model).__name__))

    X = rai.test.drop(columns=[rai.target_column])
    n_nan = int(X.isna().any(axis=1).sum())
    print("test rows: {}, with missing values: {}".format(len(X), n_nan))
    check(np.array_equal(np.asarray(model.predict(X)), np.asarray(rai._predict_output)),
          "predictions on the reloaded test data equal the ones cached at construction")
    check(np.allclose(np.asarray(model.predict_proba(X)), np.asarray(rai._predict_proba_output), rtol=0, atol=1e-12),
          "probabilities on the reloaded test data equal the ones cached at construction")
    rai.get_data()
    check(True, "get_data() runs on the reloaded dashboard")
    check(len(rai.error_analysis.get()) >= 1, "error analysis report present")
    check(len(rai.explainer.get()) >= 1, "explanation present")

    # gather writes the constructor's rai_insights.json, with the model id, into the dashboard
    with open(os.path.join(args.dashboard, "rai_insights.json")) as f:
        name, version = json.load(f)["model_id"].rsplit(":", 1)
    import mlflow

    local = mlflow.artifacts.download_artifacts(artifact_uri="models:/{}/{}".format(name, version),
                                                dst_path=tempfile.mkdtemp())
    metas = glob.glob(os.path.join(local, "**", "replay_meta.json"), recursive=True)
    check(len(metas) == 1, "registered model {}:{} is a replay model".format(name, version))
    with open(metas[0]) as f:
        created_with = json.load(f)["created_with"]
    print("replay predictions computed with: {}".format(created_with))
    if args.expected_model_python:
        check(created_with["python"].startswith(args.expected_model_python),
              "replay predictions computed with Python {} (expected {})".format(created_with["python"], args.expected_model_python))

    if args.score_card:
        pdfs = glob.glob(os.path.join(args.score_card, "**", "*.pdf"), recursive=True)
        check(bool(pdfs) and os.path.getsize(pdfs[0]) > 0, "score card PDF written ({})".format(pdfs[:1]))

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
