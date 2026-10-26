"""
Prediction replay runs a component in the model's own environment. An Azure ML pipeline node cannot
override a component's environment (the override is accepted but ignored), so the replay tests use
copies of the component files. Each copy differs only in `code` and `environment`, plus its own `name`,
so a registered copy can never replace the default component. This keeps each copy in sync with its
original.
"""
import os

import pytest
import yaml

TESTS = os.path.join(os.path.dirname(__file__), "..")
ROOT = os.path.join(TESTS, "..")

VARIANTS = [
    ("components/prepare_model_and_data_for_rai/component.yml",
     "tests/replay_assets/prepare_model_and_data_for_rai.modern.yml"),
    ("tests/prepare_e2e_survival_assets/component.yml",
     "tests/replay_assets/prepare_e2e_survival_assets.modern.yml"),
]


def _load(relative):
    with open(os.path.join(ROOT, relative)) as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize("original,variant", VARIANTS)
def test_variant_differs_only_in_name_code_and_environment(original, variant):
    a, b = _load(original), _load(variant)
    assert b["environment"] != a["environment"]
    assert b["name"] != a["name"]
    # `code` must point at the original component folder
    original_dir = os.path.normpath(os.path.join(ROOT, os.path.dirname(original), a["code"]))
    variant_dir = os.path.normpath(os.path.join(ROOT, os.path.dirname(variant), b["code"]))
    assert variant_dir == original_dir
    for key in ("name", "code", "environment"):
        a.pop(key)
        b.pop(key)
    assert a == b
