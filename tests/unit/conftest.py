import os
import sys

# Import risk_wrapper directly, without pulling in mlflow/azure at import time.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "components", "prepare_model_and_data_for_rai"),
)
