# risk_wrapper.py
import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin


class SkSurvRiskWrapper(BaseEstimator, ClassifierMixin):
    """Turns any estimator exposing predict() (risk score, any scale) into a binary classifier.

    p = expit(s - cutoff), s = model.predict(X), cutoff = threshold; equals 0.5 at s == cutoff.
    This is a rank-preserving score transformation, not a calibrated probability: predicted
    labels and rankings are those of s itself. Valid domain is the whole real line; s and
    threshold must additionally be strictly positive when log_transform=True, since the score
    is log-transformed (together with the cutoff) before the sigmoid. Non-finite scores are
    rejected. Beyond |s - cutoff| ~= 745 the sigmoid saturates to 0/1 in floating point, which
    cannot create ties that weren't already there.
    """

    def __init__(self, model, threshold=1.0, decision_threshold=0.5, log_transform=False):
        self.model = model
        self.threshold = threshold
        self.decision_threshold = decision_threshold
        self.log_transform = log_transform

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        s = np.asarray(self.model.predict(X), dtype=float)
        if not np.all(np.isfinite(s)):
            raise ValueError("Risk score must be finite; got non-finite value(s).")

        cutoff = self.threshold
        if self.log_transform:
            if np.any(s <= 0) or not (cutoff > 0):
                raise ValueError(
                    "log_transform=True requires strictly positive scores and threshold."
                )
            s = np.log(s)
            cutoff = np.log(cutoff)

        p = expit(s - cutoff)
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.decision_threshold).astype(int)


def prepare_binary_classification_dataset(df, time_column, event_column, time_horizon, binary_label_name):
    """
    Approssima la survival analysis come classificazione binaria.

    A patient is kept if an event was observed, or if follow-up extended beyond time_horizon;
    patients censored before time_horizon are excluded. Kept patients are labelled 1 if the
    event occurred at or before time_horizon, 0 otherwise.

    Args:
        df (pd.DataFrame): Dataframe originale con le feature, tte, evento.
        time_column (str): Nome della colonna contenente il time-to-event.
        event_column (str): Nome della colonna evento (True se evento osservato).
        time_horizon_days (float): Tempo di cutoff per la trasformazione binaria.

    Returns:
        pd.DataFrame: Nuovo dataframe con riga per ogni paziente etichettata con 0/1.
    """
    n_in = len(df)
    keep_mask = (df[event_column]) | (df[time_column] > time_horizon)
    filtered_df = df.loc[keep_mask].copy()

    filtered_df[binary_label_name] = (
        (filtered_df[event_column]) & (filtered_df[time_column] <= time_horizon)
    ).astype(int)
    filtered_df = filtered_df.drop(columns=[time_column, event_column])

    n_excluded = n_in - len(filtered_df)
    print(
        f"[prepare_binary_classification_dataset] rows in: {n_in}, "
        f"excluded as censored before horizon: {n_excluded}, kept: {len(filtered_df)}"
    )

    return filtered_df
