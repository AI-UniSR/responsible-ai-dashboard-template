# risk_wrapper.py
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class SkSurvRiskWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model, threshold=1.0, decision_threshold=0.5):
        self.model = model
        self.threshold = threshold
        self.decision_threshold = decision_threshold

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        risk = self.model.predict(X)
        p = risk / (risk + self.threshold)
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.decision_threshold).astype(int)
    
# utils

def prepare_binary_classification_dataset(df, time_column, event_column, time_horizon, binary_label_name):
    """
    Approssima la survival analysis come classificazione binaria.

    Args:
        df (pd.DataFrame): Dataframe originale con le feature, tte, evento.
        time_column (str): Nome della colonna contenente il time-to-event.
        event_column (str): Nome della colonna evento (True se evento osservato).
        time_horizon_days (float): Tempo di cutoff per la trasformazione binaria.

    Returns:
        pd.DataFrame: Nuovo dataframe con riga per ogni paziente etichettata con 0/1.
    """
    df = df.copy()

    # Condizione per tenere solo pazienti "etichettabili"
    keep_mask = (df[event_column]) | (df[time_column] > time_horizon)
    filtered_df = df[keep_mask]

    # Costruzione della nuova etichetta binaria
    def get_label(row):
        if row[event_column] and row[time_column] <= time_horizon:
            return 1
        else:
            return 0

    filtered_df[binary_label_name] = filtered_df.apply(get_label, axis=1)
    # Drop original survival columns
    filtered_df = filtered_df.drop(columns=[time_column, event_column])

    return filtered_df