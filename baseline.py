import numpy as np

from features import FEATURE_COLUMNS


class PreviousRevenueBaseline:
    """Forecast the next 30 days as the revenue of the previous 30 days."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        values = np.asarray(X, dtype=float)
        column = FEATURE_COLUMNS.index("previous_30")
        return values[:, column]
