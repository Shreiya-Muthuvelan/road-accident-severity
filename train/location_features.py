"""Build raw-coordinate and smoothed grid-cell target-encoding features.

Fit this encoder on the training slice only. Training rows get inner
out-of-fold encodings, so a row's own label is not used for its cell rate.
Later rows use statistics from the full training slice; unseen cells fall
back to that slice's overall target mean.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

LOC_RAW = ["latitude", "longitude"]
LOC_TE = ["te_cell_fine", "te_cell_coarse"]


class LocationEncoder:
    """Create smoothed target rates for fine and coarse coordinate grids."""

    def __init__(self, fine=0.05, coarse=0.25, smoothing=20, n_inner=5, seed=0):
        """Set grid sizes, smoothing strength, and inner-fold settings."""
        self.sizes = {"te_cell_fine": fine, "te_cell_coarse": coarse}
        self.m, self.n_inner, self.seed = smoothing, n_inner, seed
        self.stats_ = {}

    @staticmethod
    def _cell(X, size):
        """Turn each coordinate into a stable grid-cell key."""
        la = np.floor(X["latitude"].to_numpy() / size).astype(int)
        lo = np.floor(X["longitude"].to_numpy() / size).astype(int)
        return pd.Series(la.astype(str) + "_" + lo.astype(str), index=X.index)

    @staticmethod
    def _stats(cells, y):
        """Collect the positive-label count and row count for each cell."""
        return pd.DataFrame({"c": cells.to_numpy(), "y": y.to_numpy()}).groupby("c")["y"].agg(["sum", "count"])

    def _encode(self, cells, g, prior):
        """Shrink small-cell rates toward the training-set prior."""
        s = cells.map(g["sum"]).fillna(0).to_numpy()
        n = cells.map(g["count"]).fillna(0).to_numpy()
        return (s + self.m * prior) / (n + self.m)

    def fit_transform(self, X, y):
        """Add out-of-fold cell rates for training rows and save full-fit stats."""
        y = pd.Series(np.asarray(y), index=X.index)
        out = X.copy()
        for name, size in self.sizes.items():
            cells = self._cell(X, size)
            enc = np.zeros(len(X))
            kf = KFold(self.n_inner, shuffle=True, random_state=self.seed)
            # Each training row gets a rate built without its own fold's labels.
            for tr, va in kf.split(X):
                g = self._stats(cells.iloc[tr], y.iloc[tr])
                enc[va] = self._encode(cells.iloc[va], g, y.iloc[tr].mean())
            out[name] = enc
            self.stats_[name] = (self._stats(cells, y), y.mean())
        return out

    def transform(self, X):
        """Add cell rates using the statistics saved from the training slice."""
        out = X.copy()
        for name, size in self.sizes.items():
            g, prior = self.stats_[name]
            out[name] = self._encode(self._cell(X, size), g, prior)
        return out
