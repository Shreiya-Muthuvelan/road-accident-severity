"""Leak-safe location features: raw lat/long + smoothed target encoding of grid cells.

Fit ONLY on the training slice. Training rows get inner out-of-fold encodings
(so a row never sees its own label); validation/test rows use stats from the
whole training slice. Unseen cells fall back to the training prior.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

LOC_RAW = ["latitude", "longitude"]
LOC_TE = ["te_cell_fine", "te_cell_coarse"]


class LocationEncoder:
    def __init__(self, fine=0.05, coarse=0.25, smoothing=20, n_inner=5, seed=0):
        self.sizes = {"te_cell_fine": fine, "te_cell_coarse": coarse}
        self.m, self.n_inner, self.seed = smoothing, n_inner, seed
        self.stats_ = {}

    @staticmethod
    def _cell(X, size):
        la = np.floor(X["latitude"].to_numpy() / size).astype(int)
        lo = np.floor(X["longitude"].to_numpy() / size).astype(int)
        return pd.Series(la.astype(str) + "_" + lo.astype(str), index=X.index)

    @staticmethod
    def _stats(cells, y):
        return pd.DataFrame({"c": cells.to_numpy(), "y": y.to_numpy()}).groupby("c")["y"].agg(["sum", "count"])

    def _encode(self, cells, g, prior):
        s = cells.map(g["sum"]).fillna(0).to_numpy()
        n = cells.map(g["count"]).fillna(0).to_numpy()
        return (s + self.m * prior) / (n + self.m)

    def fit_transform(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        out = X.copy()
        for name, size in self.sizes.items():
            cells = self._cell(X, size)
            enc = np.zeros(len(X))
            kf = KFold(self.n_inner, shuffle=True, random_state=self.seed)
            for tr, va in kf.split(X):
                g = self._stats(cells.iloc[tr], y.iloc[tr])
                enc[va] = self._encode(cells.iloc[va], g, y.iloc[tr].mean())
            out[name] = enc
            self.stats_[name] = (self._stats(cells, y), y.mean())
        return out

    def transform(self, X):
        out = X.copy()
        for name, size in self.sizes.items():
            g, prior = self.stats_[name]
            out[name] = self._encode(self._cell(X, size), g, prior)
        return out
