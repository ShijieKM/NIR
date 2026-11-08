"""Stateless, per-spectrum preprocessing for near-infrared regression."""

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array


class SpectralTransform(TransformerMixin, BaseEstimator):
    def __init__(self, method="raw"):
        self.method = method

    def fit(self, x, y=None):
        x = check_array(x)
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, x):
        x = check_array(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("Spectrum length differs from training data")
        if self.method not in ("raw", "snv", "derivative", "snv_derivative"):
            raise ValueError(f"Unknown preprocessing: {self.method}")
        if self.method in ("snv", "snv_derivative"):
            mean = x.mean(axis=1, keepdims=True)
            std = x.std(axis=1, keepdims=True)
            if np.any(std <= np.finfo(float).eps):
                raise ValueError("SNV cannot normalize a constant spectrum")
            x = (x - mean) / std
        if self.method in ("derivative", "snv_derivative"):
            x = savgol_filter(x, window_length=15, polyorder=2, deriv=1, axis=1)
        return x
