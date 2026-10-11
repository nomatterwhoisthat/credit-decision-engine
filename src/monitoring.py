"""Model monitoring: feature/score PSI and anomaly detection (IsolationForest) on incoming applications."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from .metrics import psi


def psi_report(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    rows = [(c, psi(ref[c], cur[c])) for c in ref.columns]
    out = pd.DataFrame(rows, columns=["feature", "psi"]).sort_values("psi", ascending=False)
    out["status"] = pd.cut(out["psi"], [-1, 0.1, 0.25, np.inf], labels=["stable", "watch", "alert"])
    return out


def simulate_drift(X: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Clearly synthetic stress scenario used only to demonstrate the monitor:
    macro shock -> more delays, higher utilisation, lower repayments."""
    rng = np.random.default_rng(seed)
    Z = X.copy()
    for c in [f"PAY_{i}" for i in range(1, 4)]:
        Z[c] = Z[c] + (rng.random(len(Z)) < 0.25).astype(int)
    for c in [f"BILL_AMT{i}" for i in range(1, 4)]:
        Z[c] = Z[c] * 1.25
    for c in [f"PAY_AMT{i}" for i in range(1, 4)]:
        Z[c] = Z[c] * 0.7
    return Z


class AnomalyDetector:
    def __init__(self, contamination: float = 0.01, seed: int = 0):
        self.m = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed, n_jobs=2)

    def fit(self, X): self.m.fit(X); return self
    def score(self, X): return -self.m.score_samples(X)        # higher = more anomalous
    def flag(self, X): return self.m.predict(X) == -1
