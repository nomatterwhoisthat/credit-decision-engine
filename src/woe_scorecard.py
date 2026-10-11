"""Classic bank-style WoE logistic-regression scorecard (the interpretable baseline regulators expect).

Steps: supervised binning (decision tree) -> WoE/IV -> IV + correlation based selection ->
statsmodels Logit (coefficients, p-values) -> points scaled with base score / PDO.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier


class WoEBinner:
    def __init__(self, max_bins: int = 6, min_leaf: float = 0.05):
        self.max_bins, self.min_leaf = max_bins, min_leaf
        self.edges_: dict[str, np.ndarray] = {}
        self.woe_: dict[str, np.ndarray] = {}
        self.iv_: dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y = np.asarray(y)
        for c in X.columns:
            x = X[c].to_numpy(float)
            if np.unique(x).size <= 2:
                edges = np.unique(x)[:-1] + 0.5 * np.diff(np.unique(x)) if np.unique(x).size == 2 else np.array([])
            else:
                t = DecisionTreeClassifier(max_leaf_nodes=self.max_bins, min_samples_leaf=self.min_leaf, random_state=0)
                t.fit(x.reshape(-1, 1), y)
                edges = np.sort(t.tree_.threshold[t.tree_.feature >= 0])
            self.edges_[c] = edges
            b = np.digitize(x, edges)
            n_b = np.bincount(b, minlength=len(edges) + 1)
            bad = np.bincount(b, weights=y, minlength=len(edges) + 1)
            good = n_b - bad
            dist_good = (good + 0.5) / (good.sum() + 0.5 * len(good))
            dist_bad = (bad + 0.5) / (bad.sum() + 0.5 * len(bad))
            w = np.log(dist_good / dist_bad)
            self.woe_[c] = w
            self.iv_[c] = float(((dist_good - dist_bad) * w).sum())
        return self

    def transform(self, X: pd.DataFrame, cols=None) -> pd.DataFrame:
        cols = cols or list(self.edges_)
        out = {c: self.woe_[c][np.digitize(X[c].to_numpy(float), self.edges_[c])] for c in cols}
        return pd.DataFrame(out, index=X.index)


class Scorecard:
    def __init__(self, iv_min: float = 0.02, corr_max: float = 0.8, base_score=600, base_odds=50, pdo=20):
        self.iv_min, self.corr_max = iv_min, corr_max
        self.base_score, self.base_odds, self.pdo = base_score, base_odds, pdo

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.binner = WoEBinner().fit(X, y)
        iv = pd.Series(self.binner.iv_).sort_values(ascending=False)
        cand = [c for c in iv.index if iv[c] >= self.iv_min]
        W = self.binner.transform(X, cand)
        keep: list[str] = []
        for c in cand:                                    # greedy: highest IV first, drop highly correlated
            if all(abs(W[c].corr(W[k])) < self.corr_max for k in keep):
                keep.append(c)
        # backward elimination: WoE = log(good/bad) so every coefficient must be negative; drop wrong-sign or p > 0.05
        while True:
            model = sm.Logit(np.asarray(y), sm.add_constant(W[keep])).fit(disp=0)
            bad = [c for c in keep if model.params[c] >= 0 or model.pvalues[c] > 0.05]
            if not bad:
                break
            worst = max(bad, key=lambda c: model.pvalues[c] if model.params[c] < 0 else 1.0)
            keep.remove(worst)
        self.features_ = keep
        self.iv_table_ = iv.rename("IV").to_frame()
        self.model_ = model
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        W = self.binner.transform(X, self.features_)
        # WoE here is log(good/bad) so the Logit models P(default) with negative coefficients ~ -1 -> fine
        return self.model_.predict(sm.add_constant(W, has_constant="add")).to_numpy()

    def points(self, X: pd.DataFrame) -> np.ndarray:
        p = np.clip(self.predict_proba(X), 1e-6, 1 - 1e-6)
        factor = self.pdo / np.log(2)
        offset = self.base_score - factor * np.log(self.base_odds)
        return offset + factor * np.log((1 - p) / p)

    def summary_table(self) -> pd.DataFrame:
        m = self.model_
        return pd.DataFrame({"coef": m.params, "std_err": m.bse, "p_value": m.pvalues})
