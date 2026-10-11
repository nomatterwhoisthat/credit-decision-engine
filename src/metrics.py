"""Credit-risk model metrics: Gini, KS, PSI, calibration (ECE), Hosmer-Lemeshow."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, brier_score_loss


def gini(y, p) -> float:
    return 2 * roc_auc_score(y, p) - 1


def ks(y, p) -> float:
    y = np.asarray(y); p = np.asarray(p)
    return float(stats.ks_2samp(p[y == 1], p[y == 0]).statistic)


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error with equal-frequency bins."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["b"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    g = df.groupby("b").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((g["n"] * (g["y"] - g["p"]).abs()).sum() / g["n"].sum())


def hosmer_lemeshow(y, p, bins: int = 10) -> tuple[float, float]:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["b"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    g = df.groupby("b").agg(obs=("y", "sum"), exp=("p", "sum"), n=("y", "size"))
    hl = (((g["obs"] - g["exp"]) ** 2) / (g["exp"] * (1 - g["exp"] / g["n"]))).sum()
    return float(hl), float(1 - stats.chi2.cdf(hl, bins - 2))


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index; bins from quantiles of the expected (reference) sample."""
    expected = np.asarray(expected, float); actual = np.asarray(actual, float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e = np.clip(e, 1e-4, None); a = np.clip(a, 1e-4, None)
    return float(((a - e) * np.log(a / e)).sum())


def summary(y, p, name: str = "") -> dict:
    hl, hl_p = hosmer_lemeshow(y, p)
    return {"model": name, "auc": float(roc_auc_score(y, p)), "gini": gini(y, p), "ks": ks(y, p),
            "brier": float(brier_score_loss(y, p)), "ece": ece(y, p), "hl_stat": hl, "hl_pvalue": hl_p}
