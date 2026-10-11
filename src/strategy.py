"""Decision strategy layer: from PD to expected-NPV-maximising credit decisions.

Economic model (explicit assumptions, all configurable in `Economics`):
  * exposure at default EAD_i = CCF * limit_i * m           (m = 1 approve the existing limit, m = 0 decline)
  * revenue if the client does not default  R_i = apr * max(util_i, util_floor) * limit_i * m
  * loss if the client defaults             L_i = LGD * EAD_i
  * expected NPV(m)                         = (1 - PD_i) * R_i - PD_i * L_i      (m = 0 -> decline, NPV = 0)
Because expected NPV is linear in m, the optimum is binary: approve iff (1-PD)*R > PD*L, i.e. a client-specific
break-even PD that depends on utilisation (low-usage clients bring little revenue but carry full limit risk).
Limit-size optimisation would need data on how PD reacts to the limit (causal / experimental data), which this dataset lacks.
Realised profit is back-tested on the hold-out with true outcomes.

Limitations (documented in README): the data only contains clients the bank already approved, so there is
selection bias (no reject inference); PD does not depend on the limit.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Economics:
    apr: float = 0.22          # net yield on average drawn balance over the horizon
    lgd: float = 0.60
    ccf: float = 0.75          # share of the limit assumed drawn at default
    util_floor: float = 0.10
    multipliers: tuple = (0.0, 1.0)   # decline / approve full limit


def _components(df: pd.DataFrame, util: np.ndarray, eco: Economics, m: float):
    limit = df["LIMIT_BAL"].to_numpy(float)
    rev = eco.apr * np.maximum(util, eco.util_floor) * limit * m
    loss = eco.lgd * eco.ccf * limit * m
    return rev, loss


def expected_npv(df, util, pd_hat, eco: Economics, m: float) -> np.ndarray:
    rev, loss = _components(df, util, eco, m)
    return (1 - pd_hat) * rev - pd_hat * loss


def optimal_decision(df, util, pd_hat, eco: Economics) -> tuple[np.ndarray, np.ndarray]:
    """Returns chosen multiplier per client (0 = decline, 1 = approve) and the corresponding expected NPV."""
    npvs = np.stack([expected_npv(df, util, pd_hat, eco, m) for m in eco.multipliers], axis=1)
    idx = npvs.argmax(1)
    return np.array(eco.multipliers)[idx], npvs[np.arange(len(df)), idx]


def realised_profit(df, util, y, mult, eco: Economics) -> float:
    """Back-test: profit using the true outcome y (1 = default)."""
    total = 0.0
    for m in set(mult):
        if m == 0:
            continue
        sel = mult == m
        rev, loss = _components(df[sel], util[sel], eco, m)
        total += float(np.where(y[sel] == 1, -loss, rev).sum())
    return total


def bootstrap_profit(df, util, y, mult, eco, n=300, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(df), len(df))
        vals.append(realised_profit(df.iloc[i], util[i], y[i], mult[i], eco))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def approve_lowest_pd_share(score: np.ndarray, rate: float) -> np.ndarray:
    """Plain score cut-off baseline: approve exactly the `rate` share with the lowest score, full limit."""
    m = np.zeros(len(score)); m[np.argsort(score, kind="stable")[: int(round(rate * len(score)))]] = 1.0
    return m


def ecl(df, pd_hat, eco: Economics) -> float:
    """12-month expected credit loss of an approved book (IFRS 9 stage-1 style): sum PD * LGD * EAD."""
    return float((pd_hat * eco.lgd * eco.ccf * df["LIMIT_BAL"].to_numpy(float)).sum())
