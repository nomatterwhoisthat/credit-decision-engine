"""Loading and renaming of the UCI/OpenML 'Default of credit card clients' dataset (Taiwan, 2005).

30 000 clients, 6 months of repayment status (PAY_*), bill amounts (BILL_AMT*) and payments (PAY_AMT*).
Target: default in the next month. LIMIT_BAL is used as exposure (EAD) proxy in the strategy layer.
"""
from __future__ import annotations
import warnings
from pathlib import Path
import pandas as pd

CACHE = Path(__file__).resolve().parents[1] / "data" / "taiwan_credit.csv"

COLS = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]          # 1 = most recent month (Sep), 6 = oldest (Apr)
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)

# Attributes excluded from every model (protected / not used for decisions). Used only for the fairness audit.
PROTECTED = ["SEX", "MARRIAGE"]


def load() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_csv(CACHE)
    from sklearn.datasets import fetch_openml
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = fetch_openml("default-of-credit-card-clients", version=1, as_frame=True, parser="auto")
    df = d.data.astype(float).copy()
    df.columns = COLS
    df["default"] = d.target.astype(int).values
    df.insert(0, "client_id", range(len(df)))
    CACHE.parent.mkdir(exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df
