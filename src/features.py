"""Feature engineering on 6-month behavioural history. Pure pandas, deterministic, no target leakage."""
from __future__ import annotations
import numpy as np
import pandas as pd

PAY = [f"PAY_{i}" for i in range(1, 7)]
BILL = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT = [f"PAY_AMT{i}" for i in range(1, 7)]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["LIMIT_BAL"] = df["LIMIT_BAL"]
    X["AGE"] = df["AGE"]
    X["EDUCATION"] = df["EDUCATION"].clip(upper=4)  # 0,5,6 are undocumented codes -> 'other'
    for c in PAY + BILL + PAYAMT:
        X[c] = df[c]

    pay = df[PAY].to_numpy()
    bill = df[BILL].to_numpy()
    paid = df[PAYAMT].to_numpy()
    limit = df["LIMIT_BAL"].to_numpy()[:, None]

    # delinquency behaviour
    late = (pay > 0)
    X["n_late_months"] = late.sum(1)
    X["max_delay"] = pay.max(1)
    X["mean_delay"] = pay.mean(1)
    X["recent_delay_3m"] = pay[:, :3].mean(1)
    X["delay_trend"] = pay[:, :3].mean(1) - pay[:, 3:].mean(1)        # >0: getting worse
    X["n_revolving"] = (pay == 0).sum(1)
    X["n_paid_in_full"] = (pay == -1).sum(1)
    X["longest_late_streak"] = [_longest_run(r) for r in late]

    # utilisation and repayment ratios
    util = bill / np.maximum(limit, 1)
    X["util_mean"] = util.mean(1)
    X["util_last"] = util[:, 0]
    X["util_max"] = util.max(1)
    X["util_trend"] = util[:, :3].mean(1) - util[:, 3:].mean(1)
    X["over_limit_months"] = (util > 1).sum(1)
    pr = paid / np.maximum(np.abs(bill), 1)
    X["pay_ratio_mean"] = np.clip(pr, 0, 5).mean(1)
    X["pay_ratio_last"] = np.clip(pr[:, 0], 0, 5)
    X["paid_total_to_bill_total"] = paid.sum(1) / np.maximum(np.abs(bill).sum(1), 1)
    X["bill_volatility"] = bill.std(1) / (np.abs(bill).mean(1) + 1)
    X["zero_payment_months"] = (paid == 0).sum(1)
    return X


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best
