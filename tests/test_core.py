import numpy as np, pandas as pd
from src import metrics, strategy, features
from src.strategy import Economics


def test_psi_identical_is_zero_and_shift_is_positive():
    rng = np.random.default_rng(0); a = rng.normal(size=5000)
    assert metrics.psi(a, a) < 1e-6
    assert metrics.psi(a, a + 1.0) > 0.25


def test_gini_perfect_and_random():
    y = np.array([0] * 50 + [1] * 50)
    assert abs(metrics.gini(y, y) - 1) < 1e-9
    assert abs(metrics.gini(y, np.random.default_rng(1).random(100))) < 0.4


def test_breakeven_decision():
    df = pd.DataFrame({"LIMIT_BAL": [100_000.0, 100_000.0]})
    util = np.array([0.8, 0.8]); eco = Economics()
    mult, npv = strategy.optimal_decision(df, util, np.array([0.01, 0.9]), eco)
    assert list(mult) == [1.0, 0.0] and npv[1] == 0.0


def test_features_no_nan_and_shapes():
    from src import data
    df = data.load().head(500)
    X = features.build_features(df)
    assert len(X) == 500 and not X.isna().any().any()
    assert "default" not in X.columns and "SEX" not in X.columns and "MARRIAGE" not in X.columns
