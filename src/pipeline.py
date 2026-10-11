"""Training pipeline: split -> scorecard / LightGBM / CatBoost (Optuna) -> calibration -> artifacts."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from . import data, features, metrics
from .woe_scorecard import Scorecard

SEED = 42
optuna.logging.set_verbosity(optuna.logging.WARNING)


def split():
    df = data.load()
    tr, tmp = train_test_split(df, test_size=0.4, stratify=df["default"], random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.5, stratify=tmp["default"], random_state=SEED)
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def fit_catboost(Xtr, ytr, Xva, yva, n_trials: int = 12):
    def objective(trial):
        params = dict(depth=trial.suggest_int("depth", 3, 7), learning_rate=trial.suggest_float("lr", 0.03, 0.15, log=True),
                      l2_leaf_reg=trial.suggest_float("l2", 1, 20, log=True))
        m = CatBoostClassifier(iterations=600, eval_metric="AUC", random_seed=SEED, verbose=0, thread_count=2,
                               early_stopping_rounds=50, **params)
        m.fit(Xtr, ytr, eval_set=(Xva, yva))
        return m.get_best_score()["validation"]["AUC"]
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials)
    p = study.best_params
    best = CatBoostClassifier(iterations=800, eval_metric="AUC", random_seed=SEED, verbose=0, thread_count=2,
                              early_stopping_rounds=60, depth=p["depth"], learning_rate=p["lr"], l2_leaf_reg=p["l2"])
    best.fit(Xtr, ytr, eval_set=(Xva, yva))
    return best, study.best_params


def fit_lgbm(Xtr, ytr, Xva, yva):
    m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, random_state=SEED, n_jobs=2, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc", callbacks=[lgb.early_stopping(60, verbose=False)])
    return m


def run(n_trials: int = 12):
    tr, va, te = split()
    Xtr, Xva, Xte = (features.build_features(d) for d in (tr, va, te))
    ytr, yva, yte = tr["default"].to_numpy(), va["default"].to_numpy(), te["default"].to_numpy()

    sc = Scorecard().fit(Xtr, ytr)
    cb, best_params = fit_catboost(Xtr, ytr, Xva, yva, n_trials)
    lg = fit_lgbm(Xtr, ytr, Xva, yva)

    raw = {"scorecard": (sc.predict_proba(Xva), sc.predict_proba(Xte)),
           "lightgbm": (lg.predict_proba(Xva)[:, 1], lg.predict_proba(Xte)[:, 1]),
           "catboost": (cb.predict_proba(Xva)[:, 1], cb.predict_proba(Xte)[:, 1])}
    cal, test_pd, rows = {}, {}, []
    for name, (pva, pte) in raw.items():
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(pva, yva)   # calibrate on validation only
        cal[name] = iso
        test_pd[name] = iso.predict(pte)
        rows.append(metrics.summary(yte, pte, f"{name} (raw)"))
        rows.append(metrics.summary(yte, test_pd[name], f"{name} (isotonic-calibrated)"))
    return dict(tr=tr, va=va, te=te, Xtr=Xtr, Xva=Xva, Xte=Xte, ytr=ytr, yva=yva, yte=yte, scorecard=sc, catboost=cb,
                lgbm=lg, cal=cal, raw=raw, test_pd=test_pd, metrics=pd.DataFrame(rows), best_params=best_params)
