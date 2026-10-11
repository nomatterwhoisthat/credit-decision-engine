"""Reproduces every number and figure in README: `python run_all.py`  (about 3-5 minutes on 2 CPU cores)."""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_curve
from src import pipeline, strategy, monitoring, metrics, features
from src.strategy import Economics

warnings.filterwarnings("ignore")
OUT = Path("results"); OUT.mkdir(exist_ok=True)


def main():
    R = pipeline.run(n_trials=12)
    te, Xte, yte = R["te"], R["Xte"], R["yte"]
    R["metrics"].round(4).to_csv(OUT / "model_metrics.csv", index=False)
    print(R["metrics"].round(4).to_string(index=False))

    # scorecard artefacts
    sc = R["scorecard"]
    sc.iv_table_.round(4).to_csv(OUT / "iv_table.csv")
    sc.summary_table().round(4).to_csv(OUT / "scorecard_logit_summary.csv")

    # ROC
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for k, (_, pte) in R["raw"].items():
        f, t, _ = roc_curve(yte, pte); ax.plot(f, t, label=f"{k} (Gini {metrics.gini(yte, pte):.3f})")
    ax.plot([0, 1], [0, 1], "k:"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(); ax.set_title("ROC, hold-out")
    fig.tight_layout(); fig.savefig(OUT / "roc.png", dpi=130); plt.close(fig)

    # calibration plot
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, p in [("catboost raw", R["raw"]["catboost"][1]), ("catboost calibrated", R["test_pd"]["catboost"])]:
        d = pd.DataFrame({"y": yte, "p": p}); d["b"] = pd.qcut(d.p.rank(method="first"), 10, labels=False)
        g = d.groupby("b").mean(); ax.plot(g.p, g.y, "o-", label=name)
    ax.plot([0, 1], [0, 1], "k:"); ax.set_xlabel("mean predicted PD"); ax.set_ylabel("observed default rate"); ax.legend()
    ax.set_title("Calibration by decile"); fig.tight_layout(); fig.savefig(OUT / "calibration.png", dpi=130); plt.close(fig)

    # SHAP
    expl = shap.TreeExplainer(R["catboost"])
    sv = expl.shap_values(Xte.iloc[:2000])
    plt.figure(); shap.summary_plot(sv, Xte.iloc[:2000], show=False, max_display=15)
    plt.tight_layout(); plt.savefig(OUT / "shap_summary.png", dpi=130); plt.close()

    # ---- strategy back-test ----
    eco = Economics()
    util = (Xte["util_mean"].to_numpy())
    pdc = R["test_pd"]["catboost"]; pds = R["test_pd"]["scorecard"]
    full = np.ones(len(te))
    res = {}

    def evaluate(name, mult):
        lo, hi = strategy.bootstrap_profit(te, util, yte, mult, eco, n=200)
        res[name] = dict(realised_profit=strategy.realised_profit(te, util, yte, mult, eco), ci95=[lo, hi],
                         approval_rate=float((mult > 0).mean()), avg_limit_mult=float(mult.mean()),
                         default_rate_of_approved=float(yte[mult > 0].mean()) if (mult > 0).any() else None)
    evaluate("approve_all_full_limit", full)
    mult_cb, npv_cb = strategy.optimal_decision(te, util, pdc, eco)
    evaluate("npv_strategy_catboost", mult_cb)
    mult_sc, _ = strategy.optimal_decision(te, util, pds, eco)
    evaluate("npv_strategy_scorecard", mult_sc)
    rate = float((mult_cb > 0).mean())                     # same approval rate, but a plain PD cut-off (ignores revenue/exposure per client)
    evaluate("pd_cutoff_same_approval_rate", strategy.approve_lowest_pd_share(R["raw"]["catboost"][1], rate))
    oracle = np.where(yte == 1, 0.0, 1.0)
    evaluate("oracle_upper_bound", oracle)
    json.dump(res, open(OUT / "strategy_backtest.json", "w"), indent=2)
    print(json.dumps(res, indent=2))

    # sensitivity of the conclusion to the economic assumptions
    rows = []
    for lgd in (0.4, 0.6, 0.8):
        for apr in (0.15, 0.22, 0.30):
            e2 = Economics(apr=apr, lgd=lgd)
            m_npv, _ = strategy.optimal_decision(te, util, pdc, e2)
            m_cut = strategy.approve_lowest_pd_share(R["raw"]["catboost"][1], float((m_npv > 0).mean()))
            rows.append(dict(lgd=lgd, apr=apr, approval_rate=float((m_npv > 0).mean()),
                             profit_npv=strategy.realised_profit(te, util, yte, m_npv, e2) / 1e6,
                             profit_same_rate_cutoff=strategy.realised_profit(te, util, yte, m_cut, e2) / 1e6,
                             profit_approve_all=strategy.realised_profit(te, util, yte, full, e2) / 1e6))
    pd.DataFrame(rows).round(3).to_csv(OUT / "sensitivity.csv", index=False)
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    # cut-off sweep figure
    rates = np.linspace(0.3, 1.0, 36); prof = []
    order = np.argsort(pdc)
    for r in rates:
        m = np.zeros(len(te)); m[order[: int(r * len(te))]] = 1.0
        prof.append(strategy.realised_profit(te, util, yte, m, eco))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rates, np.array(prof) / 1e6, label="approve lowest-PD share, full limit")
    ax.axhline(res["npv_strategy_catboost"]["realised_profit"] / 1e6, color="g", ls="--", label="expected-NPV strategy (client-specific break-even PD)")
    ax.set_xlabel("approval rate"); ax.set_ylabel("realised profit, mln NT$ (hold-out)"); ax.legend(); ax.set_title("Approval rate vs profit")
    fig.tight_layout(); fig.savefig(OUT / "approval_vs_profit.png", dpi=130); plt.close(fig)

    # ECL
    eclv = {"ecl_total_ntd": strategy.ecl(te, pdc, eco), "ecl_approved_book_ntd": strategy.ecl(te[mult_cb > 0], pdc[mult_cb > 0], eco),
            "exposure_total_ntd": float(te["LIMIT_BAL"].sum())}
    json.dump(eclv, open(OUT / "ecl.json", "w"), indent=2)

    # ---- monitoring ----
    rep_same = monitoring.psi_report(R["Xtr"], Xte); rep_same.to_csv(OUT / "psi_train_vs_test.csv", index=False)
    Xdrift = monitoring.simulate_drift(Xte); Xdrift_f = features.build_features(R["te"].assign(**{c: Xdrift[c] for c in features.PAY + features.BILL + features.PAYAMT}))
    rep_drift = monitoring.psi_report(R["Xtr"], Xdrift_f); rep_drift.to_csv(OUT / "psi_simulated_shock.csv", index=False)
    pd_shock = R["cal"]["catboost"].predict(R["catboost"].predict_proba(Xdrift_f)[:, 1])
    mon = {"max_feature_psi_train_vs_test": float(rep_same.psi.max()), "n_alert_features_simulated_shock": int((rep_drift.status == "alert").sum()),
           "score_psi_simulated_shock": metrics.psi(R["raw"]["catboost"][1], R["catboost"].predict_proba(Xdrift_f)[:, 1]),
           "mean_pd_normal": float(pdc.mean()), "mean_pd_shock": float(pd_shock.mean())}
    ad = monitoring.AnomalyDetector().fit(R["Xtr"]); mon["anomaly_flag_rate_normal"] = float(ad.flag(Xte).mean())
    mon["anomaly_flag_rate_shock"] = float(ad.flag(Xdrift_f).mean())
    json.dump(mon, open(OUT / "monitoring.json", "w"), indent=2); print(mon)

    # ---- fairness audit (protected attributes were NOT used as features) ----
    aud = te[["SEX", "AGE"]].copy(); aud["approved"] = mult_cb > 0; aud["default"] = yte
    aud["age_band"] = pd.cut(aud.AGE, [0, 29, 39, 49, 100], labels=["<30", "30-39", "40-49", "50+"])
    fair = {"by_sex": aud.groupby("SEX").agg(approval_rate=("approved", "mean"), default_rate_if_all=("default", "mean")).round(4).to_dict("index"),
            "by_age": aud.groupby("age_band").agg(approval_rate=("approved", "mean"), default_rate_if_all=("default", "mean")).round(4).to_dict("index")}
    ar = aud.groupby("SEX").approved.mean(); fair["disparate_impact_ratio_sex"] = float(ar.min() / ar.max())
    json.dump(fair, open(OUT / "fairness.json", "w"), indent=2, default=str); print(fair)

    # artefacts for the agent project
    Path("artifacts").mkdir(exist_ok=True)
    R["catboost"].save_model("artifacts/catboost_pd.cbm")
    import joblib
    joblib.dump(R["cal"]["catboost"], "artifacts/isotonic_catboost.joblib")
    json.dump({"features": list(Xte.columns), "best_params": R["best_params"]}, open("artifacts/meta.json", "w"), indent=2)
    print("done")


if __name__ == "__main__":
    main()
