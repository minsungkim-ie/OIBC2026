"""LightGBM 학습/예측. 타깃 = 설비이용률(cf), 야간(elev<=0)은 0 고정."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import EARLY_STOP, FEATURES, LGBM_PARAMS, PRED_CF_CLIP
from .io import k_of


def sample_weight(d):
    """capacity_kw × k(ts)."""
    return (d["capacity_kw"].to_numpy(dtype="float64") * k_of(d["ts"])).astype("float32")


def fit_lgbm(Xtr, ytr, wtr, Xes=None, yes=None, wes=None, n_estimators=None, params=None, feats=FEATURES):
    p = dict(params or LGBM_PARAMS)
    if n_estimators is not None:
        p["n_estimators"] = n_estimators
    m = lgb.LGBMRegressor(**p)
    kw = dict(sample_weight=wtr)
    if "site_id" in feats:
        kw["categorical_feature"] = ["site_id"]
    if Xes is not None:
        kw.update(eval_set=[(Xes[feats], yes)], eval_sample_weight=[wes], eval_metric="l1",
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False), lgb.log_evaluation(200)])
    m.fit(Xtr[feats], ytr, **kw)
    return m


def predict_kwh(m, d, feats=FEATURES, clip=PRED_CF_CLIP):
    """cf_pred.clip(0, 1.1) × capacity_kw, 야간 0."""
    cf = np.zeros(len(d), dtype="float64")
    day = (d["elev"] > 0).to_numpy()
    if day.any():
        cf[day] = m.predict(d.loc[day, feats])
    cf = np.clip(cf, 0, clip)
    return cf * d["capacity_kw"].to_numpy(dtype="float64")


def importance_top(m, feats=FEATURES, n=20):
    imp = pd.DataFrame({"feature": feats, "gain": m.booster_.feature_importance("gain")})
    return imp.sort_values("gain", ascending=False).head(n).reset_index(drop=True)
