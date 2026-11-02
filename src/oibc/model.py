"""LightGBM 학습/예측. 타깃 = 설비이용률(cf), 야간(elev<=0)은 0 고정."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .config import EARLY_STOP, FEATURES, LGBM_PARAMS, PRED_CF_CLIP, PRED_SCALE, RF_PARAMS
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


def fit_rf(Xtr, ytr, wtr, params=None, feats=FEATURES):
    """RandomForest (MSE, 가중). 트리마다 max_samples 비율만 부트스트랩하므로 메모리가 작다."""
    m = RandomForestRegressor(**(params or RF_PARAMS))
    m.fit(Xtr[feats].to_numpy(dtype="float32"), np.asarray(ytr, dtype="float32"), sample_weight=wtr)
    return m


def predict_kwh(m, d, feats=FEATURES, clip=PRED_CF_CLIP, scale=PRED_SCALE):
    """cf_pred.clip(0, 1.1) × scale × capacity_kw, 야간 0. scale은 과대예측 보정 배율(v2)."""
    cf = np.zeros(len(d), dtype="float64")
    day = (d["elev"] > 0).to_numpy()
    if day.any():
        X = d.loc[day, feats]
        cf[day] = m.predict(X.to_numpy(dtype="float32") if isinstance(m, RandomForestRegressor) else X)
    cf = np.clip(cf, 0, clip)
    return cf * scale * d["capacity_kw"].to_numpy(dtype="float64")


def importance_top(m, feats=FEATURES, n=20):
    gain = m.feature_importances_ if isinstance(m, RandomForestRegressor) else m.booster_.feature_importance("gain")
    imp = pd.DataFrame({"feature": feats, "gain": gain})
    return imp.sort_values("gain", ascending=False).head(n).reset_index(drop=True)
