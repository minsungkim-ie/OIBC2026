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


class _Wrapped:
    """XGBoost·CatBoost 모델을 LightGBM과 같은 인터페이스(predict, best_iteration_)로 감싼다."""

    def __init__(self, m, best_iter, feats):
        self.m, self.best_iteration_, self.feats = m, best_iter, feats

    def predict(self, X):
        return self.m.predict(X[self.feats])


def fit_model(algo, Xtr, ytr, wtr, Xes=None, yes=None, wes=None, n_estimators=None, feats=FEATURES, seed=None):
    """algo: "lgb" | "xgb" | "cat". 가중치는 평균 1로 정규화해 쓴다(XGB·Cat의 최소 잎 가중치가 표본 수 단위가 되게).

    조기종료는 검증 세트(Xes)가 있을 때만, 최종 학습은 n_estimators(폴드 B best_iter)로 고정.
    """
    from .config import CAT_PARAMS, XGB_PARAMS, SEED
    seed = SEED if seed is None else seed
    if algo == "lgb":
        p = dict(LGBM_PARAMS, seed=seed)
        return fit_lgbm(Xtr, ytr, wtr, Xes, yes, wes, n_estimators=n_estimators, params=p, feats=feats)
    s = float(np.mean(wtr))
    wtr = np.asarray(wtr) / s
    wes = None if wes is None else np.asarray(wes) / s
    if algo == "xgb":
        import xgboost as xgb
        p = dict(XGB_PARAMS, random_state=seed)
        if n_estimators is not None:
            p["n_estimators"] = n_estimators
        if Xes is not None:
            p["early_stopping_rounds"] = EARLY_STOP
        m = xgb.XGBRegressor(**p)
        kw = dict(sample_weight=wtr, verbose=200)
        if Xes is not None:
            kw.update(eval_set=[(Xes[feats], yes)], sample_weight_eval_set=[wes])
        m.fit(Xtr[feats], ytr, **kw)
        best = (m.best_iteration + 1) if Xes is not None else p["n_estimators"]
        return _Wrapped(m, best, feats)
    if algo == "cat":
        import catboost as cb
        p = dict(CAT_PARAMS, random_seed=seed)
        if n_estimators is not None:
            p["iterations"] = n_estimators
        m = cb.CatBoostRegressor(**p)
        kw = dict(sample_weight=wtr)
        if Xes is not None:
            kw.update(eval_set=cb.Pool(Xes[feats], yes, weight=wes), early_stopping_rounds=EARLY_STOP,
                      use_best_model=True)
        m.fit(Xtr[feats], ytr, **kw)
        best = (m.get_best_iteration() + 1) if Xes is not None else p["iterations"]
        return _Wrapped(m, best, feats)
    raise ValueError(algo)


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
