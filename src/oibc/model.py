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


class _MLP:
    """표준화 + 결측 채움 + sklearn MLPRegressor. predict·best_iteration_(= 에폭 수)은 _Wrapped와 같다."""

    def __init__(self, feats, mu, sd, m, best_iter):
        self.feats, self.mu, self.sd, self.m, self.best_iteration_ = feats, mu, sd, m, best_iter

    @staticmethod
    def prep_raw(X, feats):
        x = X[feats].copy()
        for c in feats:
            if c == "eff_days":
                x[c] = x[c].fillna(0.0)
            elif c.startswith("eff_"):
                x[c] = x[c].fillna(1.0)
        a = x.to_numpy(dtype="float64")
        assert not np.isnan(a).any(), "MLP 입력 NaN"
        return a

    def transform(self, X):
        from .config import MLP_CLIP
        return np.clip((self.prep_raw(X, self.feats) - self.mu) / self.sd, -MLP_CLIP, MLP_CLIP).astype("float32")

    def predict(self, X):
        return self.m.predict(self.transform(X))


def fit_mlp(Xtr, ytr, wtr, Xes=None, yes=None, wes=None, n_epochs=None, feats=FEATURES, seed=None, log=print):
    """조기종료: 에폭마다 Xes의 가중 L1, MLP_PATIENCE 에폭 개선 없으면 멈추고 최적 에폭 가중치로 되돌린다.
    최종 학습(Xes 없음)은 n_epochs(폴드 B 최적 에폭)만큼 같은 학습률 일정으로 돌린다."""
    import copy
    from sklearn.neural_network import MLPRegressor
    from .config import MLP_LR_DECAY, MLP_LR_STEP, MLP_MAX_EPOCHS, MLP_PARAMS, MLP_PATIENCE, SEED
    feats = [f for f in feats if f != "site_id"]
    raw = _MLP.prep_raw(Xtr, feats)
    mu, sd = raw.mean(axis=0), raw.std(axis=0)
    sd[sd < 1e-9] = 1.0
    del raw
    wrap = _MLP(feats, mu, sd, None, 0)
    xtr = wrap.transform(Xtr)
    ytr = np.asarray(ytr, dtype="float32")
    wtr = np.asarray(wtr, dtype="float64")
    wtr = (wtr / wtr.mean()).astype("float32")
    m = MLPRegressor(**MLP_PARAMS, random_state=SEED if seed is None else seed)
    xes = wrap.transform(Xes) if Xes is not None else None
    best, best_ep, best_m, since = np.inf, 0, None, 0
    # 학습률은 고정 계단 일정(MLP_LR_STEP 에폭마다 × MLP_LR_DECAY) → 최종 학습이 같은 일정을 n_epochs까지 재현
    n_max = MLP_MAX_EPOCHS if n_epochs is None else n_epochs
    for ep in range(1, n_max + 1):
        lr = MLP_PARAMS["learning_rate_init"] * MLP_LR_DECAY ** ((ep - 1) // MLP_LR_STEP)
        if ep > 1:
            m._optimizer.learning_rate = lr
        m.partial_fit(xtr, ytr, sample_weight=wtr)
        if xes is None:
            continue
        p = m.predict(xes)
        l1 = float(np.sum(wes * np.abs(p - yes)) / np.sum(wes))
        if l1 < best - 1e-6:
            best, best_ep, best_m, since = l1, ep, copy.deepcopy(m), 0
        else:
            since += 1
        log(f"    mlp ep {ep}: es L1 {l1:.5f} (best {best:.5f} @ {best_ep}, lr {lr:.1e})")
        if since >= MLP_PATIENCE:
            break
    if xes is not None:
        m = best_m
    wrap.m, wrap.best_iteration_ = m, (best_ep if xes is not None else n_max)
    return wrap


def fit_model(algo, Xtr, ytr, wtr, Xes=None, yes=None, wes=None, n_estimators=None, feats=FEATURES, seed=None):
    """algo: "lgb" | "xgb" | "cat" | "mlp".가중치는 평균 1로 정규화해 쓴다(XGB·Cat의 최소 잎 가중치가 표본 수 단위가 되게).

    조기종료는 검증 세트(Xes)가 있을 때만, 최종 학습은 n_estimators(폴드 B best_iter)로 고정.
    """
    from .config import CAT_PARAMS, XGB_PARAMS, SEED
    seed = SEED if seed is None else seed
    if algo == "lgb":
        p = dict(LGBM_PARAMS, seed=seed)
        return fit_lgbm(Xtr, ytr, wtr, Xes, yes, wes, n_estimators=n_estimators, params=p, feats=feats)
    if algo == "mlp":
        return fit_mlp(Xtr, ytr, wtr, Xes, yes, wes, n_epochs=n_estimators, feats=feats, seed=seed)
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
