"""v14: 그룹 단위 직접 모델.

발전소별 예측을 더하는 대신 (그룹, ts) 한 행으로 그룹 설비이용률을 직접 예측한다. 채점이 그룹 합계의 |오차|라서
발전소별 L1 최적값(중앙값)의 합과 그룹 합계의 최적값이 다른 부분(흐린 날 과소, 맑은 날 과대)을 보완한다.

피처: 그룹 안 발전소 예보 피처의 설비용량 가중 평균, 구름·UV의 발전소 간 표준편차,
      같은 대상일 6~19시 그룹 평균 구름·UV·강수확률 프로필(같은 대상일 안에서만 → 규칙 4),
      10시·17시 라운드를 한 행에 함께(r10 열은 _r10 접미사).
타깃: 그룹 cf = 실적 합 ÷ 실적이 있는 발전소 설비용량 합. 손실: SMP·k·용량 가중 L1 (평가식과 같은 단위).
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import FEATURES, K_HIGH, K_HOURS, SEED
from .extra import EXTRA_FEATS

NUM = [f for f in FEATURES if f not in ("site_id", "lat", "lon", "wa_dist", "wb_dist", "capacity_kw",
                                       "hour", "month", "doy_sin", "doy_cos")]
SD_COLS = ["wa_cloud", "wb_cloud", "wb_uv_idx"]
PROF = ["wa_cloud", "wb_cloud", "wb_uv_idx", "wb_precip_prob"]
PROF_HOURS = range(6, 20)
META = ("g", "ts", "td", "y", "e_has", "cap_has", "capacity_kw", "frac")
GROUP_PARAMS = dict(objective="l1", learning_rate=0.03, num_leaves=31, min_child_samples=50, feature_fraction=0.5,
                    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, n_estimators=5000, seed=SEED,
                    deterministic=True, force_row_wise=True, verbose=-1)
GROUP_ES = 300


def group_frame(d, group_of, extra=False):
    """d: build_frame 결과(site_id 정수 코드, energy 있으면 keep 열도). group_of: 정수 site_id → "G1"/"G2".

    피처는 행이 있는 발전소 전부의 용량가중 평균, 타깃은 keep·실적 있는 발전소만.
    """
    num = NUM + (EXTRA_FEATS if extra else [])     # v16: extra=True면 확장 예보 피처(oibc.extra)도 용량가중 평균
    d = d[[c for c in num + ["site_id", "ts", "capacity_kw", "energy", "keep"] if c in d.columns]].copy()
    d["g"] = d["site_id"].map(group_of)
    assert d["g"].notna().all()
    has = d["energy"].notna() & d["keep"].astype(bool) if "energy" in d.columns else pd.Series(False, index=d.index)
    cap = d["capacity_kw"].astype("float64")
    tmp = pd.DataFrame({"g": d["g"], "ts": d["ts"], "capacity_kw": cap, "cap_has": cap * has,
                        "e_has": d["energy"].where(has, 0.0) if "energy" in d.columns else 0.0})
    for c in num:
        tmp["_" + c] = d[c].astype("float64") * cap
    G = tmp.groupby(["g", "ts"]).sum()
    for c in num:
        G[c] = G.pop("_" + c) / G["capacity_kw"]
    G = G.join(d.groupby(["g", "ts"])[SD_COLS].std().fillna(0.0).add_suffix("_sd")).reset_index()
    G["y"] = G["e_has"] / G["cap_has"].replace(0, np.nan)
    G["frac"] = G["cap_has"] / G["capacity_kw"]
    ts = G["ts"].dt
    G["hour"] = ts.hour
    doy = ts.dayofyear
    G["doy_sin"], G["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    G["gid"] = (G["g"] == "G2").astype(int)
    G["td"] = (G["ts"] - pd.Timedelta(hours=1)).dt.floor("D")
    for c in PROF:
        pv = G.pivot_table(index=["g", "td"], columns="hour", values=c).reindex(columns=list(PROF_HOURS))
        pv.columns = [f"P_{c}_{h}" for h in pv.columns]
        G = G.merge(pv.reset_index(), on=["g", "td"], how="left")
    return G


def joint(g17, g10):
    """두 라운드를 한 행으로 (17시 기준, 10시 피처는 _r10). 주간(그룹 평균 elev > 0) 행만."""
    drop = ["y", "e_has", "cap_has", "capacity_kw", "frac", "hour", "doy_sin", "doy_cos", "gid", "td"]
    G = g17.merge(g10.drop(columns=drop), on=["g", "ts"], suffixes=("", "_r10"))
    return G[G["elev"] > 0].reset_index(drop=True)


# v15: 그룹 모델 변형. target "csi" = 그룹 cf ÷ 그룹 평균 cosz의 청천 cf(학습 가중치에 청천값을 곱해 같은 cf 손실),
# params = GROUP_PARAMS 덮어쓰기, aug = 학습에 무작위 2분할 그룹 n개를 추가(평가는 실제 구성 그룹). 검증 EXPERIMENTS 11-12.
GROUP_VARIANTS = {
    "g": dict(),
    "g_csi": dict(target="csi"),
    "g_lr": dict(params=dict(learning_rate=0.015, feature_fraction=0.3)),
    "g_augcsi": dict(target="csi", aug=6),
    # v16 후보: 위 4개에 확장 예보 피처(그룹 용량가중 평균)를 더한 것
    "gx": dict(extra=True),
    "gx_csi": dict(target="csi", extra=True),
    "gx_lr": dict(params=dict(learning_rate=0.015, feature_fraction=0.3), extra=True),
    "gx_augcsi": dict(target="csi", aug=6, extra=True),
}
AUG_SEED = 42
CS_FLOOR = 0.05


EXTRA_COLS = set(EXTRA_FEATS) | {c + "_r10" for c in EXTRA_FEATS}


def group_feats(G, extra=None):
    """extra=False면 확장 예보 피처 열을 뺀다 (G에 있어도 기존 변형은 v14·v15와 같은 피처로 학습)."""
    return [c for c in G.columns if c not in META and c != "aug" and (extra or c not in EXTRA_COLS)]


def aug_partitions(n_sites, n):
    """무작위 2분할 n개 ({정수 site_id: "G1"/"G2"}). 시드 고정."""
    rng = np.random.default_rng(AUG_SEED)
    return [{i: ("G1" if l == 0 else "G2") for i, l in enumerate(rng.integers(0, 2, n_sites))} for _ in range(n)]


def _target(tr, w, target):
    y = tr["y"].to_numpy("float64")
    if target != "csi":
        return y, w
    from .cv import clear_sky_cf
    cs = np.maximum(clear_sky_cf(tr["cosz"]), CS_FLOOR)
    return np.clip(y / cs, 0, 3), w * cs


def group_weight(G, smp):
    """SMP(학습 구간) · k · 실적 발전소 용량."""
    k = np.where(G["ts"].dt.hour.isin(list(K_HOURS)), K_HIGH, 1)
    return (G["ts"].map(smp).to_numpy("float64") * k * G["cap_has"].to_numpy("float64"))


def train_rows(G, t0, t1):
    return G[(G["ts"] >= t0) & (G["ts"] <= t1) & (G["frac"] > 0.5) & G["y"].notna()].reset_index(drop=True)


def fit_group(tr, smp, es_days=60, n_estimators=None, target="cf", params=None, extra=False):
    """n_estimators 없으면 마지막 es_days일로 조기종료, 있으면 그 수로 전체 학습. 반환 모델에 target을 붙인다."""
    feats = group_feats(tr, extra)
    y, w = _target(tr, group_weight(tr, smp), target)
    p = dict(GROUP_PARAMS, **(params or {}))
    if n_estimators is not None:
        p["n_estimators"] = n_estimators
        m = lgb.LGBMRegressor(**p).fit(tr[feats], y, sample_weight=w)
    else:
        es = (tr["ts"] > tr["ts"].max() - pd.Timedelta(days=es_days)).to_numpy()
        m = lgb.LGBMRegressor(**p).fit(
            tr.loc[~es, feats], y[~es], sample_weight=w[~es], eval_set=[(tr.loc[es, feats], y[es])],
            eval_sample_weight=[w[es]], callbacks=[lgb.early_stopping(GROUP_ES, verbose=False)])
    m.group_target = target
    m.group_feats = feats
    return m


def predict_group_kwh(m, G):
    """그룹 cf 예측 × 그룹 설비용량(행이 있는 발전소 전부) → kWh. G는 joint() 결과(주간만)."""
    p = m.predict(G[getattr(m, "group_feats", None) or group_feats(G)])
    if getattr(m, "group_target", "cf") == "csi":
        from .cv import clear_sky_cf
        p = p * np.maximum(clear_sky_cf(G["cosz"]), CS_FLOOR)
    cf = np.clip(p, 0, 1.1)
    return pd.Series(cf * G["capacity_kw"].to_numpy("float64"), index=pd.MultiIndex.from_frame(G[["g", "ts"]]))
