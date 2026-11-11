"""2폴드(연 단위 롤링) 검증 설정과 공통 함수.

폴드마다 "앞 2년 학습 → 다음 1년 검증". 평가 구간(1년)과 같은 모양을 두 해에 반복해,
두 해 모두에서 W가 줄어드는 변경만 채택한다.
"""
import numpy as np
import pandas as pd

from .config import FEATURES, ROOT, TZ, _ts

OUT_CV = ROOT / "outputs" / "v4"
CACHE = OUT_CV / "cache"

# 예보·피처를 한 번에 만들 전체 범위 (대상일 단위: 01:00 시작, 00:00 끝)
CV_START = _ts("2021-06-01 01:00")
CV_END = _ts("2025-06-01 00:00")

# 폴드: (학습 시작, 학습 끝, 검증 시작, 검증 끝). 조기종료는 학습 구간 마지막 ES_DAYS일.
FOLDS = {
    "A": (_ts("2021-06-01 01:00"), _ts("2023-06-01 00:00"), _ts("2023-06-01 01:00"), _ts("2024-06-01 00:00")),
    "B": (_ts("2022-06-01 01:00"), _ts("2024-06-01 00:00"), _ts("2024-06-01 01:00"), _ts("2025-06-01 00:00")),
}
ES_DAYS = 60

# v1 피처 = v2 피처 + site_id(범주형) − 요일/주말
V1_FEATURES = ["site_id"] + [f for f in FEATURES if f not in ("dow", "weekend")]

# 변형: target "cf" = 설비이용률, "csi" = 청천지수(설비이용률 ÷ 청천 기대 설비이용률)
# 발전소 효율 피처: site_id 대신 학습 구간에서 잰 "같은 시각 전체 발전소 중앙값 대비 설비이용률 비"를 시간대별로
EFF_FEATS = ["eff_am", "eff_mid", "eff_pm", "eff_days"]
EFF_WINDOW_DAYS = 365

# eff 값에 따라 효율 피처를 붙이는 방식이 다르다.
#   True    : (v7) 학습 구간 마지막 EFF_WINDOW_DAYS일로 한 번 재서 학습·검증 행 전부에 같은 값 → 학습 행은 자기 정답이 피처에 섞인다
#   "year"  : (v8 후보) 시점 기준. 6월 1일마다 그 직전 1년 실적으로 다시 잰다. 평가 연도(1년 내내 고정)와 같은 조건
#   "month" : (v8 후보) 시점 기준. 매달 1일마다 직전 1년 실적으로 다시 잰다
# 시점 기준은 블록 첫 대상일의 예측시점(전일 17:00)까지의 실적만 쓴다 (CLAUDE.md 규칙 2·6).
VARIANTS = {
    "v1": dict(feats=V1_FEATURES, target="cf"),
    "v2": dict(feats=FEATURES, target="cf"),
    "csi": dict(feats=FEATURES, target="csi"),
    "v2e": dict(feats=FEATURES + EFF_FEATS, target="cf", eff=True),
    "csie": dict(feats=FEATURES + EFF_FEATS, target="csi", eff=True),
    "v2y": dict(feats=FEATURES + EFF_FEATS, target="cf", eff="year"),
    "csiy": dict(feats=FEATURES + EFF_FEATS, target="csi", eff="year"),
    "v2m": dict(feats=FEATURES + EFF_FEATS, target="cf", eff="month"),
    "csim": dict(feats=FEATURES + EFF_FEATS, target="csi", eff="month"),
    # v9: 같은 피처·타깃을 다른 알고리즘(algo)·시드(seed)로 학습해 앙상블을 다양화. algo 없으면 LightGBM, seed 없으면 42
    "v2e_s1": dict(feats=FEATURES + EFF_FEATS, target="cf", eff=True, seed=1001),
    "csie_s1": dict(feats=FEATURES + EFF_FEATS, target="csi", eff=True, seed=1001),
    "v2e_x": dict(feats=FEATURES + EFF_FEATS, target="cf", eff=True, algo="xgb"),
    "csie_x": dict(feats=FEATURES + EFF_FEATS, target="csi", eff=True, algo="xgb"),
    "v2e_c": dict(feats=FEATURES + EFF_FEATS, target="cf", eff=True, algo="cat"),
    "csie_c": dict(feats=FEATURES + EFF_FEATS, target="csi", eff=True, algo="cat"),
    "v2_x": dict(feats=FEATURES, target="cf", algo="xgb"),
    "csi_x": dict(feats=FEATURES, target="csi", algo="xgb"),
    "v2_c": dict(feats=FEATURES, target="cf", algo="cat"),
    "csi_c": dict(feats=FEATURES, target="csi", algo="cat"),
    # v11: 신경망(sklearn MLP). site_id는 fit_mlp가 뺀다 (정수 코드를 연속값으로 넣지 않도록)
    "csie_m": dict(feats=FEATURES + EFF_FEATS, target="csi", eff=True, algo="mlp"),
    # v12: 배깅 계열(sklearn RandomForest / ExtraTrees, 제곱오차). site_id는 fit_forest가 뺀다
    "v2e_r": dict(feats=FEATURES + EFF_FEATS, target="cf", eff=True, algo="rf"),
    "csie_r": dict(feats=FEATURES + EFF_FEATS, target="csi", eff=True, algo="rf"),
    "v2e_t": dict(feats=FEATURES + EFF_FEATS, target="cf", eff=True, algo="et"),
    "csie_t": dict(feats=FEATURES + EFF_FEATS, target="csi", eff=True, algo="et"),
    # v12: CatBoost 다른 시드 (v9 이득을 만든 Cat 계열 강화)
    "v2e_c1": dict(feats=FEATURES + EFF_FEATS, target="cf", eff=True, algo="cat", seed=1001),
    "csie_c1": dict(feats=FEATURES + EFF_FEATS, target="csi", eff=True, algo="cat", seed=1001),
    # v12: 학습 기간(win). 기본은 폴드 학습 구간 2년. 1 = 마지막 1년만, "all" = CV_START부터 (A는 2년 그대로, B는 3년)
    "v2e_w1": dict(feats=FEATURES + EFF_FEATS, target="cf", eff=True, win=1),
    "csie_w1": dict(feats=FEATURES + EFF_FEATS, target="csi", eff=True, win=1),
    "v2e_wa": dict(feats=FEATURES + EFF_FEATS, target="cf", eff=True, win="all"),
    "csie_wa": dict(feats=FEATURES + EFF_FEATS, target="csi", eff=True, win="all"),
    "v2_wa": dict(feats=FEATURES, target="cf", win="all"),
    "csi_wa": dict(feats=FEATURES, target="csi", win="all"),
    "v2e_c_wa": dict(feats=FEATURES + EFF_FEATS, target="cf", eff=True, algo="cat", win="all"),
    "csie_c_wa": dict(feats=FEATURES + EFF_FEATS, target="csi", eff=True, algo="cat", win="all"),
    # v13 후보: decay = 학습 끝에서 1년 멀어질 때마다 표본 가중치 × decay (최근 연도 중시)
    "v2e_wad": dict(feats=FEATURES + EFF_FEATS, target="cf", eff=True, win="all", decay=0.7),
    "csie_wad": dict(feats=FEATURES + EFF_FEATS, target="csi", eff=True, win="all", decay=0.7),
    # v13 후보: lgbp = LightGBM 설정 덮어쓰기 (데이터가 늘어난 만큼 잎 수 확대)
    "v2e_wal": dict(feats=FEATURES + EFF_FEATS, target="cf", eff=True, win="all", lgbp=dict(num_leaves=127)),
    "csie_wal": dict(feats=FEATURES + EFF_FEATS, target="csi", eff=True, win="all", lgbp=dict(num_leaves=127)),
}


def decay_mult(name, ts, t1):
    """변형의 decay에 따른 표본 가중치 배수: decay ** (학습 끝 t1에서 떨어진 햇수, 0부터 정수). 없으면 1."""
    dec = VARIANTS[name].get("decay")
    if dec is None:
        return np.ones(len(ts))
    age = ((pd.Timestamp(t1) - pd.DatetimeIndex(ts)) / pd.Timedelta(days=365.25)).to_numpy().astype(int)
    return np.power(float(dec), np.maximum(age, 0))


def train_start(name, t0, t1):
    """변형의 학습 시작 시각. win 없으면 폴드 기본 t0."""
    win = VARIANTS[name].get("win")
    if win is None:
        return t0
    if win == "all":
        return CV_START
    return t1 - pd.DateOffset(years=int(win)) + pd.Timedelta(hours=1)
PIT_MODES = ("year", "month")
EFF_YEAR_START_MONTH = 6        # 연 단위 블록은 6월 1일 시작 (평가 구간 2025-06-01 ~ 와 같은 모양)
EFF_CUTOFF = pd.Timedelta(hours=7)   # 대상일 00:00 − 7h = 전일 17:00 (예측시점). ts ≤ 이 시각인 실적만 사용


def _eff_prep(d):
    """효율 계산 전처리: keep·주간 행만, 같은 시각 전 발전소 중앙값 대비 비율(ratio)과 시간대(band).

    ref(같은 시각 중앙값)는 그 시각의 행끼리만 계산하므로 창(window)을 나중에 잘라도 결과가 같다.
    """
    m = d["keep"].to_numpy(bool) & (d["elev"] > 0).to_numpy()
    x = d.loc[m, ["site_id", "ts", "energy", "capacity_kw"]].copy()
    x["cf"] = x["energy"] / x["capacity_kw"]
    x["ref"] = x.groupby("ts")["cf"].transform("median")
    x = x[x["ref"] > 0.05]
    x["ratio"] = x["cf"] / x["ref"]
    h = x["ts"].dt.hour
    x["band"] = np.select([h.between(8, 10), h.between(11, 15), h.between(16, 18)], ["eff_am", "eff_mid", "eff_pm"], "")
    x["day"] = (x["ts"] - pd.Timedelta(hours=1)).dt.floor("D")
    return x[["site_id", "ts", "ratio", "band", "day"]].reset_index(drop=True)


def _eff_window(x, end):
    """_eff_prep 결과에서 (end − EFF_WINDOW_DAYS, end] 창의 발전소별 효율 (site_id 인덱스, EFF_FEATS 열)."""
    w = x[(x["ts"] > end - pd.Timedelta(days=EFF_WINDOW_DAYS)) & (x["ts"] <= end)]
    out = w[w["band"] != ""].pivot_table(index="site_id", columns="band", values="ratio", aggfunc="median")
    out["eff_days"] = w.groupby("site_id")["day"].nunique()
    return out.reindex(columns=EFF_FEATS).astype("float32")


def site_efficiency(d, end):
    """(v7 방식) d: site_id·ts·energy·capacity_kw·keep·elev. end까지의 학습 데이터만 쓴다.

    반환: site_id별 eff_am(8~10시)·eff_mid(11~15시)·eff_pm(16~18시) = 같은 시각 전 발전소 설비이용률 중앙값 대비
    그 발전소 비율의 중앙값, eff_days = 효율을 잰 날 수. 이력이 없는 발전소는 NaN(LightGBM이 결측으로 처리).
    v7 재현을 위해 결과는 이전 구현과 같다.
    """
    start = end - pd.Timedelta(days=EFF_WINDOW_DAYS)
    win = d[(d["ts"] > start) & (d["ts"] <= end)]
    return _eff_window(_eff_prep(win), end)


def eff_source(gen, sites):
    """발전량만으로 효율 계산용 프레임을 만든다 (예보 불필요 → 학습 구간 앞쪽 이력까지 쓸 수 있다).

    gen: load_generation 결과(site_id 문자열·ts·energy). 반환 site_id는 build_frame과 같은 정수 코드.
    keep은 clean_flags(음수·cf>1.2·주간 전부 0인 날 제외)와 같다.
    """
    from .features import clean_flags
    from .solar import solar
    sc = {s: i for i, s in enumerate(sites["site_id"])}
    d = gen[["site_id", "ts", "energy"]].merge(sites[["site_id", "capacity_kw", "lat", "lon"]], on="site_id",
                                               how="inner")
    d["site_id"] = d["site_id"].map(sc).astype("int32")
    d["elev"], _, _ = solar(pd.DatetimeIndex(d["ts"]), d["lat"].to_numpy(), d["lon"].to_numpy())
    keep, _ = clean_flags(d)
    return d[["site_id", "ts", "energy", "capacity_kw", "elev"]].assign(keep=keep.to_numpy())


def eff_blocks(ts, mode, freeze_at=None):
    """행마다 효율을 잴 블록의 첫 대상일(00:00, KST). freeze_at 이후 행은 freeze_at의 대상일로 고정(평가 연도 조건)."""
    td = (pd.Series(pd.DatetimeIndex(ts)) - pd.Timedelta(hours=1)).dt.floor("D")
    u = pd.DatetimeIndex(td.unique())
    if mode == "month":
        y, mth = u.year, u.month
    elif mode == "year":
        y = np.where(u.month >= EFF_YEAR_START_MONTH, u.year, u.year - 1)
        mth = np.full(len(u), EFF_YEAR_START_MONTH)
    else:
        raise ValueError(mode)
    start = pd.to_datetime(pd.DataFrame({"year": y, "month": mth, "day": 1})).dt.tz_localize(TZ)
    blk = td.map(pd.Series(start.to_numpy(), index=u))
    if freeze_at is not None:
        f0 = (pd.Timestamp(freeze_at) - pd.Timedelta(hours=1)).floor("D")
        blk = blk.where(pd.Series(pd.DatetimeIndex(ts)) < pd.Timestamp(freeze_at), f0)
    return pd.DatetimeIndex(blk)


def site_efficiency_pit(x, site_id, ts, mode, freeze_at=None, log=None):
    """시점 기준 효율 피처. x = _eff_prep(eff_source(...)).

    각 행은 자기 블록 첫 대상일의 예측시점(전일 17:00)까지, 직전 EFF_WINDOW_DAYS일 실적으로 잰 효율을 받는다.
    → 학습 행의 피처에 그 행의 정답(또는 이후 실적)이 들어가지 않는다.
    반환: 행 순서 그대로의 DataFrame (EFF_FEATS 열, float32). 이력이 없으면 NaN.
    """
    blk = eff_blocks(ts, mode, freeze_at)
    tabs = []
    for b in pd.DatetimeIndex(blk.unique()).sort_values():
        e = _eff_window(x, b - EFF_CUTOFF)
        e.index.name = "site_id"
        tabs.append(e.reset_index().assign(blk=b))
        if log:
            log(f"    효율 블록 {b.date()}: {e['eff_mid'].notna().sum()}곳, "
                f"eff_days 중앙값 {e['eff_days'].median():.0f}")
    t = pd.concat(tabs, ignore_index=True)
    t["site_id"] = t["site_id"].astype("int32")
    left = pd.DataFrame({"site_id": np.asarray(site_id, dtype="int32"), "blk": blk})
    out = left.merge(t, on=["site_id", "blk"], how="left")
    assert len(out) == len(left)
    return out[EFF_FEATS].astype("float32").reset_index(drop=True)


def attach_eff(d, name, eff_x=None, fit_end=None, freeze_at=None, log=None):
    """변형 name의 효율 피처를 d에 붙인다 (제자리). 효율 피처가 없는 변형이면 아무것도 안 한다.

    True(v7)   : site_efficiency(d, fit_end) 한 값을 전 행에
    year/month : site_efficiency_pit(eff_x, ...)  (freeze_at 이후 행은 고정)
    """
    mode = VARIANTS[name].get("eff")
    if not mode:
        return
    if mode is True:
        eff = site_efficiency(d, fit_end)
        for c in EFF_FEATS:
            d[c] = d["site_id"].map(eff[c]).astype("float32")
    else:
        e = site_efficiency_pit(eff_x, d["site_id"].to_numpy(), d["ts"], mode, freeze_at, log)
        for c in EFF_FEATS:
            d[c] = e[c].to_numpy()


def clear_sky_cf(cosz):
    """Haurwitz 청천 일사(W/m²) ÷ 1000. 태양 천정각만 쓰는 공식이라 외부 데이터가 없다."""
    c = np.maximum(np.asarray(cosz, dtype="float64"), 0.0)
    ghi = np.where(c > 0.01, 1098.0 * c * np.exp(-0.057 / np.maximum(c, 0.01)), 0.0)
    return ghi / 1000.0


CSI_FLOOR = 0.05   # 해 뜰 무렵 청천값이 0에 가까워 비율이 튀는 것을 막는 하한

# v4 앙상블: 변형별로 라운드를 (1−W17)·r10 + W17·r17로 묶고 변형끼리 평균
V4_VARIANTS = ("v2", "csi")
W17 = 0.7


def target_and_weight(d, target):
    """학습 타깃과 가중치(capacity × k). csi는 |cf 오차| = cs·|csi 오차|라 가중치에 cs를 곱해 같은 손실을 최적화한다."""
    from .model import sample_weight
    cf = (d["energy"] / d["capacity_kw"]).to_numpy("float64")
    w = sample_weight(d).astype("float64")
    if target == "cf":
        return cf, w
    cs = np.maximum(clear_sky_cf(d["cosz"]), CSI_FLOOR)
    return np.clip(cf / cs, 0, 3), w * cs


def predict_cf(m, d, feats, target):
    """설비이용률 예측 (야간 0, 0~PRED_CF_CLIP)."""
    from .config import PRED_CF_CLIP
    cf = np.zeros(len(d))
    day = (d["elev"] > 0).to_numpy()
    p = m.predict(d.loc[day, feats])
    if target == "csi":
        p = p * np.maximum(clear_sky_cf(d.loc[day, "cosz"]), CSI_FLOOR)
    cf[day] = p
    return np.clip(cf, 0, PRED_CF_CLIP)
