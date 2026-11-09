"""2폴드(연 단위 롤링) 검증 설정과 공통 함수.

폴드마다 "앞 2년 학습 → 다음 1년 검증". 평가 구간(1년)과 같은 모양을 두 해에 반복해,
두 해 모두에서 W가 줄어드는 변경만 채택한다.
"""
import numpy as np
import pandas as pd

from .config import FEATURES, ROOT, _ts

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

# eff=True면 학습 구간 마지막 EFF_WINDOW_DAYS일로 site_efficiency를 계산해 붙인다
VARIANTS = {
    "v1": dict(feats=V1_FEATURES, target="cf"),
    "v2": dict(feats=FEATURES, target="cf"),
    "csi": dict(feats=FEATURES, target="csi"),
    "v2e": dict(feats=FEATURES + EFF_FEATS, target="cf", eff=True),
    "csie": dict(feats=FEATURES + EFF_FEATS, target="csi", eff=True),
}


def site_efficiency(d, end):
    """d: site_id·ts·energy·capacity_kw·keep·elev. end까지의 학습 데이터만 쓴다(검증·평가 구간 실적 미사용).

    반환: site_id별 eff_am(8~10시)·eff_mid(11~15시)·eff_pm(16~18시) = 같은 시각 전 발전소 설비이용률 중앙값 대비
    그 발전소 비율의 중앙값, eff_days = 효율을 잰 날 수. 이력이 없는 발전소는 NaN(LightGBM이 결측으로 처리).
    """
    start = end - pd.Timedelta(days=EFF_WINDOW_DAYS)
    m = (d["ts"] > start) & (d["ts"] <= end) & d["keep"] & (d["elev"] > 0)
    x = d.loc[m, ["site_id", "ts", "energy", "capacity_kw"]].copy()
    x["cf"] = x["energy"] / x["capacity_kw"]
    x["ref"] = x.groupby("ts")["cf"].transform("median")
    x = x[x["ref"] > 0.05]
    x["ratio"] = x["cf"] / x["ref"]
    h = x["ts"].dt.hour
    x["band"] = np.select([h.between(8, 10), h.between(11, 15), h.between(16, 18)], ["eff_am", "eff_mid", "eff_pm"], "")
    out = x[x["band"] != ""].pivot_table(index="site_id", columns="band", values="ratio", aggfunc="median")
    out["eff_days"] = x.groupby("site_id")["ts"].apply(lambda t: (t - pd.Timedelta(hours=1)).dt.floor("D").nunique())
    return out.reindex(columns=EFF_FEATS).astype("float32")


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
