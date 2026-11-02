"""(site, ts) 격자 + 예보 병합 + 파생 피처. 파생은 모두 같은 대상일(target_date) 안에서만 계산한다."""
import numpy as np
import pandas as pd

from .config import (AGG_FEATS, FEATURES, LAG_COLS, SITE_FEATS, WA_COLS, WB_COLS)
from .solar import solar

ONE_H = pd.Timedelta(hours=1)


def target_date(ts):
    """대상일 = (ts - 1h).date(). ts=00:00은 전날 23~24시 구간."""
    return (ts - ONE_H).dt.floor("D")


def station_features(f, key, kind, sites, log=print):
    """한 라운드의 (지점, ts) 예보에 같은 대상일 안 시차/선행·일집계를 붙인다."""
    f = f.sort_values([key, "ts"]).reset_index(drop=True)
    f["target_date"] = target_date(f["ts"])
    ll = sites.groupby(key)[["lat", "lon"]].mean()          # 지점 주변 발전소 평균 좌표(주간 판정용)
    lat = f[key].map(ll["lat"]).to_numpy(dtype="float64")
    lon = f[key].map(ll["lon"]).to_numpy(dtype="float64")
    elev, _, _ = solar(pd.DatetimeIndex(f["ts"]), lat, lon)
    grp = f.groupby([key, "target_date"], sort=False)
    for c in LAG_COLS[kind]:
        f[f"{c}_m1"] = grp[c].shift(1).fillna(f[c])           # 대상일 경계는 자기 값
        f[f"{c}_p1"] = grp[c].shift(-1).fillna(f[c])
    cloud = f"{kind}_cloud"
    day = f[cloud].where(elev > 0)
    f[f"{cloud}_dmean"] = day.groupby([f[key], f["target_date"]]).transform("mean").fillna(f[cloud])
    if kind == "wb":
        f["wb_total_liq_dsum"] = grp["wb_total_liq"].transform("sum")
    num = [c for c in f.columns if c not in (key, "ts", "target_date")]
    f[num] = f[num].astype("float32")
    return f.drop(columns="target_date")


def build_frame(base, sites, fa_r, fb_r):
    """base: [site_id, ts(, energy)]. fa_r/fb_r: station_features 결과(라운드 하나)."""
    sc = {s: i for i, s in enumerate(sites["site_id"])}
    d = base.merge(sites.drop(columns=["current_group"]), on="site_id", how="left")
    d["site_id"] = d["site_id"].map(sc).astype("int32")
    d = d.merge(fa_r, on=["wa_station", "ts"], how="left")
    d = d.merge(fb_r, on=["wb_station", "ts"], how="left")
    d = d.drop(columns=["wa_station", "wb_station"])
    ts = pd.DatetimeIndex(d["ts"])
    d["elev"], d["azim"], d["cosz"] = solar(ts, d["lat"].to_numpy(), d["lon"].to_numpy())
    doy = ts.dayofyear.to_numpy()
    toa = 1361.0 * (1 + 0.033 * np.cos(2 * np.pi * doy / 365)) * np.maximum(d["cosz"].to_numpy(), 0)
    d["toa"] = toa.astype("float32")
    d["toa_x_clear_a"] = (d["toa"] * (1 - d["wa_cloud"] / 100)).astype("float32")
    d["toa_x_clear_b"] = (d["toa"] * (1 - d["wb_cloud"] / 100)).astype("float32")
    d["hour"] = ts.hour.to_numpy().astype("float32")
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")
    d["month"] = ts.month.to_numpy().astype("float32")
    for c in ["capacity_kw", "lat", "lon", "wa_dist", "wb_dist"]:
        d[c] = d[c].astype("float32")
    na = d[FEATURES].isna().sum().sum()
    assert na == 0, f"피처 NaN {na}개 (예보 대체 누락?)"
    return d


def clean_flags(d):
    """학습 표본 제외 마스크(발전량 정제). 반환: keep(bool), 로그용 카운트 dict."""
    cf = d["energy"] / d["capacity_kw"]
    neg = d["energy"] < 0
    big = cf > 1.2
    day = d["elev"] > 0
    tdate = target_date(d["ts"])
    dsum = d["energy"].where(day, 0).groupby([d["site_id"], tdate]).transform("sum")
    has_day = day.groupby([d["site_id"], tdate]).transform("sum") > 0
    allzero = has_day & (dsum <= 0)
    keep = ~(neg | big | allzero)
    return keep, dict(neg=int(neg.sum()), cf_gt_1_2=int(big.sum()),
                      allzero_day_rows=int(allzero.sum()), total=len(d))
