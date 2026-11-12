"""v9 후보: 예보 확장 피처. 모두 같은 대상일(또는 같은 ts) 안의 예보만 쓴다 (CLAUDE.md 규칙 3·4).

- 같은 대상일 안 시차/선행 ±2·±3시간 (운량·UV)
- 같은 대상일 주간 운량의 최소·최대·표준편차, UV 최대
- 같은 ts·같은 라운드의 전 지점 평균 운량(지역 날씨)과 그 차이 — 평가 기간 전체 통계가 아니라 그 시각 예보만 쓴다
- 라운드 간 차이(17시 − 10시 발행): 예보가 하루 사이 얼마나 바뀌었나 = 불확실성. 두 라운드 모두 예측시점 이전 발행
- 제공사 간 운량 차이 (A − B)
"""
import numpy as np
import pandas as pd

from .features import target_date
from .solar import solar

EXTRA_FEATS = (
    [f"{c}_{s}" for c in ("wa_cloud", "wb_cloud", "wb_uv_idx") for s in ("m2", "p2", "m3", "p3")]
    + ["wa_cloud_dmin", "wa_cloud_dmax", "wa_cloud_dstd", "wb_cloud_dmin", "wb_cloud_dmax", "wb_cloud_dstd",
       "wb_uv_idx_dmax", "wa_cloud_reg", "wa_cloud_dev", "wb_cloud_reg", "wb_cloud_dev",
       "wa_cloud_rdiff", "wb_cloud_rdiff", "wb_uv_idx_rdiff", "cloud_ab_diff"]
)


def _station_extra(f, key, kind, sites, other_round):
    """한 라운드·한 제공사의 (지점, ts) 확장 피처. other_round: 다른 라운드의 같은 테이블(라운드 간 차이용)."""
    f = f.sort_values([key, "ts"]).reset_index(drop=True)
    td = target_date(f["ts"])
    ll = sites.groupby(key)[["lat", "lon"]].mean()
    elev, _, _ = solar(pd.DatetimeIndex(f["ts"]), f[key].map(ll["lat"]).to_numpy("float64"),
                       f[key].map(ll["lon"]).to_numpy("float64"))
    out = f[[key, "ts"]].copy()
    grp_keys = [f[key], td]
    cols = ["wa_cloud"] if kind == "wa" else ["wb_cloud", "wb_uv_idx"]
    for c in cols:
        g = f[c].groupby(grp_keys)
        for k in (2, 3):
            out[f"{c}_m{k}"] = g.shift(k).fillna(f[c])
            out[f"{c}_p{k}"] = g.shift(-k).fillna(f[c])
    cloud = f[f"{kind}_cloud"].where(elev > 0)
    g = cloud.groupby(grp_keys)
    for stat in ("min", "max", "std"):
        out[f"{kind}_cloud_d{stat}"] = g.transform(stat).fillna(f[f"{kind}_cloud"] if stat != "std" else 0)
    if kind == "wb":
        out["wb_uv_idx_dmax"] = f["wb_uv_idx"].groupby(grp_keys).transform("max")
    reg = f.groupby("ts")[f"{kind}_cloud"].transform("mean")
    out[f"{kind}_cloud_reg"] = reg
    out[f"{kind}_cloud_dev"] = f[f"{kind}_cloud"] - reg
    o = other_round.set_index([key, "ts"])
    idx = pd.MultiIndex.from_frame(f[[key, "ts"]])
    for c in cols:
        out[f"{c}_rdiff"] = f[c].to_numpy() - o[c].reindex(idx).to_numpy()
    num = [c for c in out.columns if c not in (key, "ts")]
    out[num] = out[num].astype("float32")
    return out


def extra_tables(tabs, sites):
    """tabs: station_tables 결과 {round: (fa, fb)} → {round: (xa, xb)}. rdiff는 '이 라운드 − 다른 라운드'."""
    out = {}
    for r in tabs:
        o = [x for x in tabs if x != r][0]
        sign = 1.0 if r == 17 else -1.0     # 두 라운드 모두 17시 − 10시 방향으로 맞춘다
        xa = _station_extra(tabs[r][0], "wa_station", "wa", sites, tabs[o][0])
        xb = _station_extra(tabs[r][1], "wb_station", "wb", sites, tabs[o][1])
        for x, cs in ((xa, ["wa_cloud_rdiff"]), (xb, ["wb_cloud_rdiff", "wb_uv_idx_rdiff"])):
            x[cs] = x[cs] * sign
        out[r] = (xa, xb)
    return out


def attach_extra(d, sites, xa, xb):
    """d: site_id(int, sites 순서)·ts 프레임 → EXTRA_FEATS 열을 붙여 반환 (행 순서 유지)."""
    st = sites[["wa_station", "wb_station"]].reset_index(drop=True)
    k = pd.DataFrame({"wa_station": st["wa_station"].to_numpy()[d["site_id"].to_numpy()],
                      "wb_station": st["wb_station"].to_numpy()[d["site_id"].to_numpy()],
                      "ts": d["ts"].to_numpy()})
    k = k.merge(xa, on=["wa_station", "ts"], how="left").merge(xb, on=["wb_station", "ts"], how="left")
    k["cloud_ab_diff"] = (d["wa_cloud"].to_numpy() - d["wb_cloud"].to_numpy()).astype("float32")
    for c in EXTRA_FEATS:
        d[c] = k[c].to_numpy()
    return d
