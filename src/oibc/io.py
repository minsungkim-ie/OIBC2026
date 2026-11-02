"""로딩: 필요한 열만 읽고 ts는 tz-aware KST(ns)로 통일, 수치형 float32."""
import numpy as np
import pandas as pd

from .config import DATA, TZ


def to_kst(s):
    """문자열/tz-aware 시각 → Asia/Seoul, ns 단위(병합 시 단위 불일치 방지)."""
    s = pd.to_datetime(s)
    if getattr(s.dt, "tz", None) is None:
        s = s.dt.tz_localize(TZ)
    else:
        s = s.dt.tz_convert(TZ)
    return s.astype(f"datetime64[ns, {TZ}]")


def load_sites():
    return pd.read_csv(DATA / "sites.csv").sort_values("site_id").reset_index(drop=True)


def load_generation(start=None, end=None):
    g = pd.read_parquet(DATA / "train" / "generation.parquet", columns=["site_id", "ts", "energy"])
    g["ts"] = to_kst(g["ts"])
    if start is not None:
        g = g[(g.ts >= start) & (g.ts <= end)]
    g["energy"] = g["energy"].astype("float32")
    return g.reset_index(drop=True)


def load_forecast(split, kind, cols, start=None, end=None):
    """kind: 'wa'|'wb'. 반환 열: station, round, ts + cols."""
    key = f"{kind}_station"
    f = pd.read_parquet(DATA / split / f"forecast_{kind}.parquet", columns=[key, "round", "ts"] + cols)
    f["ts"] = to_kst(f["ts"])
    if start is not None:
        f = f[(f.ts >= start) & (f.ts <= end)]
    f[cols] = f[cols].astype("float32")
    return f.reset_index(drop=True)


def load_smp_train():
    s = pd.read_csv(DATA / "train" / "smp.csv")
    s["ts"] = to_kst(s["ts"])
    s["smp"] = s["smp"].astype("float64")
    return s


def load_smp_test():
    s = pd.read_csv(DATA / "smp.csv")
    s["ts"] = to_kst(s["ts"])
    return s


def hourly_index(start, end):
    return pd.DatetimeIndex(pd.date_range(start, end, freq="h")).as_unit("ns")


def k_of(ts):
    """가중치 k: ts의 시가 11~15이면 3, 그 외 1."""
    h = pd.DatetimeIndex(ts).hour
    return np.where((h >= 11) & (h <= 15), 3, 1)
