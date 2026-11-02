"""로컬 채점기: S = Σ SMP·실적 − 0.10·Σ SMP·k·|예측−실적| (그룹 합산 후)."""
import numpy as np
import pandas as pd

from .config import PENALTY_RATE
from .io import k_of


def group_frame(df, comp, smp):
    """df: site_id, ts, pred_kwh, energy(실적 있는 행만). comp: Series site→group. smp: Series ts→smp."""
    d = df[["site_id", "ts", "pred_kwh", "energy"]].copy()
    d["group"] = d["site_id"].map(comp)
    assert d["group"].notna().all()
    g = d.groupby(["group", "ts"], as_index=False).agg(pred=("pred_kwh", "sum"), act=("energy", "sum"))
    g["smp"] = g["ts"].map(smp)
    assert g["smp"].notna().all(), "SMP 매칭 실패"
    g["k"] = k_of(g["ts"])
    g["err"] = g["pred"] - g["act"]
    g["pen"] = PENALTY_RATE * g["smp"] * g["k"] * g["err"].abs()
    g["rev"] = g["smp"] * g["act"]
    return g


def summarize(g, cap_by_group=None):
    rev, W = g["rev"].sum(), g["pen"].sum()
    # W_LB: 두 그룹 오차 합의 절댓값으로 계산한 하한 (구성과 무관)
    tot = g.groupby("ts").agg(e=("err", "sum"), smp=("smp", "first"), k=("k", "first"))
    W_LB = (PENALTY_RATE * tot["smp"] * tot["k"] * tot["e"].abs()).sum()
    out = dict(S=rev - W, rev=rev, W=W, W_LB=W_LB, abs_err=g["err"].abs().sum())
    if cap_by_group is not None:
        day = g["pred"] > 0
        out["nMAE_day"] = (g.loc[day, "err"].abs() / g.loc[day, "group"].map(cap_by_group)).mean()
    for grp, sub in g.groupby("group"):
        out[f"S_{grp}"] = sub["rev"].sum() - sub["pen"].sum()
        out[f"W_{grp}"] = sub["pen"].sum()
    return out


def breakdown(g):
    m = g.assign(month=g["ts"].dt.strftime("%Y-%m"), hour=g["ts"].dt.hour)
    return m.groupby("month")["pen"].sum(), m.groupby("hour")["pen"].sum()


def residual_matrix(df, sites_order, ts_index, smp, pred_col="pred_kwh"):
    """R[i,t]=pred−actual (실적 결측은 0), w[t]=0.10·SMP·k."""
    pos = pd.Series(np.arange(len(ts_index)), index=ts_index)
    si = pd.Series(np.arange(len(sites_order)), index=sites_order)
    R = np.zeros((len(sites_order), len(ts_index)), dtype="float64")
    i = df["site_id"].map(si).to_numpy()
    t = df["ts"].map(pos).to_numpy()
    R[i, t] = (df[pred_col] - df["energy"]).to_numpy()
    w = PENALTY_RATE * smp.reindex(ts_index).to_numpy() * k_of(ts_index)
    return R, w
