"""v18: 신경망 변형 최종 학습(학습 구간 전체) → 평가 구간 그룹 예측 (24_nn_final 일반화).

사용: python scripts/28_nn_final_v18.py <tag> <seed,seed,...> [--params '{...}'] [--prev]
  → outputs/v18/nn_test_<tag>_s<seed>.parquet (g, ts, nn_kwh)
sched="cos"면 조기종료 없이 학습 구간 전체로 학습, 아니면 마지막 60일 조기종료(24_nn_final과 같음).
--prev: 평가 구간 첫날의 앞선 대상일(5/30·5/31 대상일 예보, 각각 5/29·5/30 17시 발행)은 학습 텐서에서 가져온다.
"""
import argparse
import json
import os
import warnings

from _bootstrap import log
import numpy as np
import pandas as pd
import torch

from oibc.config import ROOT, ROUNDS
from oibc.cv import CACHE, CV_START
from oibc.features import build_frame
from oibc.io import load_sites, load_smp_test, load_smp_train
from oibc.pipeline import station_tables
from oibc.seqnn import Trainer, add_prev, day_tensors, group_sum_kwh, norm_stats

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("seeds")
ap.add_argument("--params", default="{}")
ap.add_argument("--prev", action="store_true")
a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("NN_THREADS", "8")))
params = json.loads(a.params)
OUT = ROOT / "outputs" / "v18"
OUT.mkdir(parents=True, exist_ok=True)
sites = load_sites()
comp = pd.read_csv(ROOT / "submissions" / "v4" / "composition.csv").set_index("site_id")["group"]
gid = (sites["site_id"].map(comp) == "G2").astype(int).to_numpy()
cap = sites["capacity_kw"].to_numpy("float32")
smp = load_smp_train().set_index("ts")["smp"]

f = {r: pd.read_parquet(CACHE / f"frame_r{r}.parquet") for r in ROUNDS}
D = day_tensors(f[17][f[17]["ts"] >= CV_START], f[10][f[10]["ts"] >= CV_START], len(sites))
del f

idx = pd.DatetimeIndex(load_smp_test()["ts"])
tabs = station_tables("test", idx[0], idx[-1], sites, log)
base = pd.MultiIndex.from_product([sites["site_id"], idx], names=["site_id", "ts"]).to_frame(index=False)
Fte = {r: build_frame(base, sites, *tabs[r]) for r in ROUNDS}
Dte = day_tensors(Fte[17], Fte[10], len(sites))
del Fte
assert Dte["present"].all() and len(Dte["days"]) == 365, (Dte["present"].mean(), len(Dte["days"]))
if a.prev:
    Dte = add_prev(Dte, src={"days": D["days"][-3:], "X": D["X"][-3:]})
    D = add_prev(D)
assert np.isfinite(Dte["X"].astype("float32")).all()

days = D["days"]
tr = np.arange(len(days))
if params.get("sched") == "cos":
    fit, es = tr, None
else:
    es = tr[days > days.max() - pd.Timedelta(days=60)]
    fit = tr[~np.isin(tr, es)]
mean, std = norm_stats(D["X"], fit)
log(f"[{a.tag}] 학습 일 {len(fit)}, 조기종료 {0 if es is None else len(es)}일, X {D['X'].shape}, params {params}")

for s in [int(x) for x in a.seeds.split(",")]:
    T = Trainer(gid, cap, mean, std, seed=s, params=params)
    m = T.fit(D, smp, fit, es, log=log)
    cf = T.predict(m, Dte)
    g = group_sum_kwh(cf, Dte, gid, cap).rename("nn_kwh").reset_index()
    g.to_parquet(OUT / f"nn_test_{a.tag}_s{s}.parquet", index=False)
    log(f"[{a.tag}] 시드 {s}: best_epoch {m.best_epoch}, 평가 합계 {g['nn_kwh'].sum() / 1e6:.2f} GWh")
