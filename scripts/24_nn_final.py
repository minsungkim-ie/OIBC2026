"""v17: 그룹 합 손실 하루 단위 신경망(oibc.seqnn) 최종 학습 → 평가 구간 그룹 예측.

사용: python scripts/24_nn_final.py <seed> [...]   → outputs/v17/nn_test_s<seed>.parquet (g, ts, nn_kwh)
절차는 22_nn_cv와 같다: 학습 구간 전체(2021-06-01 ~ 2025-06-01 00:00) 중 마지막 60일로 조기종료,
정규화 통계는 조기종료 앞 학습 일만. 평가 구간 입력은 그날의 전일 10시·17시 예보만 (station_tables·build_frame).
"""
import os
import sys
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
from oibc.seqnn import Trainer, day_tensors, group_sum_kwh, norm_stats

warnings.filterwarnings("ignore")
torch.set_num_threads(int(os.environ.get("NN_THREADS", "8")))
seeds = [int(s) for s in sys.argv[1:]]
OUT = ROOT / "outputs" / "v17"
OUT.mkdir(parents=True, exist_ok=True)
sites = load_sites()
comp = pd.read_csv(ROOT / "submissions" / "v4" / "composition.csv").set_index("site_id")["group"]
gid = (sites["site_id"].map(comp) == "G2").astype(int).to_numpy()
cap = sites["capacity_kw"].to_numpy("float32")
smp = load_smp_train().set_index("ts")["smp"]

f = {r: pd.read_parquet(CACHE / f"frame_r{r}.parquet") for r in ROUNDS}
D = day_tensors(f[17][f[17]["ts"] >= CV_START], f[10][f[10]["ts"] >= CV_START], len(sites))
del f
days = D["days"]
tr = np.arange(len(days))
es = tr[days > days.max() - pd.Timedelta(days=60)]
fit = tr[~np.isin(tr, es)]
mean, std = norm_stats(D["X"], fit)
log(f"학습 일 {len(fit)} ({days[fit].min().date()} ~ {days[fit].max().date()}), 조기종료 {len(es)}일")

idx = pd.DatetimeIndex(load_smp_test()["ts"])
tabs = station_tables("test", idx[0], idx[-1], sites, log)
base = pd.MultiIndex.from_product([sites["site_id"], idx], names=["site_id", "ts"]).to_frame(index=False)
Fte = {r: build_frame(base, sites, *tabs[r]) for r in ROUNDS}
Dte = day_tensors(Fte[17], Fte[10], len(sites))
del Fte
assert Dte["present"].all() and len(Dte["days"]) == 365, (Dte["present"].mean(), len(Dte["days"]))
assert np.isfinite(Dte["X"].astype("float32")).all()

for s in seeds:
    T = Trainer(gid, cap, mean, std, seed=s)
    m = T.fit(D, smp, fit, es, log=log)
    cf = T.predict(m, Dte)
    g = group_sum_kwh(cf, Dte, gid, cap).rename("nn_kwh").reset_index()
    g.to_parquet(OUT / f"nn_test_s{s}.parquet", index=False)
    log(f"시드 {s}: best_epoch {m.best_epoch}, 평가 합계 {g['nn_kwh'].sum() / 1e6:.2f} GWh")
