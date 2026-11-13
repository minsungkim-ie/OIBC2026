"""v18: 그룹 합 손실 신경망 변형 2폴드 OOF (22_nn_cv 일반화).

사용: python scripts/26_nn_cv_v18.py <tag> <seed,seed,...> [--params '{"sched":"cos","max_epochs":20}'] [--prev] [--folds A,B]
  → outputs/v18/nn_oof_<폴드>_<tag>_s<seed>.parquet (g, ts, nn_kwh)
sched="cos"면 조기종료 없이 폴드 학습 구간 전체로 학습(최종 학습과 같은 규칙). 아니면 22_nn_cv처럼 마지막 60일 조기종료.
--prev: 앞선 대상일 예보 흐름 피처(seqnn.add_prev) 추가. 정규화 통계는 폴드 학습 일만.
"""
import argparse
import json
import os
import warnings

from _bootstrap import log
import numpy as np
import pandas as pd
import torch

from oibc.config import ROOT
from oibc.cv import CACHE, CV_START, FOLDS
from oibc.io import load_sites, load_smp_train
from oibc.seqnn import Trainer, add_prev, day_tensors, group_sum_kwh, norm_stats

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("tag")
ap.add_argument("seeds")
ap.add_argument("--params", default="{}")
ap.add_argument("--prev", action="store_true")
ap.add_argument("--folds", default="A,B")
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

f = {r: pd.read_parquet(CACHE / f"frame_r{r}.parquet") for r in (10, 17)}
f = {r: d[d["ts"] >= CV_START] for r, d in f.items()}
D = day_tensors(f[17], f[10], len(sites))
del f
if a.prev:
    D = add_prev(D)
log(f"[{a.tag}] 일 텐서 {D['X'].shape}, params {params}, prev {a.prev}")
days = D["days"]
for seed in [int(s) for s in a.seeds.split(",")]:
    res = {}
    for fo in a.folds.split(","):
        t0, t1, v0, v1 = FOLDS[fo]
        d1 = (t1 - pd.Timedelta(hours=1)).floor("D")
        tr = np.where(days <= d1)[0]
        if params.get("sched") == "cos":
            fit, es = tr, None
        else:
            es = tr[days[tr] > d1 - pd.Timedelta(days=60)]
            fit = tr[~np.isin(tr, es)]
        va = np.where((days >= (v0 - pd.Timedelta(hours=1)).floor("D")) & (days <= (v1 - pd.Timedelta(hours=1)).floor("D")))[0]
        mean, std = norm_stats(D["X"], fit)
        T = Trainer(gid, cap, mean, std, seed=seed, params=params)
        m = T.fit(D, smp, fit, es, log=log)
        sub = {k: v[va] for k, v in D.items()}
        cf = T.predict(m, sub)
        g = group_sum_kwh(cf, sub, gid, cap).rename("nn_kwh").reset_index()
        g.to_parquet(OUT / f"nn_oof_{fo}_{a.tag}_s{seed}.parquet", index=False)
        res[fo] = m.best_epoch
        log(f"[{a.tag}] 시드 {seed} 폴드 {fo}: best_epoch {m.best_epoch}, 검증 합계 {g['nn_kwh'].sum() / 1e6:.2f} GWh")
    json.dump(res, open(OUT / f"nn_best_epoch_{a.tag}_s{seed}.json", "w"))
