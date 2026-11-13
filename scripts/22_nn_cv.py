"""v17 후보: 그룹 합 손실 하루 단위 신경망(oibc.seqnn) 2폴드 OOF.

사용: python scripts/22_nn_cv.py <seed> [폴드...]   → outputs/v17/nn_oof_<폴드>_s<seed>.parquet (g, ts, nn_kwh)
폴드 학습 = CV_START ~ 학습 끝(4년 변형과 같은 규칙), 마지막 60일로 조기종료(그룹 손실). v4 구성.
"""
import json
import sys
import warnings

from _bootstrap import log
import numpy as np
import pandas as pd
import torch

from oibc.config import ROOT
from oibc.cv import CACHE, CV_START, FOLDS
from oibc.io import load_sites, load_smp_train
from oibc.seqnn import Trainer, day_tensors, group_sum_kwh, norm_stats

warnings.filterwarnings("ignore")
import os
torch.set_num_threads(int(os.environ.get("NN_THREADS", "8")))
seed = int(sys.argv[1])
folds = sys.argv[2:] or list(FOLDS)
OUT = ROOT / "outputs" / "v17"
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
log(f"일 텐서 {D['X'].shape}, {D['X'].nbytes / 1e9:.2f} GB")
days = D["days"]
res = {}
for fo in folds:
    t0, t1, v0, v1 = FOLDS[fo]
    d1 = (t1 - pd.Timedelta(hours=1)).floor("D")
    tr = np.where(days <= d1)[0]
    es = tr[days[tr] > d1 - pd.Timedelta(days=60)]
    fit = tr[~np.isin(tr, es)]
    va = np.where((days >= (v0 - pd.Timedelta(hours=1)).floor("D")) & (days <= (v1 - pd.Timedelta(hours=1)).floor("D")))[0]
    mean, std = norm_stats(D["X"], fit)
    T = Trainer(gid, cap, mean, std, seed=seed)
    m = T.fit(D, smp, fit, es, log=log)
    sub = {k: v[va] for k, v in D.items()}
    cf = T.predict(m, sub)
    g = group_sum_kwh(cf, sub, gid, cap).rename("nn_kwh").reset_index()
    g.to_parquet(OUT / f"nn_oof_{fo}_s{seed}.parquet", index=False)
    res[fo] = m.best_epoch
    log(f"폴드 {fo}: best_epoch {m.best_epoch}, 검증 합계 {g['nn_kwh'].sum() / 1e6:.2f} GWh")
json.dump(res, open(OUT / f"nn_best_epoch_s{seed}.json", "w"))
