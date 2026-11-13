"""v18: 신경망 변형 묶음 비교. 기준 = v17 레시피(발전소 합 v9 OOF 0.7 + g700 0.3, 여기에 v17 NN 시드 42~44 u 0.3).

사용: python scripts/27_nn_blend_v18.py <후보> [<후보> ...]
  후보 = "태그:시드,시드+태그:시드,..." (나열한 OOF 전부 평균). 태그 v17 = outputs/v17/nn_oof_<폴드>_s<시드>,
  그 밖 = outputs/v18/nn_oof_<폴드>_<태그>_s<시드>.
각 후보를 u 0.2~0.6으로 섞어 v4 구성 W(교차 α)와 S를 내고, 기준 대비 %를 출력한다. → outputs/v18/27_blend.csv에 덧붙임
"""
import os
import sys
import warnings

from _bootstrap import log
import numpy as np
import pandas as pd

from oibc.config import PENALTY_RATE, ROOT
from oibc.cv import FOLDS, OUT_CV, W17
from oibc.io import k_of, load_sites, load_smp_train

warnings.filterwarnings("ignore")
OUT = ROOT / "outputs" / "v18"
SPEC = [("v2", .125), ("csi", .125), ("v2e", .125), ("csie", .125), ("v2e_c", .25), ("csie_c", .25)]
BASELINE = "v17:42,43,44"
sites = load_sites()
comp = pd.read_csv(ROOT / "submissions" / "v4" / "composition.csv").set_index("site_id")["group"]
smp = load_smp_train().set_index("ts")["smp"]
gall = pd.read_parquet(ROOT / "outputs" / "v17" / "g700_oof.parquet")
folds = os.environ.get("FOLDS", ",".join(FOLDS)).split(",")


def files(spec, f):
    out = []
    for part in spec.split("+"):
        tag, seeds = part.split(":")
        for s in seeds.split(","):
            out.append(ROOT / "outputs" / "v17" / f"nn_oof_{f}_s{s}.parquet" if tag == "v17"
                       else OUT / f"nn_oof_{f}_{tag}_s{s}.parquet")
    return out


cur = {}
for f in folds:
    parts = []
    for v, w in SPEC:
        a = pd.read_parquet(OUT_CV / "oof" / f"{v}_{f}_r10.parquet")
        b = pd.read_parquet(OUT_CV / "oof" / f"{v}_{f}_r17.parquet")
        parts.append(w * ((1 - W17) * a["pred_kwh"].to_numpy() + W17 * b["pred_kwh"].to_numpy()))
    a["pred"] = np.sum(parts, axis=0)
    g = a.assign(g=a["site_id"].map(comp)).groupby(["g", "ts"])[["pred", "energy"]].sum()
    g["w"] = PENALTY_RATE * g.index.get_level_values("ts").map(smp).to_numpy() * k_of(g.index.get_level_values("ts"))
    g["rev"] = g.index.get_level_values("ts").map(smp).to_numpy() * g["energy"]
    gp = gall[gall["fold"] == f].set_index(["g", "ts"])["g_kwh"].reindex(g.index)
    g["base"] = np.where(gp.notna(), 0.7 * g["pred"] + 0.3 * gp.fillna(0), g["pred"])
    cur[f] = g


def nn(spec, f):
    g = cur[f]
    v = np.mean([pd.read_parquet(p).set_index(["g", "ts"])["nn_kwh"].reindex(g.index).to_numpy()
                 for p in files(spec, f)], axis=0)
    return np.nan_to_num(v)


def W(g, x, a):
    return float((g["w"] * (a * x - g["energy"]).abs()).sum())


def best_a(g, x):
    grid = np.arange(0.90, 1.10, 0.002)
    return grid[int(np.argmin([W(g, x, a) for a in grid]))]


def evaluate(spec, us):
    n = {f: nn(spec, f) for f in folds}
    rows = []
    for u in us:
        x = {f: (1 - u) * cur[f]["base"].to_numpy() + u * n[f] for f in folds}
        ab = {f: best_a(cur[f], x[f]) for f in folds}
        rec = dict(spec=spec, u=u)
        for f in folds:
            o = ([k for k in folds if k != f] or [f])[0]
            w = W(cur[f], x[f], ab[o])
            rec[f"W_{f}"] = w / 1e6
            rec[f"S_{f}"] = (float(cur[f]["rev"].sum()) - w) / 1e6
            rec[f"a_{f}"] = ab[f]
            rec[f"sum_{f}"] = float(n[f].sum()) / 1e6
        rows.append(rec)
    return rows


base = pd.DataFrame(evaluate(BASELINE, [0.3])).iloc[0]
log("기준(v17 레시피) " + ", ".join(f"{f}: W {base[f'W_{f}']:.3f}M S {base[f'S_{f}']:.1f}M" for f in folds))
rows = []
for spec in sys.argv[1:]:
    for f in folds:
        c = np.corrcoef(cur[f]["base"] - cur[f]["energy"], nn(spec, f) - cur[f]["energy"])[0, 1]
        log(f"{spec} 폴드 {f}: 오차 상관(기준 vs NN) {c:.3f}")
    rows += evaluate(spec, (0.2, 0.3, 0.35, 0.4, 0.5, 0.6))
res = pd.DataFrame(rows)
for f in folds:
    res[f"d{f}_%"] = 100 * (res[f"W_{f}"] / base[f"W_{f}"] - 1)
cols = ["spec", "u"] + [c for f in folds for c in (f"W_{f}", f"S_{f}", f"d{f}_%")] + [f"a_{f}" for f in folds] + \
       [f"sum_{f}" for f in folds]
print(res[cols].round(3).to_string(index=False))
p = OUT / "27_blend.csv"
res[cols].to_csv(p, mode="a", header=not p.exists(), index=False)
