"""v17 후보 평가: 기준 = 발전소 합(v9 OOF 대용) 0.7 + 그룹 모델 g(트리 700 고정) 0.3 (v15 구조).
여기에 그룹 합 손실 신경망(22_nn_cv OOF, 시드 평균)을 u 비중으로 섞어 v4 구성 W(교차 α)를 비교한다.

사용: python scripts/23_nn_blend.py <seed,seed,...> [폴드...]
"""
import sys
import warnings

from _bootstrap import log
import numpy as np
import pandas as pd

from oibc.config import PENALTY_RATE, ROOT, ROUNDS
from oibc.cv import CACHE, CV_START, FOLDS, OUT_CV, W17
from oibc.group import fit_group, group_frame, joint, predict_group_kwh, train_rows
from oibc.io import k_of, load_sites, load_smp_train

warnings.filterwarnings("ignore")
seeds = sys.argv[1].split(",")
folds = sys.argv[2:] or list(FOLDS)
OUT = ROOT / "outputs" / "v17"
SPEC = [("v2", .125), ("csi", .125), ("v2e", .125), ("csie", .125), ("v2e_c", .25), ("csie_c", .25)]
sites = load_sites()
comp = pd.read_csv(ROOT / "submissions" / "v4" / "composition.csv").set_index("site_id")["group"]
group_of = {i: comp[s] for i, s in enumerate(sites["site_id"])}
smp = load_smp_train().set_index("ts")["smp"]

gcache = OUT / "g700_oof.parquet"
if gcache.exists():
    gall = pd.read_parquet(gcache)
else:
    G = {}
    for r in ROUNDS:
        d = pd.read_parquet(CACHE / f"frame_r{r}.parquet")
        G[r] = group_frame(d[d["ts"] >= CV_START], group_of)
    Gall = joint(G[17], G[10])
    parts = []
    for f, (t0, t1, v0, v1) in FOLDS.items():
        m = fit_group(train_rows(Gall, CV_START, t1), smp, n_estimators=700)
        p = predict_group_kwh(m, Gall[(Gall["ts"] >= v0) & (Gall["ts"] <= v1)]).rename("g_kwh").reset_index()
        parts.append(p.assign(fold=f))
    gall = pd.concat(parts)
    gall.to_parquet(gcache, index=False)


def W(g, col, a):
    return float((g["w"] * (a * g[col] - g["energy"]).abs()).sum()) / 1e6


def best_a(g, col):
    grid = np.arange(0.90, 1.10, 0.002)
    return grid[int(np.argmin([W(g, col, a) for a in grid]))]


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
    gp = gall[gall["fold"] == f].set_index(["g", "ts"])["g_kwh"].reindex(g.index)
    g["base"] = np.where(gp.notna(), 0.7 * g["pred"] + 0.3 * gp.fillna(0), g["pred"])
    nn = np.mean([pd.read_parquet(OUT / f"nn_oof_{f}_s{s}.parquet").set_index(["g", "ts"])["nn_kwh"]
                  .reindex(g.index).to_numpy() for s in seeds], axis=0)
    g["nn"] = np.nan_to_num(nn)
    cur[f] = g
    c = np.corrcoef(g["base"] - g["energy"], g["nn"] - g["energy"])[0, 1]
    log(f"폴드 {f}: 오차 상관(기준 vs NN) {c:.3f}, 합계 기준 {g['base'].sum() / 1e6:.2f} / NN {g['nn'].sum() / 1e6:.2f}"
        f" / 실적 {g['energy'].sum() / 1e6:.2f} GWh")
rows = []
for u in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
    for f in folds:
        cur[f]["bl"] = (1 - u) * cur[f]["base"] + u * cur[f]["nn"]
    ab = {f: best_a(cur[f], "bl") for f in folds}
    rec = dict(u=u)
    for f in folds:
        o = [x for x in folds if x != f]
        rec[f"W_{f}_in"] = W(cur[f], "bl", ab[f])
        rec[f"W_{f}_xa"] = W(cur[f], "bl", ab[o[0]]) if o else np.nan
        rec[f"a_{f}"] = ab[f]
    rows.append(rec)
res = pd.DataFrame(rows)
for f in folds:
    k = f"W_{f}_xa" if len(folds) > 1 else f"W_{f}_in"
    res[f"d{f}_%"] = 100 * (res[k] / res[k].iloc[0] - 1)
print(res.round(3).to_string(index=False))
res.to_csv(OUT / f"23_nn_blend_{'_'.join(seeds)}.csv", index=False)
