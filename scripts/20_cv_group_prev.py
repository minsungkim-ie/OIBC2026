"""v17 후보 2폴드 검증: 그룹 모델(g)에 앞선 대상일 예보 흐름 피처(group.attach_prev)를 더한 gp 비교.

사용: python scripts/20_cv_group_prev.py
- 발전소 합 = 이 PC에 있는 v9 6개 OOF(v2·csi·v2e·csie·v2e_c·csie_c, 가중 .125×4·.25×2). v15의 확장 피처 발전소 합
  OOF는 이 PC에 없어 대용으로 쓴다 (그룹 모델끼리의 상대 비교가 목적).
- 그룹 모델은 17_cv_group_blend와 같은 규칙(CV_START~학습 끝, 마지막 60일 조기종료). v4 구성, 전체 발전소, 교차 α.
"""
from _bootstrap import log
import warnings
import numpy as np
warnings.filterwarnings("ignore")
import pandas as pd

from oibc.config import PENALTY_RATE, ROOT, ROUNDS
from oibc.cv import CACHE, CV_START, FOLDS, OUT_CV, W17
from oibc.group import (attach_prev, day_agg, fit_group, group_frame, group_feats, joint, predict_group_kwh,
                        train_rows)
from oibc.io import k_of, load_sites, load_smp_train

SPEC = [("v2", .125), ("csi", .125), ("v2e", .125), ("csie", .125), ("v2e_c", .25), ("csie_c", .25)]
BLEND_W = (0.0, 0.3, 0.4, 0.5)

sites = load_sites()
comp = pd.read_csv(ROOT / "submissions" / "v4" / "composition.csv").set_index("site_id")["group"]
group_of = {i: comp[s] for i, s in enumerate(sites["site_id"])}
smp = load_smp_train().set_index("ts")["smp"]
G = {}
for r in ROUNDS:
    d = pd.read_parquet(CACHE / f"frame_r{r}.parquet")
    G[r] = group_frame(d[d["ts"] >= CV_START], group_of)
    del d
Gall = joint(G[17], G[10])
Gall = attach_prev(Gall, day_agg(Gall))
prev_cols = [c for c in Gall.columns if c.startswith(("D1_", "D2_", "D3m_", "dD1_"))]
log(f"그룹 프레임 {len(Gall):,}행, 흐름 피처 {len(prev_cols)}개")
CORE = ["wa_cloud", "wb_cloud", "wa_humidity", "wb_rel_hum", "wb_vis"]
VARIANTS = {
    "g": [],
    "gp_all": prev_cols,
    "gp_d": [c for c in prev_cols if c.startswith("dD1_")],
    "gp_core": [f"{p}{c}" for p in ("D1_", "dD1_", "D3m_") for c in CORE],
}

import sys
N_FIX = int(sys.argv[1]) if len(sys.argv) > 1 else None   # 트리 수 고정 (없으면 조기종료)
gpred = {}
for f, (t0, t1, v0, v1) in FOLDS.items():
    tr = train_rows(Gall, CV_START, t1)
    gv = Gall[(Gall["ts"] >= v0) & (Gall["ts"] <= v1)]
    for name, keep in VARIANTS.items():
        drop = [c for c in prev_cols if c not in keep]
        m = fit_group(tr.drop(columns=drop), smp, n_estimators=N_FIX)
        gpred[(name, f)] = predict_group_kwh(m, gv.drop(columns=drop))
        log(f"폴드 {f} {name}: n {N_FIX}, 피처 {len(group_feats(tr.drop(columns=drop)))}")
        if name == "gp_all":
            imp = pd.Series(m.feature_importances_, index=group_feats(tr)).sort_values(ascending=False)
            rank = {c: i + 1 for i, c in enumerate(imp.index)}
            top = [f"{c}({rank[c]})" for c in imp.index if c in prev_cols][:8]
            log(f"   흐름 피처 중요도 상위(전체 순위): {', '.join(top)}")


def site_sum(f):
    parts = []
    for v, w in SPEC:
        a = pd.read_parquet(OUT_CV / "oof" / f"{v}_{f}_r10.parquet")
        b = pd.read_parquet(OUT_CV / "oof" / f"{v}_{f}_r17.parquet")
        parts.append(w * ((1 - W17) * a["pred_kwh"].to_numpy() + W17 * b["pred_kwh"].to_numpy()))
    a["pred"] = np.sum(parts, axis=0) / sum(w for _, w in SPEC)
    g = a.assign(g=a["site_id"].map(comp)).groupby(["g", "ts"])[["pred", "energy"]].sum()
    g["w"] = PENALTY_RATE * g.index.get_level_values("ts").map(smp).to_numpy() * k_of(g.index.get_level_values("ts"))
    return g


def W(g, a):
    return float((g["w"] * (a * g["bl"] - g["energy"]).abs()).sum()) / 1e6


def best_a(g):
    grid = np.arange(0.90, 1.08, 0.002)
    return grid[int(np.argmin([W(g, a) for a in grid]))]


ss = {f: site_sum(f) for f in FOLDS}
rows = []
for name in VARIANTS:
    for bw in BLEND_W + (1.0,):
        cur = {}
        for f in FOLDS:
            g = ss[f].copy()
            gp = gpred[(name, f)].reindex(g.index)
            g["bl"] = np.where(gp.notna(), (1 - bw) * g["pred"] + bw * gp.fillna(0), g["pred"])
            cur[f] = g
        ab = {f: best_a(g) for f, g in cur.items()}
        rows.append(dict(model=name, w=bw, A_xa=W(cur["A"], ab["B"]), B_xa=W(cur["B"], ab["A"]),
                         A_in=W(cur["A"], ab["A"]), B_in=W(cur["B"], ab["B"]), a_A=ab["A"], a_B=ab["B"]))
res = pd.DataFrame(rows)
base = res[(res.model == "g") & (res.w == 0.3)].iloc[0]
res["dA_%"] = 100 * (res["A_xa"] / base["A_xa"] - 1)
res["dB_%"] = 100 * (res["B_xa"] / base["B_xa"] - 1)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
out = ROOT / "outputs" / "v17"
out.mkdir(parents=True, exist_ok=True)
res.to_csv(out / f"20_cv_group_prev{'_n' + str(N_FIX) if N_FIX else ''}.csv", index=False)
