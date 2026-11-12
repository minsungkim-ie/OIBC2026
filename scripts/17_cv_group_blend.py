"""2폴드 검증: 발전소 합 앙상블(OOF) × (1−w) + 그룹 단위 직접 모델(oibc.group) × w 를 v4 구성 W로 비교.

사용: python scripts/17_cv_group_blend.py <앙상블 이름=변형:가중,변형:가중,...> [...]
  예) v12=v2_wa:.125,csi_wa:.125,v2e_wa:.125,csie_wa:.125,v2e_c_wa:.25,csie_c_wa:.25

- 그룹 모델은 폴드마다 CV_START ~ 학습 구간 끝으로 학습(마지막 60일 조기종료), 검증 연도 주간 행을 예측한다
  (16_group_blend의 최종 학습과 같은 규칙). 야간·그룹 예측 없는 시각은 발전소 합 그대로.
- 발전소 합은 OOF(11_cv_train)의 라운드 가중 (1−W17)·r10 + W17·r17 후 변형 가중 평균.
- 배율 α: 같은 폴드 최적(in) / 다른 폴드 최적(xa, 비교 기준). 전체 발전소, [이력](폴드 시작 시점 1년 이력) 둘 다.
"""
import sys

from _bootstrap import log
import numpy as np
import pandas as pd

from oibc.config import PENALTY_RATE, ROOT, ROUNDS
from oibc.cv import CACHE, CV_START, FOLDS, OUT_CV, W17, site_efficiency
from oibc.group import fit_group, group_frame, joint, predict_group_kwh
from oibc.io import k_of, load_sites, load_smp_train

ens = {}
for a in sys.argv[1:]:
    name, spec = a.split("=", 1)
    ens[name] = [(v, float(w)) for v, w in (x.split(":") for x in spec.split(","))]
BLEND_W = (0.0, 0.3, 0.4)

sites = load_sites()
comp = pd.read_csv(ROOT / "submissions" / "v4" / "composition.csv").set_index("site_id")["group"]
group_of = {i: comp[s] for i, s in enumerate(sites["site_id"])}
smp = load_smp_train().set_index("ts")["smp"]
OOF = OUT_CV / "oof"

G = {}
for r in ROUNDS:
    d = pd.read_parquet(CACHE / f"frame_r{r}.parquet")
    d = d[d["ts"] >= CV_START]
    if r == 17:
        hist = {f: set(sites["site_id"].to_numpy()[site_efficiency(d, FOLDS[f][1]).index.to_numpy()]) for f in FOLDS}
    G[r] = group_frame(d, group_of)
    del d
Gall = joint(G[17], G[10])
del G
log(f"그룹 프레임 {len(Gall):,}행")

gpred = {}
for f, (t0, t1, v0, v1) in FOLDS.items():
    from oibc.group import train_rows
    m = fit_group(train_rows(Gall, CV_START, t1), smp)
    gv = Gall[(Gall["ts"] >= v0) & (Gall["ts"] <= v1)]
    gpred[f] = predict_group_kwh(m, gv)
    log(f"폴드 {f}: 그룹 모델 best_iter {m.best_iteration_}, 검증 주간 {len(gv):,}행")


def site_sum(spec, f, keep_sites=None):
    parts, ws = [], []
    for v, w in spec:
        a = pd.read_parquet(OOF / f"{v}_{f}_r10.parquet")
        b = pd.read_parquet(OOF / f"{v}_{f}_r17.parquet")
        parts.append((1 - W17) * a["pred_kwh"].to_numpy() + W17 * b["pred_kwh"].to_numpy())
        ws.append(w)
    a["pred"] = np.average(parts, axis=0, weights=ws)
    if keep_sites is not None:
        a = a[a["site_id"].isin(keep_sites)]
    g = a.assign(g=a["site_id"].map(comp)).groupby(["g", "ts"])[["pred", "energy"]].sum()
    g["w"] = PENALTY_RATE * g.index.get_level_values("ts").map(smp).to_numpy() * k_of(g.index.get_level_values("ts"))
    return g


def W(g, a):
    return float((g["w"] * (a * g["bl"] - g["energy"]).abs()).sum()) / 1e6


def best_a(g):
    grid = np.arange(0.94, 1.04, 0.002)
    return grid[int(np.argmin([W(g, a) for a in grid]))]


rows = []
for name, spec in ens.items():
    for scope in ("all", "hist"):
        gs = {}
        for f in FOLDS:
            g = site_sum(spec, f, hist[f] if scope == "hist" else None)
            gp = gpred[f].reindex(g.index)
            if scope == "hist":
                # [이력]은 발전소 일부만 채점하므로 그룹 모델 예측을 그 발전소 몫(용량 비)으로 줄인다
                cap = sites.set_index("site_id")["capacity_kw"]
                share = {gg: cap[[s for s in hist[f] if comp[s] == gg]].sum() / cap[comp == gg].sum()
                         for gg in ("G1", "G2")}
                gp = gp * np.array([share[x] for x in g.index.get_level_values("g")])
            gs[f] = (g, gp)
        for bw in BLEND_W:
            rec = dict(ens=name, scope=scope, w=bw)
            cur = {}
            for f, (g, gp) in gs.items():
                g = g.copy()
                g["bl"] = np.where(gp.notna(), (1 - bw) * g["pred"] + bw * gp.fillna(0), g["pred"])
                cur[f] = g
            ab = {f: best_a(g) for f, g in cur.items()}
            for f, o in (("A", "B"), ("B", "A")):
                rec[f"W_{f}_in"] = W(cur[f], ab[f])
                rec[f"W_{f}_xa"] = W(cur[f], ab[o])
                rec[f"a_{f}"] = ab[f]
            rows.append(rec)
    log(f"{name} 채점 완료")

res = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(res.round(3).to_string(index=False))
base = res[(res["ens"] == list(ens)[0]) & (res["w"] == 0.3)].set_index("scope")
print(f"\n기준: {list(ens)[0]} w=0.3 (v14 방식), 교차 α")
for _, r in res.iterrows():
    b = base.loc[r["scope"]]
    print(f"  {r['ens']:8s} {r['scope']:4s} w={r['w']:.1f}  A {100 * (r['W_A_xa'] / b['W_A_xa'] - 1):+.2f}%  "
          f"B {100 * (r['W_B_xa'] / b['W_B_xa'] - 1):+.2f}%")
res.to_csv(OUT_CV / "17_cv_group_blend.csv", index=False)
