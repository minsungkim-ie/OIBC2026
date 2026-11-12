"""2폴드 검증: 그룹 모델 변형(oibc.group.GROUP_VARIANTS) 조합 × 블렌드 비중을 v4 구성 W로 비교.

사용: python scripts/18_cv_group_variants.py <발전소 합 앙상블 변형:가중,...>
  v16 = python scripts/18_cv_group_variants.py v2x_wa:.125,csix_wa:.125,v2ex_wa:.125,csiex_wa:.125,v2ex_c_wa:.25,csiex_c_wa:.25

- 그룹 모델: 폴드마다 CV_START ~ 학습 구간 끝(마지막 60일 조기종료)으로 학습, 검증 연도 주간 행 예측.
  aug 변형은 무작위 2분할 그룹 행을 학습에만 더한다 (16_group_blend와 같은 규칙). 예측·채점은 v4 구성 그룹만.
- 확장 예보 피처(extra=True 변형)는 캐시 extra_r{r}.parquet(10b_cv_extra)을 그룹 용량가중 평균으로 쓴다.
- 조합: 그룹 변형 평균 × 비중 w → (1−w)·발전소 합 + w·그룹 (야간·그룹 예측 없는 시각은 발전소 합).
- 배율: 다른 폴드 최적 α (xa)로 비교. 결과 outputs/v4/18_cv_group_variants.csv
"""
import sys

from _bootstrap import log
import numpy as np
import pandas as pd

from oibc.config import PENALTY_RATE, ROOT, ROUNDS
from oibc.cv import CACHE, CV_START, FOLDS, OUT_CV, W17
from oibc.group import GROUP_VARIANTS, aug_partitions, fit_group, group_frame, joint, predict_group_kwh, train_rows
from oibc.io import k_of, load_sites, load_smp_train

SITE = [(v, float(w)) for v, w in (x.split(":") for x in sys.argv[1].split(","))]
COMBOS = {
    "① g": ["g"],
    "② g4": ["g", "g_csi", "g_lr", "g_augcsi"],
    "③ gx": ["gx"],
    "④ gx4": ["gx", "gx_csi", "gx_lr", "gx_augcsi"],
}
BLEND_W = (0.3, 0.4, 0.5)
N_AUG = 6

sites = load_sites()
comp = pd.read_csv(ROOT / "submissions" / "v4" / "composition.csv").set_index("site_id")["group"]
group_of = {i: comp[s] for i, s in enumerate(sites["site_id"])}
smp = load_smp_train().set_index("ts")["smp"]
OOF = OUT_CV / "oof"

FR = {}
for r in ROUNDS:
    d = pd.read_parquet(CACHE / f"frame_r{r}.parquet")
    x = pd.read_parquet(CACHE / f"extra_r{r}.parquet")
    for c in x.columns:
        d[c] = x[c].to_numpy()
    FR[r] = d[d["ts"] >= CV_START].reset_index(drop=True)
    del d, x
Gall = joint(group_frame(FR[17], group_of, extra=True), group_frame(FR[10], group_of, extra=True))
Gaug = [joint(group_frame(FR[17], p, extra=True), group_frame(FR[10], p, extra=True)).assign(aug=k + 1)
        for k, p in enumerate(aug_partitions(len(sites), N_AUG))]
del FR
log(f"그룹 프레임 {len(Gall):,}행 × {Gall.shape[1]}열, 증강 분할 {N_AUG}개")

gpred = {}
for f, (t0, t1, v0, v1) in FOLDS.items():
    gv = Gall[(Gall["ts"] >= v0) & (Gall["ts"] <= v1)]
    for v in sorted({v for vs in COMBOS.values() for v in vs}):
        cfg = GROUP_VARIANTS[v]
        tr = pd.concat([train_rows(Gall, CV_START, t1)] + [train_rows(a, CV_START, t1) for a in Gaug[:cfg.get("aug", 0)]],
                       ignore_index=True)
        m = fit_group(tr, smp, target=cfg.get("target", "cf"), params=cfg.get("params"), extra=cfg.get("extra", False))
        gpred[(f, v)] = predict_group_kwh(m, gv)
        log(f"폴드 {f} {v}: best_iter {m.best_iteration_}, 피처 {len(m.group_feats)}개, 학습 {len(tr):,}행")


def site_sum(f):
    parts = []
    for v, _ in SITE:
        a = pd.read_parquet(OOF / f"{v}_{f}_r10.parquet")
        b = pd.read_parquet(OOF / f"{v}_{f}_r17.parquet")
        parts.append((1 - W17) * a["pred_kwh"].to_numpy() + W17 * b["pred_kwh"].to_numpy())
    a["pred"] = np.average(parts, axis=0, weights=[w for _, w in SITE])
    g = a.assign(g=a["site_id"].map(comp)).groupby(["g", "ts"])[["pred", "energy"]].sum()
    g["w"] = PENALTY_RATE * g.index.get_level_values("ts").map(smp).to_numpy() * k_of(g.index.get_level_values("ts"))
    return g


def W(g, bl, a):
    return float((g["w"] * (a * bl - g["energy"]).abs()).sum()) / 1e6


def best_a(g, bl):
    grid = np.arange(0.94, 1.04, 0.002)
    return grid[int(np.argmin([W(g, bl, a) for a in grid]))]


S = {f: site_sum(f) for f in FOLDS}
rows = []
for name, vs in COMBOS.items():
    for bw in BLEND_W:
        bl = {}
        for f, g in S.items():
            gp = pd.concat([gpred[(f, v)] for v in vs], axis=1).mean(axis=1).reindex(g.index)
            bl[f] = np.where(gp.notna(), (1 - bw) * g["pred"] + bw * gp.fillna(0), g["pred"])
        ab = {f: best_a(S[f], bl[f]) for f in FOLDS}
        rec = dict(combo=name, w=bw)
        for f, o in (("A", "B"), ("B", "A")):
            rec[f"W_{f}_xa"] = W(S[f], bl[f], ab[o])
            rec[f"a_{f}"] = ab[f]
        rows.append(rec)
res = pd.DataFrame(rows)
b = res[(res["combo"] == "① g") & (res["w"] == 0.3)].iloc[0]
res["dA"] = 100 * (res["W_A_xa"] / b["W_A_xa"] - 1)
res["dB"] = 100 * (res["W_B_xa"] / b["W_B_xa"] - 1)
pd.set_option("display.width", 200)
print("기준: ① g, w=0.3 (= v15 구조), 다른 폴드 최적 α")
print(res.round(3).to_string(index=False))
res.to_csv(OUT_CV / "18_cv_group_variants.csv", index=False)
