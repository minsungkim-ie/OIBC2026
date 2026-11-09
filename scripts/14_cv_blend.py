"""v9: 알고리즘 계열별 혼합 가중치 탐색 (2폴드, [이력] 조건, 참조 구성).

사용: python scripts/14_cv_blend.py lgb=v2,csi,v2e,csie xgb=v2e_x,csie_x cat=v2e_c,csie_c
  계열 = 변형들의 단순 평균(각 변형은 라운드 0.3·r10 + 0.7·r17). 계열끼리 0.1 간격 단체(simplex) 격자로 섞는다.
v11: python scripts/14_cv_blend.py lgb=v2,csi,v2e,csie cat=v2e_c,csie_c mlp=csie_m --base 0.5,0.5 --add 0,0.05,0.1,0.15 --out v11
  --add: 마지막 계열의 비중 w 목록. 나머지 계열은 --base 비율 × (1 − w). 격자 대신 이 행들만 채점한다
  --out: 결과 폴더 (outputs/<out>, 기본 v9)

채점은 12_cv_eval과 같은 정의: 폴드 시작 시점에 이력 있는 발전소만, 참조 구성 G1/G2 합,
W = Σ 0.1·SMP·k·|α·예측 − 실적|, α는 항상 다른 폴드에서 정한다(가중 L1 최적 배율).
가중치도 교차로 고른다: 폴드 A에서 고른 가중치를 B에서 채점, 반대도. 결과 → outputs/v9/14_blend.csv·.md
"""
import itertools
import sys

from _bootstrap import log
import numpy as np
import pandas as pd

from oibc.config import PENALTY_RATE, ROOT
from oibc.cv import FOLDS, OUT_CV, W17, eff_source, site_efficiency
from oibc.io import k_of, load_generation, load_sites, load_smp_train

args = sys.argv[1:]
opts = {}
for key in ("--base", "--add", "--out"):
    if key in args:
        i = args.index(key)
        opts[key] = args[i + 1]
        del args[i:i + 2]
OUT = ROOT / "outputs" / opts.get("--out", "v9")
OUT.mkdir(parents=True, exist_ok=True)
OOF = OUT_CV / "oof"
fams = {}
for a in args:
    k, v = a.split("=")
    fams[k] = v.split(",")
assert fams, "계열을 하나 이상 지정"
names = list(fams)

sites = load_sites()
ref = sites.set_index("site_id")["current_group"].map({"A": "G1", "B": "G2"})
smp = load_smp_train().set_index("ts")["smp"]
_src = eff_source(load_generation(), sites)
EST = {}
for f, (_, t1, _, _) in FOLDS.items():
    e = site_efficiency(_src, t1)
    EST[f] = set(sites["site_id"].to_numpy()[e.index[e["eff_mid"].notna()].to_numpy()])
del _src


def fam_group(fold):
    """계열별 (그룹, 시각) 예측 합 행렬 P[계열, 행], 실적 E[행], 가중 w[행]."""
    P, base = [], None
    for n in names:
        acc = None
        for v in fams[n]:
            a = pd.read_parquet(OOF / f"{v}_{fold}_r10.parquet")
            b = pd.read_parquet(OOF / f"{v}_{fold}_r17.parquet")
            p = (1 - W17) * a["pred_kwh"].to_numpy() + W17 * b["pred_kwh"].to_numpy()
            acc = p if acc is None else acc + p
            if base is None:
                base = a[["site_id", "ts", "energy"]]
        df = base.assign(pred=acc / len(fams[n]))
        df = df[df["site_id"].isin(EST[fold]) & df["energy"].notna()]
        g = df.assign(group=df["site_id"].map(ref)).groupby(["group", "ts"])[["pred", "energy"]].sum().reset_index()
        P.append(g["pred"].to_numpy())
    w = PENALTY_RATE * g["ts"].map(smp).to_numpy() * k_of(g["ts"])
    REV[fold] = float((g["ts"].map(smp).to_numpy() * g["energy"].to_numpy()).sum())
    return np.vstack(P), g["energy"].to_numpy(), w


def best_scale(p, e, w):
    m = p > 1e-6
    r, ww = e[m] / p[m], (w * p)[m]
    o = np.argsort(r)
    c = np.cumsum(ww[o])
    return float(r[o][np.searchsorted(c, c[-1] / 2)])


REV = {}
D = {f: fam_group(f) for f in FOLDS}
log(f"계열 {fams}")
if "--add" in opts:
    b = np.array([float(x) for x in opts["--base"].split(",")])
    assert len(b) == len(names) - 1 and abs(b.sum() - 1) < 1e-9, f"--base {b}"
    grid = [np.append(b * (1 - a), a) for a in (float(x) for x in opts["--add"].split(","))]
else:
    grid = [np.array(c) / 10 for c in itertools.product(range(11), repeat=len(names)) if sum(c) == 10]
other = {"A": "B", "B": "A"}
rows = []
for wv in grid:
    rec = {n: wv[i] for i, n in enumerate(names)}
    blend = {f: wv @ D[f][0] for f in FOLDS}
    for f in FOLDS:
        a = best_scale(blend[other[f]], D[other[f]][1], D[other[f]][2])
        rec[f"W_{f}_xa"] = float((D[f][2] * np.abs(a * blend[f] - D[f][1])).sum())
        rec[f"S_{f}_xa"] = REV[f] - rec[f"W_{f}_xa"]
        rec[f"xa_{f}"] = a
        rec[f"a_{f}"] = best_scale(blend[f], D[f][1], D[f][2])
    rows.append(rec)
res = pd.DataFrame(rows)
res["sum_rel"] = res["W_A_xa"] / res["W_A_xa"].max() + res["W_B_xa"] / res["W_B_xa"].max()
res.to_csv(OUT / "14_blend.csv", index=False)

# 교차 선택: 한 폴드에서 최적인 가중치를 다른 폴드에서 채점
md = ["# 14 계열 혼합 가중치 ([이력] 조건, 참조 구성, 교차 α)", "", f"계열: {fams}", "",
      f"매전수익(실적 기준, [이력] 발전소): A {REV['A']:,.0f} / B {REV['B']:,.0f}. S = 매전수익 − W", ""]
if "--add" in opts:
    r0 = res.iloc[0]
    md += [f"| {' | '.join(names)} | W_A | ΔA | W_B | ΔB | S_A | S_B | 교차 α A/B | 최적 α A/B |",
           "|" + "---|" * (len(names) + 8)]
    for _, q in res.iterrows():
        md.append(f"| {' | '.join(f'{q[n]:.4f}' for n in names)} | {q.W_A_xa:,.0f} | {(q.W_A_xa / r0.W_A_xa - 1) * 100:+.2f}% "
                  f"| {q.W_B_xa:,.0f} | {(q.W_B_xa / r0.W_B_xa - 1) * 100:+.2f}% | {q.S_A_xa:,.0f} | {q.S_B_xa:,.0f} "
                  f"| {q.xa_A:.4f}/{q.xa_B:.4f} | {q.a_A:.4f}/{q.a_B:.4f} |")
    md.append("")
for n in names if "--add" not in opts else []:
    one = res[res[n] == 1.0].iloc[0]
    md.append(f"- {n} 단독: W_A {one.W_A_xa/1e6:,.2f}M / W_B {one.W_B_xa/1e6:,.2f}M")
if "--add" not in opts:
    eq = res.iloc[(res[names] - 1 / len(names)).abs().sum(axis=1).argmin()]
    md.append(f"- 균등에 가장 가까운 {dict(eq[names])}: W_A {eq.W_A_xa/1e6:,.2f}M / W_B {eq.W_B_xa/1e6:,.2f}M")
for f in FOLDS:
    b = res.loc[res[f"W_{f}_xa"].idxmin()]
    o = other[f]
    md.append(f"- 폴드 {f} 최적 {dict(b[names])} → 폴드 {o}에서 W {b[f'W_{o}_xa']/1e6:,.2f}M "
              f"(폴드 {o} 자체 최적 {res[f'W_{o}_xa'].min()/1e6:,.2f}M)")
b = res.loc[res["sum_rel"].idxmin()]
md.append(f"- 두 폴드 합 최적 {dict(b[names])}: W_A {b.W_A_xa/1e6:,.2f}M / W_B {b.W_B_xa/1e6:,.2f}M, "
          f"α A/B {b.a_A:.4f}/{b.a_B:.4f}")
md += ["", "상위 10 (두 폴드 상대합 기준):", "", "```", res.sort_values("sum_rel").head(10).to_string(index=False), "```"]
(OUT / "14_blend.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
