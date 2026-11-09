"""2폴드 검증 채점: 라운드 결합·배율·연도 간 보정·앙상블을 참조 구성 W로 비교 → outputs/v4/12_cv_report.md

배율·보정값은 항상 "다른 폴드"에서 정해 이 폴드에 적용한다(같은 해로 정하고 같은 해로 채점하지 않음).
"""
from _bootstrap import log
import numpy as np
import pandas as pd

from oibc.config import PENALTY_RATE
from oibc.cv import FOLDS, OUT_CV, VARIANTS
from oibc.io import k_of, load_sites, load_smp_train

sites = load_sites()
ref = sites.set_index("site_id")["current_group"].map({"A": "G1", "B": "G2"})
smp = load_smp_train().set_index("ts")["smp"]
OOF = OUT_CV / "oof"
avail = [n for n in VARIANTS if all((OOF / f"{n}_{f}_r{r}.parquet").exists() for f in FOLDS for r in (10, 17))]
log(f"채점 대상 변형: {avail}")


def load(name, fold):
    a = pd.read_parquet(OOF / f"{name}_{fold}_r10.parquet")
    b = pd.read_parquet(OOF / f"{name}_{fold}_r17.parquet")
    assert (a["site_id"].to_numpy() == b["site_id"].to_numpy()).all() and (a["ts"] == b["ts"]).all()
    return a.rename(columns={"pred_kwh": "p10"}).assign(p17=b["pred_kwh"].to_numpy())


def group_series(df, pred):
    """그룹(참조 구성)·시각별 예측/실적 합. 실적이 있는 (발전소, 시각)만."""
    g = df.assign(group=df["site_id"].map(ref), pred=pred).groupby(["group", "ts"])[["pred", "energy"]].sum()
    g = g.reset_index()
    g["w"] = PENALTY_RATE * g["ts"].map(smp).to_numpy() * k_of(g["ts"])
    return g


def W(g, scale=1.0):
    return float((g["w"] * (scale * g["pred"] - g["energy"]).abs()).sum())


def best_scale(g, by=None):
    """가중 L1 최적 배율 = (실적/예측)의 가중 중앙값, 가중치 w·pred. by가 있으면 그 키별로."""
    def wmed(x):
        m = x["pred"] > 1e-6
        r = (x.loc[m, "energy"] / x.loc[m, "pred"]).to_numpy()
        w = (x.loc[m, "w"] * x.loc[m, "pred"]).to_numpy()
        o = np.argsort(r)
        c = np.cumsum(w[o])
        return float(r[o][np.searchsorted(c, c[-1] / 2)]) if len(r) else 1.0
    if by is None:
        return wmed(g)
    return g.groupby(by(g)).apply(wmed)


hour = lambda g: g["ts"].dt.hour
season_hour = lambda g: ((g["ts"].dt.month % 12) // 3) * 100 + g["ts"].dt.hour

data = {(n, f): load(n, f) for n in avail for f in FOLDS}
rows = []


def add(label, preds):
    """preds: {fold: (df, pred array)} → 폴드별 W, 편향, 그리고 다른 폴드 배율 적용 결과."""
    gs = {f: group_series(*preds[f]) for f in FOLDS}
    rec = dict(label=label)
    other = {"A": "B", "B": "A"}
    for f, g in gs.items():
        rec[f"W_{f}"] = W(g)
        rec[f"bias_{f}"] = g["pred"].sum() / g["energy"].sum() - 1
        rec[f"a_best_{f}"] = best_scale(g)
    for f, g in gs.items():
        o = gs[other[f]]
        a = best_scale(o)
        rec[f"W_{f}_xa"] = W(g, a)                       # 다른 해에서 정한 전체 배율
        ah = best_scale(o, hour)
        rec[f"W_{f}_xah"] = W(g, g["ts"].dt.hour.map(ah).fillna(1.0).to_numpy())   # 시간대별 배율
    rows.append(rec)
    return gs


for n in avail:
    for wt in (0.0, 0.5, 0.7, 0.8, 1.0):
        add(f"{n} · w17={wt}", {f: (data[(n, f)], (1 - wt) * data[(n, f)]["p10"] + wt * data[(n, f)]["p17"])
                                for f in FOLDS})
    log(f"{n} 채점 완료")

# 앙상블 (라운드 w17=0.7 고정)
for combo in (("v2", "csi"), ("v1", "v2"), ("v1", "v2", "csi"), ("v2e", "csie"), ("v2", "csi", "v2e", "csie"),
              ("csi", "csie"), ("v2e", "csi")):
    if all(c in avail for c in combo):
        add(" + ".join(combo) + " · w17=0.7",
            {f: (data[(combo[0], f)],
                 np.mean([0.3 * data[(c, f)]["p10"] + 0.7 * data[(c, f)]["p17"] for c in combo], axis=0))
             for f in FOLDS})

res = pd.DataFrame(rows)
base = res.loc[res["label"] == "v2 · w17=0.5"].iloc[0] if "v2" in avail else res.iloc[0]
md = ["# 12 2폴드 검증 리포트 (참조 구성, 배율 1.0 기준)", "",
      "폴드 A: 2021-06~2023-05 학습 → 2023-06~2024-05 검증 / 폴드 B: 2022-06~2024-05 학습 → 2024-06~2025-05 검증",
      "", "W 단위: 백만 원. Δ는 v2 방식(w17=0.5, 배율 없음) 대비. xa = 다른 폴드에서 정한 전체 배율 적용, "
      "xah = 다른 폴드에서 정한 시간대별 배율 적용.", "",
      "| 변형 | W_A | W_B | ΔA | ΔB | 편향 A | 편향 B | 최적배율 A/B | W_A xa | W_B xa | W_A xah | W_B xah |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for _, r in res.iterrows():
    md.append(f"| {r['label']} | {r.W_A/1e6:,.2f} | {r.W_B/1e6:,.2f} | {(r.W_A/base.W_A-1)*100:+.2f}% | "
              f"{(r.W_B/base.W_B-1)*100:+.2f}% | {r.bias_A*100:+.1f}% | {r.bias_B*100:+.1f}% | "
              f"{r.a_best_A:.3f}/{r.a_best_B:.3f} | {r.W_A_xa/1e6:,.2f} | {r.W_B_xa/1e6:,.2f} | "
              f"{r.W_A_xah/1e6:,.2f} | {r.W_B_xah/1e6:,.2f} |")
(OUT_CV / "12_cv_report.md").write_text("\n".join(md), encoding="utf-8")
res.to_csv(OUT_CV / "12_cv_results.csv", index=False)
print("\n".join(md))
