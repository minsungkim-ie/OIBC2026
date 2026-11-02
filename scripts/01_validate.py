"""시간순 검증: VAL_TRAIN으로 학습 → VALID 예측, OOF 잔차 저장, 리포트."""
import gc
import json

import lightgbm as lgb
from _bootstrap import log
import numpy as np
import pandas as pd

from oibc.config import (BASE_FEATURES, BASE_PARAMS, FEATURES, OUT, ROUNDS, VALID, VAL_ES_DAYS,
                         VAL_TRAIN)
from oibc.features import build_frame, clean_flags
from oibc.io import k_of, load_generation, load_sites, load_smp_test, load_smp_train
from oibc.model import fit_lgbm, importance_top, predict_kwh, sample_weight
from oibc.pipeline import station_tables
from oibc.scoring import breakdown, group_frame, summarize

OUT.mkdir(parents=True, exist_ok=True)
sites = load_sites()
ref_comp = sites.set_index("site_id")["current_group"].map({"A": "G1", "B": "G2"})
cap_by_group = sites.assign(g=sites["current_group"].map({"A": "G1", "B": "G2"})).groupby("g")["capacity_kw"].sum()

# k 규칙 대조: 평가 구간 smp.csv의 k 열과 같은 규칙인지
smp_te = load_smp_test()
assert (smp_te["k"].to_numpy() == k_of(smp_te["ts"])).all(), "k 규칙 불일치"
log("k 규칙(시 11~15 → 3) smp.csv와 일치 확인")

smp_tr = load_smp_train()
smp = smp_tr.set_index("ts")["smp"]

start, end = VAL_TRAIN[0], VALID[1]
log("예보 대체·지점 테이블 생성 (학습 구간)")
tabs = station_tables("train", start, end, sites, log)
gen = load_generation(start, end)
log(f"발전량 {len(gen):,}행 {gen.ts.min()} ~ {gen.ts.max()}")

es_cut = VAL_TRAIN[1] - pd.Timedelta(days=VAL_ES_DAYS)
best_iter, preds = {}, {}
for r in ROUNDS:
    log(f"=== round {r} ===")
    d = build_frame(gen[["site_id", "ts", "energy"]], sites, *tabs[r])
    keep, cnt = clean_flags(d)
    log(f"정제 제외(전체 구간 기준): 음수 {cnt['neg']:,} / cf>1.2 {cnt['cf_gt_1_2']:,} / "
        f"종일0 발전소-일 행 {cnt['allzero_day_rows']:,} (총 {cnt['total']:,}행)")
    day = d["elev"] > 0
    in_tr = (d["ts"] >= VAL_TRAIN[0]) & (d["ts"] <= VAL_TRAIN[1])
    tr = in_tr & keep & day & (d["ts"] <= es_cut)
    es = in_tr & keep & day & (d["ts"] > es_cut)
    y = (d["energy"] / d["capacity_kw"]).astype("float32")
    w = sample_weight(d)
    log(f"학습 {int(tr.sum()):,}행 / 조기종료 {int(es.sum()):,}행")
    m = fit_lgbm(d[tr], y[tr], w[tr], d[es], y[es], w[es])
    best_iter[r] = int(m.best_iteration_)
    log(f"best_iteration={best_iter[r]}")
    importance_top(m).to_csv(OUT / f"feature_importance_r{r}.csv", index=False)

    va = (d["ts"] >= VALID[0]) & (d["ts"] <= VALID[1])
    dv = d[va]
    preds[("v1", r)] = pd.Series(predict_kwh(m, dv), index=dv.index)
    preds[("raw", r)] = pd.Series(predict_kwh(m, dv, scale=1.0), index=dv.index)   # 보정 전
    del m

    # 비교군: baseline.py 방식 근사 (l2, 동일 정제 학습 표본, VAL_TRAIN 전체)
    trb = in_tr & keep & day
    mb = lgb.LGBMRegressor(**BASE_PARAMS)
    yb = y[trb].clip(0, 1.2)
    mb.fit(d.loc[trb, BASE_FEATURES], yb)
    cfb = np.zeros(len(dv))
    dd = (dv["elev"] > 0).to_numpy()
    cfb[dd] = mb.predict(dv.loc[dd, BASE_FEATURES])
    preds[("base", r)] = pd.Series(np.clip(cfb, 0, None) * dv["capacity_kw"].to_numpy(), index=dv.index)
    log(f"baseline 근사 학습·예측 완료 (round {r})")
    meta = dv[["ts"]].copy()
    meta["site_id"] = gen.loc[dv.index, "site_id"].to_numpy()
    meta["energy"] = gen.loc[dv.index, "energy"].to_numpy()
    del d, dv, mb
    gc.collect()

json.dump({str(r): best_iter[r] for r in ROUNDS}, open(OUT / "best_iter.json", "w"))

oof = meta.copy()
for r in ROUNDS:
    oof[f"pred_v1_r{r}"] = preds[("v1", r)].reindex(meta.index).to_numpy()
    oof[f"pred_base_r{r}"] = preds[("base", r)].reindex(meta.index).to_numpy()
    oof[f"pred_raw_r{r}"] = preds[("raw", r)].reindex(meta.index).to_numpy()
oof["pred_kwh"] = oof[[f"pred_v1_r{r}" for r in ROUNDS]].mean(axis=1)
oof["pred_raw"] = oof[[f"pred_raw_r{r}" for r in ROUNDS]].mean(axis=1)
oof["pred_base"] = oof[[f"pred_base_r{r}" for r in ROUNDS]].mean(axis=1)
oof["resid"] = oof["pred_kwh"] - oof["energy"]
oof[["site_id", "ts", "energy", "pred_kwh", "pred_base", "resid"]
    + [f"pred_v1_r{r}" for r in ROUNDS]].to_parquet(OUT / "oof_valid.parquet", index=False)
log(f"oof_valid.parquet 저장 ({len(oof):,}행)")

# 채점
rows, brk = [], {}
for name, col in (("baseline 근사 (참조 구성)", "pred_base"), ("v2 모델 보정 전 (참조 구성)", "pred_raw"),
                  ("v2 모델 (참조 구성)", "pred_kwh")):
    g = group_frame(oof.assign(pred_kwh=oof[col]), ref_comp, smp)
    s = summarize(g, cap_by_group)
    rows.append((name, s))
    brk[name] = breakdown(g)
for r in ROUNDS:
    g = group_frame(oof.assign(pred_kwh=oof[f"pred_v1_r{r}"]), ref_comp, smp)
    rows.append((f"v2 단일 round {r} (참조 구성)", summarize(g, cap_by_group)))

md = ["# 01 검증 리포트 (VALID 2024-06-01 01:00 ~ 2025-06-01 00:00)", "",
      f"best_iteration: {best_iter}", "",
      "| 비교 | S | 매전수익 | W | W_LB | 절대오차 합(kWh) | nMAE(주간) |", "|---|---|---|---|---|---|---|"]
for n, s in rows:
    md.append(f"| {n} | {s['S']:,.0f} | {s['rev']:,.0f} | {s['W']:,.0f} | {s['W_LB']:,.0f} | "
              f"{s['abs_err']:,.0f} | {s['nMAE_day']:.4f} |")
mo, hr = brk["v2 모델 (참조 구성)"]
mob, hrb = brk["baseline 근사 (참조 구성)"]
md += ["", "## 월별 W (참조 구성)", "", "| 월 | baseline 근사 | v2 |", "|---|---|---|"]
md += [f"| {k} | {mob[k]:,.0f} | {mo[k]:,.0f} |" for k in mo.index]
md += ["", "## 시간대별 W (참조 구성)", "", "| 시 | baseline 근사 | v2 |", "|---|---|---|"]
md += [f"| {k} | {hrb[k]:,.0f} | {hr[k]:,.0f} |" for k in hr.index]
(OUT / "01_validation_report.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md[:12]))
log("완료")
