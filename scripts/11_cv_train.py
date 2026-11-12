"""2폴드 검증 학습: 변형 × 폴드 × 라운드마다 학습 → 검증 연도 발전소별 예측(배율 1.0) 저장.

사용: python scripts/11_cv_train.py v1 v2 csi   (인자 없으면 전부)
     python scripts/11_cv_train.py v2y csiy v2m csim   (v8 후보: 시점 기준 효율 피처)

효율 피처 변형
- eff=True(v2e·csie, v7): 학습 구간 마지막 1년으로 잰 한 값을 학습·조기종료·검증 행 전부에 붙인다.
- eff="year"/"month"(v2y·csiy·v2m·csim): 학습·조기종료 행은 시점 기준(블록 첫 대상일 전일 17시까지의 직전 1년),
  검증 행은 폴드 시작 시점 값으로 고정한다(평가 연도와 같은 조건). 이력은 발전량 전체(2021-01-19~)에서 잰다.
"""
import gc
import json
import sys

from _bootstrap import log
import numpy as np
import pandas as pd

from oibc.config import ROUNDS
from oibc.cv import (CACHE, EFF_FEATS, ES_DAYS, FOLDS, OUT_CV, PIT_MODES, VARIANTS, _eff_prep, attach_eff,
                     eff_source, predict_cf, target_and_weight, train_start,
                     decay_mult)
from oibc.io import load_generation, load_sites
from oibc.model import fit_model, importance_top

names = sys.argv[1:] or list(VARIANTS)
sites = load_sites()
OOF = OUT_CV / "oof"
OOF.mkdir(parents=True, exist_ok=True)
unknown = [n for n in names if n not in VARIANTS]
assert not unknown, f"모르는 변형: {unknown}"

eff_x = None
if any(VARIANTS[n].get("eff") in PIT_MODES for n in names):
    eff_x = _eff_prep(eff_source(load_generation(), sites))
    log(f"시점 기준 효율용 이력 {len(eff_x):,}행 ({eff_x['ts'].min()} ~ {eff_x['ts'].max()})")


for r in ROUNDS:
    d = pd.read_parquet(CACHE / f"frame_r{r}.parquet")
    d["site_id"] = d["site_id"].astype("int32")
    if any(VARIANTS[n].get("extra") for n in names):
        x = pd.read_parquet(CACHE / f"extra_r{r}.parquet")    # scripts/10b_cv_extra.py, frame과 같은 행 순서
        assert len(x) == len(d)
        for c in x.columns:
            d[c] = x[c].to_numpy()
        del x
    log(f"round {r}: 캐시 {len(d):,}행 로드")
    for name in names:
        feats, target = VARIANTS[name]["feats"], VARIANTS[name]["target"]
        for fold, (t0, t1, v0, v1) in FOLDS.items():
            out = OOF / f"{name}_{fold}_r{r}.parquet"
            if out.exists():
                log(f"  {out.name} 이미 있음 → 건너뜀")
                continue
            if VARIANTS[name].get("eff"):
                attach_eff(d, name, eff_x=eff_x, fit_end=t1, freeze_at=v0, log=log)
                vm = (d["ts"] >= v0) & (d["ts"] <= v1)
                ev = d.loc[vm].groupby("site_id")["eff_mid"].first()
                log(f"  {name} fold {fold}: 검증 행 효율 피처 {int(ev.notna().sum())}곳 "
                    f"(eff_mid 중앙값 {ev.median():.3f}, 5~95% {ev.quantile(.05):.3f}~{ev.quantile(.95):.3f})")
            es_cut = t1 - pd.Timedelta(days=ES_DAYS)
            base = d["keep"] & (d["elev"] > 0)
            tr = base & (d["ts"] >= train_start(name, t0, t1)) & (d["ts"] <= es_cut)
            es = base & (d["ts"] > es_cut) & (d["ts"] <= t1)
            y, w = target_and_weight(d, target)
            w = w * decay_mult(name, d["ts"], t1)
            algo = VARIANTS[name].get("algo", "lgb")
            m = fit_model(algo, d[tr], y[tr.to_numpy()], w[tr.to_numpy()], d[es], y[es.to_numpy()],
                          w[es.to_numpy()], feats=feats, seed=VARIANTS[name].get("seed"),
                          lgbp=VARIANTS[name].get("lgbp"))
            va = (d["ts"] >= v0) & (d["ts"] <= v1)
            dv = d[va]
            pred = predict_cf(m, dv, feats, target) * dv["capacity_kw"].to_numpy("float64")
            sid = sites["site_id"].to_numpy()[dv["site_id"].to_numpy()]
            pd.DataFrame({"site_id": sid, "ts": dv["ts"].to_numpy(), "energy": dv["energy"].to_numpy(),
                          "pred_kwh": pred}).to_parquet(out, index=False)
            info = dict(best_iter=int(m.best_iteration_), n_train=int(tr.sum()), n_es=int(es.sum()))
            json.dump(info, open(OOF / f"{name}_{fold}_r{r}.json", "w"))
            if algo == "lgb":
                importance_top(m, feats).to_csv(OOF / f"{name}_{fold}_r{r}_imp.csv", index=False)
            log(f"  {name} fold {fold} r{r}: {info}, 검증 {len(dv):,}행")
            del m, dv
            gc.collect()
    del d
    gc.collect()
log("완료")
