"""2폴드 검증 학습: 변형 × 폴드 × 라운드마다 학습 → 검증 연도 발전소별 예측(배율 1.0) 저장.

사용: python scripts/11_cv_train.py v1 v2 csi   (인자 없으면 전부)
"""
import gc
import json
import sys

from _bootstrap import log
import numpy as np
import pandas as pd

from oibc.config import ROUNDS
from oibc.cv import (CACHE, EFF_FEATS, ES_DAYS, FOLDS, OUT_CV, VARIANTS, predict_cf, site_efficiency,
                     target_and_weight)
from oibc.io import load_sites
from oibc.model import fit_lgbm, importance_top

names = sys.argv[1:] or list(VARIANTS)
sites = load_sites()
OOF = OUT_CV / "oof"
OOF.mkdir(parents=True, exist_ok=True)


for r in ROUNDS:
    d = pd.read_parquet(CACHE / f"frame_r{r}.parquet")
    d["site_id"] = d["site_id"].astype("int32")
    log(f"round {r}: 캐시 {len(d):,}행 로드")
    for name in names:
        feats, target = VARIANTS[name]["feats"], VARIANTS[name]["target"]
        for fold, (t0, t1, v0, v1) in FOLDS.items():
            out = OOF / f"{name}_{fold}_r{r}.parquet"
            if out.exists():
                log(f"  {out.name} 이미 있음 → 건너뜀")
                continue
            if VARIANTS[name].get("eff"):
                eff = site_efficiency(d, t1)
                for c in EFF_FEATS:
                    d[c] = d["site_id"].map(eff[c]).astype("float32")
                log(f"  {name} fold {fold}: 효율 피처 {eff['eff_mid'].notna().sum()}곳 "
                    f"(eff_mid 중앙값 {eff['eff_mid'].median():.3f}, 5~95% {eff['eff_mid'].quantile(.05):.3f}"
                    f"~{eff['eff_mid'].quantile(.95):.3f})")
            es_cut = t1 - pd.Timedelta(days=ES_DAYS)
            base = d["keep"] & (d["elev"] > 0)
            tr = base & (d["ts"] >= t0) & (d["ts"] <= es_cut)
            es = base & (d["ts"] > es_cut) & (d["ts"] <= t1)
            y, w = target_and_weight(d, target)
            m = fit_lgbm(d[tr], y[tr.to_numpy()], w[tr.to_numpy()], d[es], y[es.to_numpy()], w[es.to_numpy()],
                         feats=feats)
            va = (d["ts"] >= v0) & (d["ts"] <= v1)
            dv = d[va]
            pred = predict_cf(m, dv, feats, target) * dv["capacity_kw"].to_numpy("float64")
            sid = sites["site_id"].to_numpy()[dv["site_id"].to_numpy()]
            pd.DataFrame({"site_id": sid, "ts": dv["ts"].to_numpy(), "energy": dv["energy"].to_numpy(),
                          "pred_kwh": pred}).to_parquet(out, index=False)
            info = dict(best_iter=int(m.best_iteration_), n_train=int(tr.sum()), n_es=int(es.sum()))
            json.dump(info, open(OOF / f"{name}_{fold}_r{r}.json", "w"))
            importance_top(m, feats).to_csv(OOF / f"{name}_{fold}_r{r}_imp.csv", index=False)
            log(f"  {name} fold {fold} r{r}: {info}, 검증 {len(dv):,}행")
            del m, dv
            gc.collect()
    del d
    gc.collect()
log("완료")
