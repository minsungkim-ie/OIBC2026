"""FINAL_TRAIN으로 재학습 → 평가 구간 예측 → 그룹 합산 → 제출 파일 생성."""
import gc
import json

import numpy as np
import pandas as pd

from _bootstrap import log
from oibc.config import FINAL_TRAIN, MODEL_KIND, OUT, ROUNDS, SUB
from oibc.features import build_frame, clean_flags
from oibc.io import load_generation, load_sites, load_smp_test
from oibc.model import fit_lgbm, fit_rf, predict_kwh, sample_weight
from oibc.pipeline import station_tables

SUB.mkdir(parents=True, exist_ok=True)
sites = load_sites()
best_iter = {int(k): v for k, v in json.load(open(OUT / "best_iter.json")).items()}
alpha = json.load(open(OUT / "scale.json"))["alpha"]          # 01이 검증 OOF로 정한 배율
log(f"모델 {MODEL_KIND}, 예측 배율 α={alpha}")
comp = pd.read_csv(OUT / "composition_opt.csv")
assert len(comp) == 253 and set(comp["site_id"]) == set(sites["site_id"])
grp = comp.set_index("site_id")["group"]

smp = load_smp_test()
index = pd.DatetimeIndex(smp["ts"])
log(f"채점 시각 {len(index):,}h ({index[0]} ~ {index[-1]})")

log("학습 구간 예보 대체")
tabs_tr = station_tables("train", FINAL_TRAIN[0], FINAL_TRAIN[1], sites, log)
log("평가 구간 예보 대체 (각 날짜는 그 날짜 예보만 사용, 통계 fit 없음)")
tabs_te = station_tables("test", index[0], index[-1], sites, log)
gen = load_generation(*FINAL_TRAIN)
base_te = pd.MultiIndex.from_product([sites["site_id"], index], names=["site_id", "ts"]).to_frame(index=False)

per_round = []
for r in ROUNDS:
    log(f"=== round {r} (n_estimators={best_iter[r]}) ===")
    d = build_frame(gen[["site_id", "ts", "energy"]], sites, *tabs_tr[r])
    keep, cnt = clean_flags(d)
    log(f"정제 제외: 음수 {cnt['neg']:,} / cf>1.2 {cnt['cf_gt_1_2']:,} / 종일0 행 {cnt['allzero_day_rows']:,}")
    sel = keep & (d["elev"] > 0)
    y = (d["energy"] / d["capacity_kw"]).astype("float32")
    if MODEL_KIND == "rf":
        m = fit_rf(d[sel], y[sel], sample_weight(d)[sel])
    else:
        m = fit_lgbm(d[sel], y[sel], sample_weight(d)[sel], n_estimators=best_iter[r])
    del d, y
    gc.collect()
    dt = build_frame(base_te, sites, *tabs_te[r])
    out = pd.DataFrame({"site_id": base_te["site_id"].to_numpy(), "ts": base_te["ts"].to_numpy(),
                        "pred_kwh": predict_kwh(m, dt, scale=alpha)})
    log(f"round {r} 예측 합계 {out['pred_kwh'].sum() / 1e6:,.2f} GWh")
    per_round.append(out)
    del m, dt
    gc.collect()

ps = per_round[0].copy()
assert (ps["site_id"].to_numpy() == per_round[1]["site_id"].to_numpy()).all()
ps["pred_kwh"] = np.mean([p["pred_kwh"].to_numpy() for p in per_round], axis=0)
ps.to_parquet(OUT / "test_site_pred.parquet", index=False)

ps["group"] = ps["site_id"].map(grp)
f = ps.groupby(["group", "ts"], as_index=False)["pred_kwh"].sum().rename(columns={"pred_kwh": "forecast_kwh"})
f = f.sort_values(["group", "ts"]).reset_index(drop=True)
f["forecast_kwh"] = f["forecast_kwh"].clip(lower=0).round(3)
f["ts"] = pd.DatetimeIndex(f["ts"]).tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S") + "+09:00"
f[["group", "ts", "forecast_kwh"]].to_csv(SUB / "forecast.csv", index=False)
comp[["site_id", "group"]].to_csv(SUB / "composition.csv", index=False)
log(f"저장: {SUB / 'composition.csv'} ({len(comp)}행), {SUB / 'forecast.csv'} ({len(f):,}행)")
