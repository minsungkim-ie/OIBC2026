"""v17: v15에서 그룹 모델만 '하루 사이 예보 변화 피처(dD1_*) 추가 그룹 모델(gp_d)'로 바꾼다.

사용: python scripts/21_group_prev_final.py v17 [--w 0.3] [--alpha 0.979]
  forecast(v17) = α × ((1 − w) × cand_v12x / 0.9697 + w × gp_d 그룹 예측)   (v15와 같은 식, 그룹 부분만 교체)
  · 그룹 예측이 없는 시각(야간)은 cand_v12x / 0.9697 × α (v15와 같음)
  · dD1_c = 대상일 D 주간 평균 예보 − D−1 대상일 주간 평균 예보 (17시 라운드). D−1 예보는 D−2 17:00 발행 →
    D의 예측시점(D−1 17:00) 이전 (CLAUDE.md 규칙 4, 2026-11-13 완화). 평가 첫날의 D−1은 학습 구간 마지막 날 예보.
  · 트리 수 = 폴드 B 학습 구간(2021-06~2024-05) 마지막 60일 조기종료 best_iter (16_group_blend와 같은 규칙)
  · 재현 확인: 같은 코드로 흐름 피처 없는 그룹 모델(g)을 학습해 v14 제출의 그룹 예측과 비교한다.
검증(20_cv_group_prev.py, 트리 700 고정, v9 OOF 발전소 합, v4 구성, 교차 α): g w0.3 103.87/194.37M →
gp_d w0.3 103.69/192.66M (−0.18%/−0.88%).
"""
import shutil
import sys
import warnings

from _bootstrap import log
import numpy as np
import pandas as pd

from oibc.config import ROOT, ROUNDS
from oibc.cv import CACHE, CV_START
from oibc.features import build_frame
from oibc.group import attach_prev, day_agg, fit_group, group_frame, joint, predict_group_kwh, train_rows
from oibc.io import load_sites, load_smp_test, load_smp_train
from oibc.pipeline import station_tables

warnings.filterwarnings("ignore")
args = sys.argv[1:]
opts = {"--w": 0.3, "--alpha": 0.979}
for key in list(opts):
    if key in args:
        i = args.index(key)
        opts[key] = float(args[i + 1])
        del args[i:i + 2]
(dst,) = args
SUB = ROOT / "submissions"
D = SUB / dst
assert not D.exists(), f"{D} 이미 있음 (제출한 폴더는 수정하지 않는다)"
OUT = ROOT / "outputs" / dst
OUT.mkdir(parents=True, exist_ok=True)
SITE_SRC, SITE_ALPHA = "cand_v12x", 0.9697

sites = load_sites()
comp = pd.read_csv(SUB / "v15" / "composition.csv").set_index("site_id")["group"]
assert comp.equals(pd.read_csv(SUB / "v4" / "composition.csv").set_index("site_id")["group"])
group_of = {i: comp[s] for i, s in enumerate(sites["site_id"])}
smp_tr = load_smp_train().set_index("ts")["smp"]

G = {}
for r in ROUNDS:
    d = pd.read_parquet(CACHE / f"frame_r{r}.parquet")
    G[r] = group_frame(d[d["ts"] >= CV_START], group_of)
    del d
Gtr = joint(G[17], G[10])
smp = load_smp_test()
index = pd.DatetimeIndex(smp["ts"])
tabs_te = station_tables("test", index[0], index[-1], sites, log)
base_te = pd.MultiIndex.from_product([sites["site_id"], index], names=["site_id", "ts"]).to_frame(index=False)
Gte = {r: group_frame(build_frame(base_te, sites, *tabs_te[r]), group_of) for r in ROUNDS}
Gte = joint(Gte[17], Gte[10])
agg = day_agg(pd.concat([Gtr, Gte], ignore_index=True))       # 예보만 (발전 실적 없음)
Gtr, Gte = attach_prev(Gtr, agg), attach_prev(Gte, agg)
prev_all = [c for c in Gtr.columns if c.startswith(("D1_", "D2_", "D3m_", "dD1_"))]
keep = [c for c in prev_all if c.startswith("dD1_")]
drop_p = [c for c in prev_all if c not in keep]
assert Gte[keep].notna().all().all(), "평가 구간 흐름 피처 NaN"
log(f"학습 {len(Gtr):,}행 / 평가 {len(Gte):,}행, 흐름 피처 {keep}")

tB = pd.Timestamp("2024-06-01 00:00", tz="Asia/Seoul")
full_end = Gtr["ts"].max()
preds = {}
for name, drop in (("g", prev_all), ("gp_d", drop_p)):
    tr = Gtr.drop(columns=drop)
    n_iter = int(fit_group(train_rows(tr, CV_START, tB), smp_tr).best_iteration_)
    m = fit_group(train_rows(tr, CV_START, full_end), smp_tr, n_estimators=n_iter)
    preds[name] = predict_group_kwh(m, Gte.drop(columns=drop))
    log(f"{name}: 폴드 B best_iter {n_iter}, 평가 주간 합계 {preds[name].sum() / 1e6:,.2f} GWh")

# 재현 확인: g ↔ v14 제출의 그룹 예측
p14 = pd.read_parquet(ROOT / "outputs" / "v14" / "16_parts.parquet")
p14["ts"] = pd.to_datetime(p14["ts"]).dt.tz_convert("Asia/Seoul")
g14 = p14.set_index(["group", "ts"])["group_kwh"]
idx = preds["g"].index
a, b = preds["g"].to_numpy(), g14.reindex(idx).to_numpy()
log(f"재현: g vs v14 그룹 예측 상관 {np.corrcoef(a, b)[0, 1]:.5f}, 합계비 {a.sum() / b.sum():.4f}, "
    f"평균 |차이| {np.abs(a - b).mean():.2f} kWh")
c = preds["gp_d"].to_numpy()
log(f"gp_d vs g: 상관 {np.corrcoef(a, c)[0, 1]:.5f}, 합계비 {c.sum() / a.sum():.4f}")

f15 = pd.read_csv(SUB / "v15" / "forecast.csv")
fs = pd.read_csv(SUB / SITE_SRC / "forecast.csv")
assert (fs["ts"] == f15["ts"]).all() and (fs["group"] == f15["group"]).all()
tsk = pd.to_datetime(f15["ts"]).dt.tz_convert("Asia/Seoul")
base = fs["forecast_kwh"].to_numpy("float64") / SITE_ALPHA
gp = preds["gp_d"].reindex(pd.MultiIndex.from_arrays([f15["group"], tsk])).to_numpy()
w, al = opts["--w"], opts["--alpha"]
new = (al * np.where(np.isnan(gp), base, (1 - w) * base + w * np.nan_to_num(gp))).clip(min=0).round(3)
assert np.isfinite(new).all()
# v15 재구성 확인 (같은 식에 v14 그룹 예측을 넣으면 v15가 나와야 함)
g14v = g14.reindex(pd.MultiIndex.from_arrays([f15["group"], tsk])).to_numpy()
rec = (0.979 * np.where(np.isnan(gp), base, 0.7 * base + 0.3 * g14v)).clip(min=0).round(3)
log(f"v15 재구성 최대 오차 {np.abs(rec - f15['forecast_kwh'].to_numpy()).max():.4f} kWh")
D.mkdir(parents=True)
pd.DataFrame({"group": f15["group"], "ts": f15["ts"], "forecast_kwh": new}).to_csv(D / "forecast.csv", index=False)
shutil.copy(SUB / "v15" / "composition.csv", D / "composition.csv")
pd.DataFrame({"group": f15["group"], "ts": f15["ts"], "base_kwh": base, "g_kwh": preds["g"].reindex(
    pd.MultiIndex.from_arrays([f15["group"], tsk])).to_numpy(), "gp_d_kwh": gp}).to_parquet(OUT / "21_parts.parquet")
log(f"저장 {D}: 합계 {new.sum() / 1e6:,.3f} GWh (v15 {f15['forecast_kwh'].sum() / 1e6:,.3f}), "
    f"v15와 상관 {np.corrcoef(new, f15['forecast_kwh'])[0, 1]:.5f}, w {w}, α {al}")
