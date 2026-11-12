"""v14: 기존 제출(발전소 합 앙상블)에 그룹 단위 직접 모델(src/oibc/group.py)을 블렌드한다. 구성은 기존 그대로.

사용: python scripts/16_group_blend.py <기존 버전> <새 버전> --base-alpha A --w W --alpha B
  v14 = python scripts/16_group_blend.py v12 v14 --base-alpha 0.9717 --w 0.3 --alpha 0.981

  forecast(새) = B × ((1 − W) × forecast(기존) / A + W × 그룹 모델 예측)
  · 그룹 모델은 10_cv_build 캐시 프레임(2021-06~2025-05 학습 구간 전체)으로 학습. 트리 수 = 폴드 B 학습 구간
    (2021-06~2024-05)의 마지막 60일 조기종료 best_iter (13_final의 win="all" 변형과 같은 규칙)
  · 평가 구간 피처는 그날의 전일 10시·17시 예보만으로 만든다 (station_tables·build_frame과 같음)
  · 그룹 평균 태양고도가 0 이하인 시각(야간)은 기존 예측 그대로

검증(2폴드, v9 6개 OOF 발전소 합, v4 구성, 다른 해 최적 α): 105.43/198.10M → W 0.3 블렌드 103.78/194.21M
(−1.6%/−2.0%). 블렌드 최적 α / 발전소 합 최적 α = 0.9955(A)·1.0000(B). EXPERIMENTS.md 11-12 참조.
"""
import shutil
import sys

from _bootstrap import log
import numpy as np
import pandas as pd

from oibc.config import ROOT, ROUNDS
from oibc.cv import CACHE, CV_START
from oibc.features import build_frame
from oibc.group import fit_group, group_frame, joint, predict_group_kwh, train_rows
from oibc.io import load_sites, load_smp_test, load_smp_train
from oibc.pipeline import station_tables

args = sys.argv[1:]
opts = {}
for key in ("--base-alpha", "--w", "--alpha"):
    i = args.index(key)
    opts[key] = float(args[i + 1])
    del args[i:i + 2]
src, dst = args
S, D = ROOT / "submissions" / src, ROOT / "submissions" / dst
assert not D.exists(), f"{D} 이미 있음 (제출한 폴더는 수정하지 않는다)"
OUT = ROOT / "outputs" / dst
OUT.mkdir(parents=True, exist_ok=True)

sites = load_sites()
comp = pd.read_csv(S / "composition.csv").set_index("site_id")["group"]
group_of = {i: comp[s] for i, s in enumerate(sites["site_id"])}       # build_frame의 정수 site_id 코드
smp_tr = load_smp_train().set_index("ts")["smp"]

# 1. 학습 프레임 (학습 구간 전체 4년, 10_cv_build 캐시)
G = {}
for r in ROUNDS:
    d = pd.read_parquet(CACHE / f"frame_r{r}.parquet")
    d = d[d["ts"] >= CV_START]
    G[r] = group_frame(d, group_of)
    del d
Gtr = joint(G[17], G[10])
log(f"그룹 학습 프레임 {len(Gtr):,}행 ({Gtr['ts'].min()} ~ {Gtr['ts'].max()}), 피처 {Gtr.shape[1]}열")

# 2. 트리 수: 폴드 B 학습 구간(2021-06 ~ 2024-05)에서 조기종료
tB = pd.Timestamp("2024-06-01 00:00", tz="Asia/Seoul")
mB = fit_group(train_rows(Gtr, CV_START, tB), smp_tr)
n_iter = int(mB.best_iteration_)
log(f"폴드 B 조기종료 best_iter {n_iter}")

# 3. 전체 학습 → 평가 구간 예측
m = fit_group(train_rows(Gtr, CV_START, Gtr["ts"].max()), smp_tr, n_estimators=n_iter)
smp = load_smp_test()
index = pd.DatetimeIndex(smp["ts"])
tabs_te = station_tables("test", index[0], index[-1], sites, log)
base_te = pd.MultiIndex.from_product([sites["site_id"], index], names=["site_id", "ts"]).to_frame(index=False)
Gte = {r: group_frame(build_frame(base_te, sites, *tabs_te[r]), group_of) for r in ROUNDS}
Gte = joint(Gte[17], Gte[10])
pg = predict_group_kwh(m, Gte).rename("group_kwh")
log(f"그룹 모델 평가 예측 합계 {pg.sum() / 1e6:,.2f} GWh (주간 {len(pg):,}행)")

# 4. 블렌드
f = pd.read_csv(S / "forecast.csv")
f["ts_k"] = pd.to_datetime(f["ts"]).dt.tz_convert("Asia/Seoul")
base = f["forecast_kwh"].to_numpy("float64") / opts["--base-alpha"]
grp_kwh = pd.MultiIndex.from_arrays([f["group"], f["ts_k"]]).map(pg.to_dict().get)
grp_kwh = np.array([b if g is None or pd.isna(g) else g for g, b in zip(grp_kwh, base)], dtype="float64")
w = opts["--w"]
f["forecast_kwh"] = (opts["--alpha"] * ((1 - w) * base + w * grp_kwh)).clip(min=0).round(3)
assert f["forecast_kwh"].notna().all() and np.isfinite(f["forecast_kwh"]).all()
D.mkdir(parents=True)
f[["group", "ts", "forecast_kwh"]].to_csv(D / "forecast.csv", index=False)
shutil.copy(S / "composition.csv", D / "composition.csv")
pd.DataFrame({"group": f["group"], "ts": f["ts"], "base_kwh": base, "group_kwh": grp_kwh}).to_parquet(
    OUT / "16_parts.parquet", index=False)
log(f"저장: {D} (기존 {src} 합계 {base.sum() * opts['--base-alpha'] / 1e6:,.2f} GWh → "
    f"{f['forecast_kwh'].sum() / 1e6:,.2f} GWh, W {w}, α {opts['--alpha']})")
