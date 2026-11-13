"""v17 제출 파일: v15 × (1 − u) + 그룹 합 손실 신경망(24_nn_final, 시드 평균) × u, 배율 조정.

사용: python scripts/25_nn_blend_final.py <새 버전> --u 0.3 --ratio R --seeds 42,43,44,45,46
  forecast = 0.979 × R × ((1 − u) × v15 / 0.979 + u × NN)
  R = 블렌드 최적 α / 기준 최적 α (23_nn_blend 2폴드 평균). v15의 배율 근거(v13 탐침 보정 포함)는 그대로 유지.
"""
import shutil
import sys

from _bootstrap import log
import numpy as np
import pandas as pd

from oibc.config import ROOT

args = sys.argv[1:]
opt = {"--u": "0.3", "--ratio": "1.0", "--seeds": "42,43,44,45,46"}
for k in list(opt):
    if k in args:
        i = args.index(k)
        opt[k] = args[i + 1]
        del args[i:i + 2]
(dst,) = args
u, R = float(opt["--u"]), float(opt["--ratio"])
seeds = opt["--seeds"].split(",")
SUB = ROOT / "submissions"
D = SUB / dst
assert not D.exists(), f"{D} 이미 있음"
f15 = pd.read_csv(SUB / "v15" / "forecast.csv")
tsk = pd.to_datetime(f15["ts"]).dt.tz_convert("Asia/Seoul")
key = pd.MultiIndex.from_arrays([f15["group"], tsk])
nn = []
for s in seeds:
    g = pd.read_parquet(ROOT / "outputs" / "v17" / f"nn_test_s{s}.parquet")
    g["ts"] = pd.to_datetime(g["ts"]).dt.tz_convert("Asia/Seoul")
    v = g.set_index(["g", "ts"])["nn_kwh"].reindex(key).to_numpy()
    assert np.isfinite(v).all(), f"시드 {s} 시각 누락"
    nn.append(v)
    log(f"시드 {s}: 합계 {v.sum() / 1e6:.2f} GWh")
nn = np.mean(nn, axis=0)
base = f15["forecast_kwh"].to_numpy("float64") / 0.979
new = (0.979 * R * ((1 - u) * base + u * nn)).clip(min=0).round(3)
assert np.isfinite(new).all()
D.mkdir(parents=True)
pd.DataFrame({"group": f15["group"], "ts": f15["ts"], "forecast_kwh": new}).to_csv(D / "forecast.csv", index=False)
shutil.copy(SUB / "v15" / "composition.csv", D / "composition.csv")
log(f"NN 평균 합계 {nn.sum() / 1e6:.2f} GWh, v15/0.979 {base.sum() / 1e6:.2f} GWh, NN/기준 상관 {np.corrcoef(nn, base)[0, 1]:.4f}")
log(f"저장 {D}: 합계 {new.sum() / 1e6:.3f} GWh (v15 {f15['forecast_kwh'].sum() / 1e6:.3f}), v15와 상관 "
    f"{np.corrcoef(new, f15['forecast_kwh'])[0, 1]:.5f}, u {u}, R {R}")
