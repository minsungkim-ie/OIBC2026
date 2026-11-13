"""v18 제출 파일: v15 × (1 − u) + 신경망(시드·변형 평균) × u, 배율 조정 (25_nn_blend_final 일반화).

사용: python scripts/29_nn_blend_final_v18.py <새 버전> --nn "v17:42,43,44,45,46+cos20:42,..." --u 0.3 --ratio 1.0
  forecast = 0.979 × R × ((1 − u) × v15 / 0.979 + u × NN)
  태그 v17 = outputs/v17/nn_test_s<시드>, 그 밖 = outputs/v18/nn_test_<태그>_s<시드>. 구성은 v15(= v4) 복사.
"""
import argparse
import shutil

from _bootstrap import log
import numpy as np
import pandas as pd

from oibc.config import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("dst")
ap.add_argument("--nn", required=True)
ap.add_argument("--u", type=float, default=0.3)
ap.add_argument("--ratio", type=float, default=1.0)
a = ap.parse_args()
SUB = ROOT / "submissions"
D = SUB / a.dst
assert not D.exists(), f"{D} 이미 있음"
f15 = pd.read_csv(SUB / "v15" / "forecast.csv")
f17 = pd.read_csv(SUB / "v17" / "forecast.csv")
assert (f15[["group", "ts"]] == f17[["group", "ts"]]).all().all()
tsk = pd.to_datetime(f15["ts"]).dt.tz_convert("Asia/Seoul")
key = pd.MultiIndex.from_arrays([f15["group"], tsk])
nn = []
for part in a.nn.split("+"):
    tag, seeds = part.split(":")
    for s in seeds.split(","):
        p = ROOT / "outputs" / "v17" / f"nn_test_s{s}.parquet" if tag == "v17" else \
            ROOT / "outputs" / "v18" / f"nn_test_{tag}_s{s}.parquet"
        g = pd.read_parquet(p)
        g["ts"] = pd.to_datetime(g["ts"]).dt.tz_convert("Asia/Seoul")
        v = g.set_index(["g", "ts"])["nn_kwh"].reindex(key).to_numpy()
        assert np.isfinite(v).all(), f"{p.name} 시각 누락"
        nn.append(v)
        log(f"{tag} 시드 {s}: 합계 {v.sum() / 1e6:.2f} GWh")
nn = np.mean(nn, axis=0)
base = f15["forecast_kwh"].to_numpy("float64") / 0.979
new = (0.979 * a.ratio * ((1 - a.u) * base + a.u * nn)).clip(min=0).round(3)
assert np.isfinite(new).all()
D.mkdir(parents=True)
pd.DataFrame({"group": f15["group"], "ts": f15["ts"], "forecast_kwh": new}).to_csv(D / "forecast.csv", index=False)
shutil.copy(SUB / "v15" / "composition.csv", D / "composition.csv")
log(f"NN 평균({len(nn) if nn.ndim > 1 else '-'}) 합계 {nn.sum() / 1e6:.2f} GWh, v15/0.979 {base.sum() / 1e6:.2f} GWh, "
    f"NN/기준 상관 {np.corrcoef(nn, base)[0, 1]:.4f}")
log(f"저장 {D}: 합계 {new.sum() / 1e6:.3f} GWh (v15 {f15['forecast_kwh'].sum() / 1e6:.3f}, v17 "
    f"{f17['forecast_kwh'].sum() / 1e6:.3f}), 상관 v15 {np.corrcoef(new, f15['forecast_kwh'])[0, 1]:.5f} / v17 "
    f"{np.corrcoef(new, f17['forecast_kwh'])[0, 1]:.5f}, u {a.u}, R {a.ratio}")
