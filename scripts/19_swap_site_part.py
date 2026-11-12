"""v15: 그룹 블렌드 제출(v14)의 그룹 모델 예측은 그대로 두고, 발전소 합 부분만 다른 제출로 바꾼다.

사용: python scripts/19_swap_site_part.py <블렌드 제출> <원래 발전소 합 제출> <원래 발전소 α> <블렌드 α> <블렌드 w>
                                        <새 발전소 합 제출> <새 발전소 α> <새 α> <새 버전>
  v15 = python scripts/19_swap_site_part.py v14 v12 0.9717 0.981 0.3 cand_v12x 0.9697 0.979 v15

  블렌드 제출 = 블렌드 α × ((1 − w) × 원래 발전소 합 / 원래 발전소 α + w × 그룹 예측)  (16_group_blend 식)
  → 그룹 예측 = (블렌드 제출 / 블렌드 α − (1 − w) × 원래 발전소 합 / 원래 발전소 α) / w 로 정확히 되돌린다.
  야간 등 그룹 예측이 없던 시각(블렌드 제출 = 블렌드 α × 원래 발전소 합 / 원래 발전소 α)은 새 발전소 합 그대로.
  새 제출 = 새 α × ((1 − w) × 새 발전소 합 / 새 발전소 α + w × 그룹 예측), 구성은 블렌드 제출 그대로.

이유: 그룹 모델을 다른 PC에서 다시 학습하면 결과가 달라진다(LightGBM 버전 등, g 트리 669 vs 312).
v14에서 리더보드로 확인된 그룹 예측을 그대로 써서, 바뀌는 것을 발전소 합 한 가지로 묶는다.
"""
import shutil
import sys

from _bootstrap import log
import numpy as np
import pandas as pd

from oibc.config import ROOT

blend, old_site, a_old_site, a_blend, w, new_site, a_new_site, a_new, out = sys.argv[1:10]
a_old_site, a_blend, w, a_new_site, a_new = map(float, (a_old_site, a_blend, w, a_new_site, a_new))
SUB = ROOT / "submissions"
D = SUB / out
assert not D.exists(), f"{D} 이미 있음"

fb = pd.read_csv(SUB / blend / "forecast.csv")
fo = pd.read_csv(SUB / old_site / "forecast.csv")
fn = pd.read_csv(SUB / new_site / "forecast.csv")
for f in (fo, fn):
    assert (f["ts"] == fb["ts"]).all() and (f["group"] == fb["group"]).all()
assert pd.read_csv(SUB / new_site / "composition.csv").equals(pd.read_csv(SUB / blend / "composition.csv"))

base_old = fo["forecast_kwh"].to_numpy("float64") / a_old_site
g = (fb["forecast_kwh"].to_numpy("float64") / a_blend - (1 - w) * base_old) / w
no_group = np.isclose(fb["forecast_kwh"].to_numpy(), np.round(a_blend * base_old, 3), atol=2e-3)
rec = np.round(a_blend * ((1 - w) * base_old + w * g), 3)          # 역산이 맞는지 확인 (식 그대로 되돌림)
err = np.abs(rec - fb["forecast_kwh"].to_numpy()).max()
assert err < 1e-6, f"{blend} 재구성 오차 {err}"
log(f"{blend} 재구성 오차 {err:.4f} kWh, 그룹 예측 없는 시각 {int(no_group.sum()):,}개")

base_new = fn["forecast_kwh"].to_numpy("float64") / a_new_site
new = (a_new * np.where(no_group, base_new, (1 - w) * base_new + w * g)).clip(min=0).round(3)
D.mkdir(parents=True)
pd.DataFrame({"group": fb["group"], "ts": fb["ts"], "forecast_kwh": new}).to_csv(D / "forecast.csv", index=False)
shutil.copy(SUB / blend / "composition.csv", D / "composition.csv")
log(f"저장: {D} (예측 합계 {new.sum() / 1e6:,.2f} GWh, {blend} {fb['forecast_kwh'].sum() / 1e6:,.2f} GWh)")
