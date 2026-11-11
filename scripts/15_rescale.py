"""기존 제출의 예측에 배율을 곱해 새 제출을 만든다 (구성은 그대로).

사용: python scripts/15_rescale.py <원본 버전> <새 버전> <배율>
  v13 = python scripts/15_rescale.py v12 v13 0.95   (v12 예측 × 0.95, 평가 연도 총량 편향 확인용)

배율은 모든 시각·그룹에 같은 상수다. 평가 구간 예보로 계산한 값이 아니다 (CLAUDE.md 규칙 3).
"""
import shutil
import sys

from _bootstrap import log
import pandas as pd

from oibc.config import ROOT

src, dst, scale = sys.argv[1], sys.argv[2], float(sys.argv[3])
S, D = ROOT / "submissions" / src, ROOT / "submissions" / dst
assert not D.exists(), f"{D} 이미 있음 (제출한 폴더는 수정하지 않는다)"
D.mkdir(parents=True)
f = pd.read_csv(S / "forecast.csv")
f["forecast_kwh"] = (f["forecast_kwh"] * scale).round(3)
f.to_csv(D / "forecast.csv", index=False)
shutil.copy(S / "composition.csv", D / "composition.csv")
log(f"{src} × {scale} → {D} (예측 합계 {f['forecast_kwh'].sum() / 1e6:,.2f} GWh)")
