"""2폴드 검증용 피처 캐시: 2021-06 ~ 2025-05 학습 구간 전체를 라운드별로 한 번만 만든다.

학습 구간에는 모든 지점의 예보가 통째로 빠진 날(2022-05-08·09 등)이 있어 채울 수 없다.
그 대상일은 0으로 임시 채운 뒤 keep=False로 학습·채점에서 뺀다.
"""
from _bootstrap import log
import pandas as pd

from oibc.config import FEATURES, ROUNDS
from oibc.cv import CACHE, CV_END, CV_START, V1_FEATURES
from oibc.features import build_frame, clean_flags, target_date
from oibc.io import load_generation, load_sites
from oibc.pipeline import station_tables

CACHE.mkdir(parents=True, exist_ok=True)
sites = load_sites()
tabs = station_tables("train", CV_START, CV_END, sites, log, allow_nan=True)
gen = load_generation(CV_START, CV_END)
log(f"발전량 {len(gen):,}행")
cols = sorted(set(FEATURES) | set(V1_FEATURES) | {"ts", "energy", "capacity_kw", "elev", "cosz"})
for r in ROUNDS:
    bad = set()
    for f in tabs[r]:
        nan = f.drop(columns=[c for c in f.columns if c.endswith("_station") or c == "ts"]).isna().any(axis=1)
        bad |= set(target_date(f.loc[nan, "ts"]))
    fa, fb = (f.fillna(0) for f in tabs[r])
    log(f"round {r}: 예보가 통째로 빠진 대상일 {len(bad)}일 → 제외: {sorted(str(b.date()) for b in bad)}")
    d = build_frame(gen[["site_id", "ts", "energy"]], sites, fa, fb)
    keep, cnt = clean_flags(d)
    keep &= ~target_date(d["ts"]).isin(bad)
    log(f"round {r}: 정제 제외 {cnt}, 예보 결손일 제외 후 keep {int(keep.sum()):,}/{len(d):,}")
    d = d[cols].assign(keep=keep.to_numpy())
    d.to_parquet(CACHE / f"frame_r{r}.parquet", index=False)
    log(f"round {r}: 저장 {len(d):,}행 × {d.shape[1]}열")
    del d
log("완료")
