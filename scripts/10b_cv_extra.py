"""2폴드 검증용 확장 예보 피처 캐시 (oibc.extra). frame_r{r}.parquet과 같은 행 순서로 extra_r{r}.parquet 저장."""
from _bootstrap import log
import pandas as pd

from oibc.config import ROUNDS
from oibc.cv import CACHE, CV_END, CV_START
from oibc.extra import EXTRA_FEATS, attach_extra, extra_tables
from oibc.io import load_sites
from oibc.pipeline import station_tables

sites = load_sites()
tabs = station_tables("train", CV_START, CV_END, sites, log, allow_nan=True)
tabs = {r: tuple(f.fillna(0) for f in v) for r, v in tabs.items()}   # 결손일은 10_cv_build에서 keep=False
xt = extra_tables(tabs, sites)
for r in ROUNDS:
    d = pd.read_parquet(CACHE / f"frame_r{r}.parquet", columns=["site_id", "ts", "wa_cloud", "wb_cloud"])
    d["site_id"] = d["site_id"].astype("int32")
    d = attach_extra(d, sites, *xt[r])
    na = int(d[EXTRA_FEATS].isna().sum().sum())
    log(f"round {r}: 확장 피처 {len(EXTRA_FEATS)}개, NaN {na}")
    d[EXTRA_FEATS].to_parquet(CACHE / f"extra_r{r}.parquet", index=False)
log("완료")
