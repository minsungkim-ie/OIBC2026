import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from oibc.config import DATA, OUT

OUT.mkdir(parents=True, exist_ok=True)
lines = []
def p(m=""):
    print(m); lines.append(str(m))

files = ["train/generation.parquet", "train/forecast_wa.parquet", "train/forecast_wb.parquet",
         "train/observed_wa.parquet", "train/observed_wb.parquet",
         "test/forecast_wa.parquet", "test/forecast_wb.parquet"]
expect = {"train/generation.parquet": 4392566, "train/forecast_wa.parquet": 4710356,
          "train/forecast_wb.parquet": 5621996, "train/observed_wa.parquet": 2359974,
          "train/observed_wb.parquet": 2824223, "test/forecast_wa.parquet": 1068456,
          "test/forecast_wb.parquet": 1278960}
ok = True
for f in files:
    d = pd.read_parquet(DATA / f)
    p(f"== {f}: rows={len(d):,} (README {expect[f]:,}) {'OK' if len(d)==expect[f] else 'MISMATCH'}")
    ok &= len(d) == expect[f]
    p(f"   dtypes: {dict(d.dtypes.astype(str))}")
    ts = pd.to_datetime(d.ts)
    p(f"   ts: {ts.min()} ~ {ts.max()}")
    na = d.isna().sum(); na = na[na > 0]
    p(f"   NaN: {dict(na) if len(na) else 'none'}")
for f, n, cols in [("sites.csv", 253, None), ("smp.csv", 8760, None), ("train/smp.csv", 38688, None)]:
    d = pd.read_csv(DATA / f)
    p(f"== {f}: rows={len(d):,} (README {n:,}) {'OK' if len(d)==n else 'MISMATCH'} cols={list(d.columns)}")
    ok &= len(d) == n
    if "ts" in d: p(f"   ts: {d.ts.iloc[0]} ~ {d.ts.iloc[-1]}")
s = pd.read_csv(DATA / "sites.csv"); p(f"capacity sum = {s.capacity_kw.sum():.1f}")
p("RESULT: " + ("README와 일치" if ok else "README와 불일치"))
(OUT / "00_data_check.txt").write_text("\n".join(lines), encoding="utf-8")
