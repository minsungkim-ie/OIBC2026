"""제출 형식 검증. (실패 목록, 경고 목록, 정보 목록)을 반환."""
import numpy as np
import pandas as pd

from .config import DATA, MIN_GROUP_CAP

TS_PATTERN = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\+09:00"


def check_submission(sub_dir):
    fails, warns, info = [], [], []
    sites = pd.read_csv(DATA / "sites.csv")
    smp = pd.read_csv(DATA / "smp.csv")
    cpath, fpath = sub_dir / "composition.csv", sub_dir / "forecast.csv"
    for p in (cpath, fpath):
        if not p.exists():
            fails.append(f"파일 없음: {p}")
    if fails:
        return fails, warns, info

    # composition
    header = cpath.read_text(encoding="utf-8").splitlines()[0].strip()
    if header != "site_id,group":
        fails.append(f"composition 헤더 오류: {header!r}")
    c = pd.read_csv(cpath, dtype=str)
    if len(c) != 253:
        fails.append(f"composition 행 수 {len(c)} != 253")
    if c["site_id"].duplicated().any():
        fails.append("composition site_id 중복")
    if set(c["site_id"]) != set(sites["site_id"]):
        fails.append("composition site_id 집합이 sites.csv와 다름")
    if not c["group"].isin(["G1", "G2"]).all():
        fails.append("composition group 값이 G1/G2가 아님")
    cap = c.merge(sites[["site_id", "capacity_kw"]], on="site_id").groupby("group")["capacity_kw"].sum()
    for g in ("G1", "G2"):
        v = float(cap.get(g, 0.0))
        info.append(f"{g} 용량 {v:,.1f} kW, {int((c['group'] == g).sum())}개소")
        if v < MIN_GROUP_CAP:
            fails.append(f"{g} 설비용량 {v:.1f} < {MIN_GROUP_CAP}")

    # forecast
    header = fpath.read_text(encoding="utf-8").splitlines()[0].strip()
    if header != "group,ts,forecast_kwh":
        fails.append(f"forecast 헤더 오류: {header!r}")
    f = pd.read_csv(fpath, dtype={"group": str, "ts": str})
    if len(f) != 17520:
        fails.append(f"forecast 행 수 {len(f)} != 17520")
    if not f["ts"].str.endswith("+09:00").all():
        fails.append("forecast ts가 +09:00으로 끝나지 않는 행 존재")
    if not f["ts"].str.fullmatch(TS_PATTERN).all():
        fails.append("forecast ts 형식이 YYYY-MM-DD HH:MM:SS+09:00 이 아님")
    if f.duplicated(["group", "ts"]).any():
        fails.append("forecast (group,ts) 중복")
    v = f["forecast_kwh"]
    if v.isna().any() or not np.isfinite(v.to_numpy(dtype="float64")).all():
        fails.append("forecast_kwh에 NaN/inf 존재")
    if (v < 0).any():
        fails.append("forecast_kwh 음수 존재")
    smp_ts = set(smp["ts"])
    for g in ("G1", "G2"):
        s = f[f["group"] == g]
        if len(s) != 8760:
            fails.append(f"{g} 행 수 {len(s)} != 8760")
        if set(s["ts"]) != smp_ts:
            fails.append(f"{g} ts 집합이 smp.csv와 다름")
    if set(f["group"]) - {"G1", "G2"}:
        fails.append("forecast group 값이 G1/G2가 아님")

    # 상식 점검(경고만)
    hh = f["ts"].str.slice(11, 13).astype(int)
    night = f.loc[(hh >= 20) | (hh <= 5), "forecast_kwh"].sum()
    tot = f["forecast_kwh"].sum()
    info.append(f"야간(20~05시) 예측 비중 {night / max(tot, 1e-9) * 100:.3f}%")
    if night / max(tot, 1e-9) >= 0.01:
        warns.append("야간 예측 합이 전체의 1% 이상")
    for g in ("G1", "G2"):
        spec = f.loc[f["group"] == g, "forecast_kwh"].sum() / float(cap.get(g, np.nan))
        info.append(f"{g} 연간 예측/용량 = {spec:,.0f} kWh/kW")
        if not 900 <= spec <= 1600:
            warns.append(f"{g} 연간 예측/용량 {spec:.0f} kWh/kW가 900~1,600 범위 밖")
    return fails, warns, info
