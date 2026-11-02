"""예보 로딩 → 결측 대체 → 지점 단위 파생 → 라운드별 지점 테이블."""
from .config import PRECIP_COLS, ROUNDS, WA_COLS, WB_COLS
from .features import station_features
from .impute import impute_forecast
from .io import hourly_index, load_forecast


def station_tables(split, start, end, sites, log=print):
    """반환: {round: (fa_r, fb_r)} — 모두 NaN 0개. 정규화·통계 fit 없음(같은 ts/대상일 정보만)."""
    ts_index = hourly_index(start, end)
    out = {r: [None, None] for r in ROUNDS}
    for kind, cols in (("wa", WA_COLS), ("wb", WB_COLS)):
        f = load_forecast(split, kind, cols, start, end)
        key = f"{kind}_station"
        f = impute_forecast(f, key, cols, ts_index, set(PRECIP_COLS), f"{split}/{kind}", log)
        for r in ROUNDS:
            fr = f[f["round"] == r].drop(columns="round")
            out[r][0 if kind == "wa" else 1] = station_features(fr, key, kind, sites, log)
    return {r: tuple(v) for r, v in out.items()}
