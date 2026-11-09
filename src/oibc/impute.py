"""예보 결측 대체. 행을 버리지 않고 (지점 × 라운드 × ts) 완전 격자로 만들어 채운다.

순서: (a) 같은 지점·ts의 다른 라운드 → (b) 같은 지점·라운드·대상일 안 선형보간
→ (c) 같은 라운드·ts의 다른 지점 중앙값 → (d) 강수류 0. 모두 같은 ts 또는 같은 대상일 안의
정보만 쓰므로 대상일 경계를 넘는 누수가 없다.
"""
import numpy as np
import pandas as pd

from .config import ROUNDS


def impute_forecast(df, key, cols, ts_index, precip, tag, log=print, allow_nan=False):
    """df: [key, round, ts] + cols. ts_index: 01:00 시작·00:00 끝의 연속 시간 격자(24h × D일).
    allow_nan: 학습 구간 전용. 모든 지점의 예보가 통째로 없는 날(2022-05-08·09 등)은 채울 수 없어 NaN으로 남긴다."""
    assert len(ts_index) % 24 == 0 and ts_index[0].hour == 1, "ts_index는 대상일 단위(01:00~다음날 00:00)여야 함"
    stations = sorted(df[key].unique())
    S, R, T, C = len(stations), len(ROUNDS), len(ts_index), len(cols)
    D = T // 24
    d = df.drop_duplicates([key, "round", "ts"])
    assert len(d) == len(df), f"{tag}: 키 중복"
    grid = pd.MultiIndex.from_product([stations, ROUNDS, ts_index], names=[key, "round", "ts"])
    g = d.set_index([key, "round", "ts"])[cols].reindex(grid)
    assert len(d) <= len(g)
    arr = g.to_numpy(dtype="float32").reshape(S, R, T, C)
    miss0 = int(np.isnan(arr).sum())
    log(f"[impute {tag}] 격자 {S}지점×{R}라운드×{T}ts = {S*R*T:,}행, 원본 {len(d):,}행 "
        f"(격자 누락행 {S*R*T-len(d):,}), 대체 전 결측 셀 {miss0:,}")

    # a. 다른 라운드
    a = arr.copy()
    for r in range(R):
        m = np.isnan(a[:, r])
        a[:, r] = np.where(m, arr[:, 1 - r], a[:, r])
    log(f"[impute {tag}] a(다른 라운드) 후 결측 {int(np.isnan(a).sum()):,}")

    # b. 같은 지점·라운드·대상일 안 선형보간 (대상일 = 24h 블록)
    b = a.reshape(S, R, D, 24, C)
    if np.isnan(b).any():
        flat = b.transpose(3, 0, 1, 2, 4).reshape(24, -1)
        df2 = pd.DataFrame(flat)
        bad = df2.columns[df2.isna().any().to_numpy()]
        df2[bad] = df2[bad].interpolate(method="linear", axis=0, limit_direction="both")
        b = df2.to_numpy(dtype="float32").reshape(24, S, R, D, C).transpose(1, 2, 3, 0, 4)
    b = np.ascontiguousarray(b).reshape(S, R, T, C)
    log(f"[impute {tag}] b(대상일 내 보간) 후 결측 {int(np.isnan(b).sum()):,}")

    # c. 같은 라운드·ts의 다른 지점 중앙값
    c = b.copy()
    if np.isnan(c).any():
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(c, axis=0, keepdims=True)
        c = np.where(np.isnan(c), np.broadcast_to(med, c.shape), c).astype("float32")
    log(f"[impute {tag}] c(타 지점 중앙값) 후 결측 {int(np.isnan(c).sum()):,}")

    # d. 강수류 0
    for i, col in enumerate(cols):
        if col in precip:
            c[..., i] = np.nan_to_num(c[..., i], nan=0.0)
    n_left = int(np.isnan(c).sum())
    log(f"[impute {tag}] d(강수 0) 후 결측 {n_left:,}")
    assert allow_nan or n_left == 0, f"{tag}: 대체 후 NaN 잔존"

    out = pd.DataFrame(c.reshape(S * R * T, C), index=grid, columns=cols).reset_index()
    assert allow_nan or out[cols].isna().sum().sum() == 0
    return out
