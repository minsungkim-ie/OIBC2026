"""제출 파일: 변형 앙상블 (라운드 w17=0.7) × 배율 α, 구성은 앙상블 검증 잔차로 최적화.

사용: python scripts/13_final.py <버전> <변형...>
  v4 = python scripts/13_final.py v4 v2 csi      (인자 없으면 v4)
  v7 = python scripts/13_final.py v7 v2 csi v2e csie --comp submissions/v4/composition.csv --alpha 0.9729
  v8 후보 = python scripts/13_final.py v8 v2y csiy --alpha-src est
  --comp: 구성 최적화 대신 이 구성 파일 사용, --alpha: 검증 결과 대신 이 배율 사용
  v9 = python scripts/13_final.py v9 v2 csi v2e csie v2e_c csie_c --weights 0.125,0.125,0.125,0.125,0.25,0.25 --comp submissions/v4/composition.csv --alpha 0.9729
  --weights: 변형별 혼합 가중치(합 1, 변형 순서대로). 없으면 단순 평균
  --alpha-src est: 배율을 "[이력]" 조건(평가 연도와 같은 조건) 결과에서 고른다 (기본은 전체 발전소)

1. α = 2폴드 검증에서 이 앙상블의 해마다 최적 배율 평균 (12_cv_results.csv)
2. 구성 = 폴드 B(2024-06~2025-05) 앙상블 잔차로 지역탐색. 홀수주 최적화 → 짝수주 채점에서 개선이 유지될 때만 채택
3. FINAL_TRAIN(2023-06~2025-05)으로 변형·라운드별 재학습. 트리 수 = 폴드 B best_iter (학습 기간 길이가 같다)
4. 평가 구간 예측 → 그룹 합산 → submissions/<버전>/
   효율 피처: eff=True(v7) 변형은 FINAL_TRAIN 마지막 1년 실적으로 잰 한 값을 학습·평가 행에 붙인다.
   eff="year"/"month" 변형은 학습 행에 시점 기준 값(블록 첫 대상일 전일 17시까지의 직전 1년),
   평가 행에는 2025-05-31 17:00까지의 직전 1년 값(1년 내내 고정)을 붙인다.
"""
import gc
import json
import sys

from _bootstrap import log
import numpy as np
import pandas as pd

from oibc.composition import local_search, objective, random_feasible
from oibc.config import FINAL_TRAIN, OPT_SEED0, OPT_STARTS, ROOT, ROUNDS
from oibc.cv import (EFF_FEATS, EFF_WINDOW_DAYS, FOLDS, OUT_CV, PIT_MODES, V4_VARIANTS, VARIANTS, W17, _eff_prep,
                     eff_source, predict_cf, site_efficiency, site_efficiency_pit, target_and_weight)
from oibc.features import build_frame, clean_flags
from oibc.io import hourly_index, load_generation, load_sites, load_smp_test, load_smp_train
from oibc.model import fit_model
from oibc.pipeline import station_tables
from oibc.scoring import residual_matrix

args = sys.argv[1:]
opts = {}
for key in ("--comp", "--alpha", "--alpha-src", "--weights"):
    if key in args:
        i = args.index(key)
        opts[key] = args[i + 1]
        del args[i:i + 2]
VERSION = args[0] if args else "v4"
VARS = tuple(args[1:]) or V4_VARIANTS
assert all(v in VARIANTS for v in VARS), f"모르는 변형: {[v for v in VARS if v not in VARIANTS]}"
WTS = (np.array([float(x) for x in opts["--weights"].split(",")]) if "--weights" in opts
       else np.full(len(VARS), 1 / len(VARS)))
assert len(WTS) == len(VARS) and abs(WTS.sum() - 1) < 1e-9, f"가중치 {WTS}"
SUB = ROOT / "submissions" / VERSION
OUT = OUT_CV / VERSION if VERSION != "v4" else OUT_CV
OUT.mkdir(parents=True, exist_ok=True)
log(f"{VERSION}: 변형 {VARS}, 가중치 {WTS.tolist()}")
SUB.mkdir(parents=True, exist_ok=True)
OOF = OUT_CV / "oof"
sites = load_sites()
order = sites["site_id"].tolist()
cap = sites["capacity_kw"].to_numpy("float64")

# 1. 배율
if "--alpha" in opts:
    alpha = float(opts["--alpha"])
    log(f"배율 α = {alpha:.4f} (지정값)")
else:
    res = pd.read_csv(OUT_CV / "12_cv_results.csv").set_index("label")
    label = (" + ".join(VARS) if len(VARS) > 1 else VARS[0]) + f" · w17={W17}"
    if opts.get("--alpha-src") == "est":
        label += " [이력]"
    alpha = float(res.loc[label, ["a_best_A", "a_best_B"]].mean())
    log(f"배율 α = {alpha:.4f} ({label}: A {res.loc[label, 'a_best_A']:.4f}, B {res.loc[label, 'a_best_B']:.4f})")


def blend(parts):
    """parts: {(variant, round): array} → 변형별 라운드 가중 후 변형 가중 평균(WTS)."""
    return np.sum([wv * ((1 - W17) * parts[(v, 10)] + W17 * parts[(v, 17)]) for wv, v in zip(WTS, VARS)], axis=0)


# 2. 구성 (폴드 B 잔차, 또는 --comp 파일)
if "--comp" in opts:
    comp = pd.read_csv(ROOT / opts["--comp"])
    assert len(comp) == 253 and set(comp["site_id"]) == set(order)
    comp = comp.set_index("site_id").loc[order].reset_index()[["site_id", "group"]]
    grp = comp.set_index("site_id")["group"]
    log(f"구성: {opts['--comp']} 사용 (G1 {cap[(comp['group'] == 'G1').to_numpy()].sum():,.1f} kW)")
else:
    fb = {(v, r): pd.read_parquet(OOF / f"{v}_B_r{r}.parquet") for v in VARS for r in ROUNDS}
    oof = fb[(VARS[0], 10)][["site_id", "ts", "energy"]].copy()
    oof["pred_kwh"] = alpha * blend({k: d["pred_kwh"].to_numpy() for k, d in fb.items()})
    ts_index = hourly_index(*FOLDS["B"][2:])
    R, w = residual_matrix(oof, order, ts_index, load_smp_train().set_index("ts")["smp"])
    ref_x = (sites["current_group"] == "B").astype(np.int8).to_numpy()
    tdate = (ts_index - pd.Timedelta(hours=1)).floor("D")
    odd = (((tdate - tdate[0]).days // 7).to_numpy() % 2) == 1


    def multi_start(Rm, wm):
        best_x, best_W = None, np.inf
        for s in range(OPT_STARTS):
            x0 = ref_x.copy() if s == 0 else random_feasible(cap, np.random.default_rng(OPT_SEED0 + s - 1))
            x, W = local_search(Rm, wm, cap, x0)
            if W < best_W:
                best_x, best_W = x, W
        return best_x, best_W


    x_odd, _ = multi_start(R[:, odd], w[odd])
    ref_even, opt_even = objective(R[:, ~odd], w[~odd], ref_x), objective(R[:, ~odd], w[~odd], x_odd)
    log(f"홀수주 최적화 → 짝수주 채점: 참조 {ref_even:,.0f} → 최적 {opt_even:,.0f} ({(1 - opt_even / ref_even) * 100:.2f}%)")
    x_full, W_full = multi_start(R, w)
    W_ref = objective(R, w, ref_x)
    x = x_full if (opt_even < ref_even and W_full < W_ref) else ref_x
    log(f"구성: 참조 W {W_ref:,.0f} → 채택 W {objective(R, w, x):,.0f} "
        f"(G1 {cap[x == 0].sum():,.1f} kW / G2 {cap[x == 1].sum():,.1f} kW)")
    comp = pd.DataFrame({"site_id": order, "group": np.where(x == 0, "G1", "G2")})
    grp = comp.set_index("site_id")["group"]
    del fb, oof, R
    gc.collect()

# 3·4. 최종 학습 → 평가 구간 예측
smp = load_smp_test()
index = pd.DatetimeIndex(smp["ts"])
tabs_tr = station_tables("train", FINAL_TRAIN[0], FINAL_TRAIN[1], sites, log)
tabs_te = station_tables("test", index[0], index[-1], sites, log)
gen = load_generation(*FINAL_TRAIN)
base_te = pd.MultiIndex.from_product([sites["site_id"], index], names=["site_id", "ts"]).to_frame(index=False)
modes = {VARIANTS[v].get("eff") for v in VARS} - {None, False}
eff_x = None
if modes & set(PIT_MODES):
    h0 = FINAL_TRAIN[0] - pd.Timedelta(days=EFF_WINDOW_DAYS + 31)     # 학습 첫 블록의 직전 1년 이력까지
    eff_x = _eff_prep(eff_source(load_generation(h0, FINAL_TRAIN[1]), sites))
    log(f"시점 기준 효율용 이력 {len(eff_x):,}행 ({eff_x['ts'].min()} ~ {eff_x['ts'].max()})")
preds = {}
for r in ROUNDS:
    d = build_frame(gen[["site_id", "ts", "energy"]], sites, *tabs_tr[r])
    keep, _ = clean_flags(d)
    sel = (keep & (d["elev"] > 0)).to_numpy()
    dt = build_frame(base_te, sites, *tabs_te[r])
    # 효율 피처 표: 방식(mode)별로 한 번씩 만들고, 변형마다 d·dt의 EFF_FEATS 열을 바꿔 끼운다
    eff_tab = {}
    if True in modes:
        eff = site_efficiency(d.assign(keep=keep.to_numpy()), FINAL_TRAIN[1])
        eff_tab[True] = tuple({c: s["site_id"].map(eff[c]).astype("float32").to_numpy() for c in EFF_FEATS}
                              for s in (d, dt))
        log(f"효율 피처(v7 방식) {eff['eff_mid'].notna().sum()}곳, eff_days 최소 {eff['eff_days'].min():.0f}일")
    for mode in modes & set(PIT_MODES):
        e_tr = site_efficiency_pit(eff_x, d["site_id"].to_numpy(), d["ts"], mode)
        e_te = site_efficiency_pit(eff_x, dt["site_id"].to_numpy(), dt["ts"], mode, freeze_at=index[0], log=log)
        eff_tab[mode] = ({c: e_tr[c].to_numpy() for c in EFF_FEATS}, {c: e_te[c].to_numpy() for c in EFF_FEATS})
        log(f"효율 피처({mode}) 평가 행 {int(pd.Series(e_te['eff_mid'].to_numpy()).groupby(dt['site_id'].to_numpy()).first().notna().sum())}곳, "
            f"eff_days 최소 {np.nanmin(e_te['eff_days'].to_numpy()):.0f}일")
    for v in VARS:
        feats, target = VARIANTS[v]["feats"], VARIANTS[v]["target"]
        mode = VARIANTS[v].get("eff")
        if mode:
            for c in EFF_FEATS:
                d[c], dt[c] = eff_tab[mode][0][c], eff_tab[mode][1][c]
        n_iter = json.load(open(OOF / f"{v}_B_r{r}.json"))["best_iter"]
        y, wt = target_and_weight(d, target)
        m = fit_model(VARIANTS[v].get("algo", "lgb"), d[sel], y[sel], wt[sel], n_estimators=n_iter, feats=feats,
                      seed=VARIANTS[v].get("seed"))
        preds[(v, r)] = predict_cf(m, dt, feats, target) * dt["capacity_kw"].to_numpy("float64")
        log(f"{v} round {r} (트리 {n_iter}개): 예측 합계 {preds[(v, r)].sum() / 1e6:,.2f} GWh")
        del m
    del d, dt
    gc.collect()

ps = base_te.copy()
ps["pred_kwh"] = alpha * blend(preds)
ps.to_parquet(OUT / "test_site_pred.parquet", index=False)
ps["group"] = ps["site_id"].map(grp)
f = ps.groupby(["group", "ts"], as_index=False)["pred_kwh"].sum().rename(columns={"pred_kwh": "forecast_kwh"})
f = f.sort_values(["group", "ts"]).reset_index(drop=True)
f["forecast_kwh"] = f["forecast_kwh"].clip(lower=0).round(3)
f["ts"] = pd.DatetimeIndex(f["ts"]).tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S") + "+09:00"
f[["group", "ts", "forecast_kwh"]].to_csv(SUB / "forecast.csv", index=False)
comp.to_csv(SUB / "composition.csv", index=False)
log(f"저장: {SUB} (예측 합계 {f['forecast_kwh'].sum() / 1e6:,.2f} GWh, α={alpha:.4f})")
