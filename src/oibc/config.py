"""경로·기간·시드·피처 목록·하이퍼파라미터 상수."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "handover 2" / "01_participant_dataset"
OUT = ROOT / "outputs"                      # v1 산출물(best_iter, composition_opt)을 읽는 곳
# v6: v1 + 17시 라운드 가중
RUN_OUT = ROOT / "outputs" / "v6"           # 이번 실행 산출물 (v1 파일을 덮어쓰지 않는다)
SUB = ROOT / "submissions" / "v6"

TZ = "Asia/Seoul"
SEED = 42
ROUNDS = (10, 17)
# 라운드 평균 가중치. v1은 50:50. v1 검증 OOF에서 17시 0.7이 W 최소(212.78M → 211.73M),
# 홀수주·짝수주 모두 50:50보다 낮음 (0.8 이상은 짝수주에서 나빠짐)
ROUND_WEIGHTS = {10: 0.3, 17: 0.7}


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=TZ)


# 기간 (ts, KST, 우측닫힘)
VAL_TRAIN = (_ts("2022-06-01 01:00"), _ts("2024-06-01 00:00"))
VAL_ES_DAYS = 60
VALID = (_ts("2024-06-01 01:00"), _ts("2025-06-01 00:00"))
FINAL_TRAIN = (_ts("2023-06-01 01:00"), _ts("2025-06-01 00:00"))

# 정제
CF_MAX = 1.2
PRED_CF_CLIP = 1.1

# 평가식
PENALTY_RATE = 0.10
K_HOURS = range(11, 16)
K_HIGH = 3
MIN_GROUP_CAP = 20000.0

# 예보 열
WA_COLS = ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain", "wa_snow",
           "wa_wind_speed", "wa_ground_press"]
WA_RAW = WA_COLS + ["wa_wind_dir"]
WB_COLS = ["wb_cloud", "wb_temp", "wb_rel_hum", "wb_dew_point", "wb_uv_idx",
           "wb_vis", "wb_ceiling", "wb_precip_prob", "wb_total_liq", "wb_snow",
           "wb_wind_speed"]
WB_RAW = WB_COLS + ["wb_wind_dir"]
PRECIP_COLS = ["wa_rain", "wa_snow", "wb_precip_prob", "wb_total_liq", "wb_snow"]

LGBM_PARAMS = dict(
    objective="l1", learning_rate=0.05, num_leaves=63, min_child_samples=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    n_estimators=3000, seed=SEED, deterministic=True, force_row_wise=True,
    verbose=-1,
)
EARLY_STOP = 200

# v9: 앙상블 다양화용 다른 부스팅 라이브러리. 가중치는 평균 1로 정규화한 뒤 쓴다 (model.fit_model).
# 값은 우리 LightGBM 설정(잎 63·최소 표본 100·열/행 0.8)과 비슷한 복잡도로 맞춘 것.
XGB_PARAMS = dict(
    objective="reg:absoluteerror", tree_method="hist", learning_rate=0.05, max_depth=7,
    min_child_weight=100, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    n_estimators=3000, eval_metric="mae", n_jobs=-1, max_bin=256,
)
CAT_PARAMS = dict(
    loss_function="MAE", learning_rate=0.08, depth=7, l2_leaf_reg=3.0, iterations=3000,
    thread_count=-1, verbose=200, allow_writing_files=False,
)

# baseline.py 근사 (참고용 비교군, objective l2)
BASE_WA = ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain", "wa_wind_speed", "wa_ground_press"]
BASE_WB = ["wb_cloud", "wb_temp", "wb_rel_hum", "wb_uv_idx", "wb_precip_prob",
           "wb_total_liq", "wb_vis", "wb_wind_speed", "wb_dew_point"]
BASE_PARAMS = dict(
    n_estimators=600, learning_rate=0.06, num_leaves=127, min_child_samples=200,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
    random_state=SEED, deterministic=True, force_row_wise=True,
)

# 구성 최적화
OPT_STARTS = 20
OPT_SEED0 = 42
SWAP_TOP = 30

# 피처 (features.py가 만든다) — SMP·실측 기상·발전량 시차 피처는 없다.
LAG_COLS = {"wa": ["wa_cloud"], "wb": ["wb_cloud", "wb_uv_idx"]}
LAG_FEATS = [f"{c}_{s}" for c in ("wa_cloud", "wb_cloud", "wb_uv_idx") for s in ("m1", "p1")]
AGG_FEATS = ["wa_cloud_dmean", "wb_cloud_dmean", "wb_total_liq_dsum"]
SUN_FEATS = ["elev", "azim", "cosz", "toa", "toa_x_clear_a", "toa_x_clear_b"]
CAL_FEATS = ["hour", "doy_sin", "doy_cos", "month"]
SITE_FEATS = ["site_id", "capacity_kw", "lat", "lon", "wa_dist", "wb_dist"]
FEATURES = WA_COLS + WB_COLS + SUN_FEATS + LAG_FEATS + AGG_FEATS + CAL_FEATS + SITE_FEATS
BASE_FEATURES = BASE_WA + BASE_WB + ["elev", "azim", "cosz", "doy_sin", "doy_cos", "hour",
                                     "capacity_kw", "lat", "lon"]
