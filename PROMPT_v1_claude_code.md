# Claude Code 작업 지시서 — OIBC v1 (기준 파이프라인)

> 사용법: OIBC 폴더(`D:\data\HONGIK\4_2\OIBC`)에서 Claude Code를 열고, 아래 `---` 사이 내용을 그대로 붙여 넣는다.
> 같은 폴더의 `CLAUDE.md`(대회 규칙)와 `설명서_v1_방법론.md`를 Claude Code가 먼저 읽게 되어 있다.

---

너는 이 폴더에서 OIBC 태양광 경진대회 v1 제출물을 만드는 ML 엔지니어다.
먼저 `CLAUDE.md`(절대 규칙)와 `설명서_v1_방법론.md`(방법 설계)를 읽고, 그 설계를 **그대로** 구현해라.
목표는 "가장 대중적이고 안정적인 방법"의 첫 제출물이다. 화려한 기법보다 누수 없음·재현성·형식 정확성이 우선이다.

## 0. 환경 (uv)
- 이 폴더를 uv 프로젝트로 만든다: `uv init --name oibc --python 3.11` (이미 있으면 건너뜀).
- 의존성: `uv add pandas numpy pyarrow lightgbm scikit-learn matplotlib tqdm`
- 소스는 `src/oibc/` 패키지, 실행 스크립트는 `scripts/`, 산출물은 `outputs/`, 제출물은 `submissions/v1/`.
- `.gitignore`에 `outputs/`, `submissions/`, `.venv/`, `handover 2/`, `__MACOSX/` 추가.
- 데이터 루트: `DATA = Path("handover 2/01_participant_dataset")` (config 상수).
- 시작 전에 각 parquet의 스키마·행 수·기간·결측 수를 출력해서 README 내용과 맞는지 확인하고 결과를 `outputs/00_data_check.txt`에 저장해라. README와 다르면 멈추고 보고해라.

## 1. 파일 구조
```
src/oibc/
  config.py       # 경로, 기간, 시드, 피처 목록, LGBM 파라미터
  io.py           # 로딩(필요 열만, ts → tz-aware KST)
  solar.py        # 태양 고도·방위·cosz (baseline.py의 NOAA 근사 재사용, 구간 중앙 -30분)
  impute.py       # 예보 결측 대체
  features.py     # (site, ts) 격자 + 예보 병합 + 파생 피처
  model.py        # LightGBM 학습/예측 (라운드 10·17 각각 → 평균)
  scoring.py      # 로컬 정산금 S, 페널티 W, MAE
  composition.py  # 그룹 구성 최적화 (지역 탐색)
  checks.py       # 제출 형식 검증
scripts/
  01_validate.py              # 시간순 검증: 학습→검증 연도 예측, OOF 잔차 저장
  02_optimize_composition.py  # 검증 잔차로 G1/G2 구성 최적화
  03_train_final_predict.py   # 전체 학습 → 평가 구간 예측 → 제출 파일 생성
  04_check_submission.py      # 형식 검증 (실패 시 exit 1)
  run_all.py                  # 01→04 순서 실행
```

## 2. 기간 설정 (config.py)
| 이름 | 기간 (ts, KST, 우측닫힘) | 용도 |
|---|---|---|
| VAL_TRAIN | 2022-06-01 01:00 ~ 2024-06-01 00:00 | 검증용 모델 학습 (2년) |
| VAL_ES | VAL_TRAIN의 마지막 60일 | 조기 종료(early stopping)용 내부 홀드아웃 |
| VALID | 2024-06-01 01:00 ~ 2025-06-01 00:00 | 성능 측정 + 구성 최적화용 잔차 (1년) |
| FINAL_TRAIN | 2023-06-01 01:00 ~ 2025-06-01 00:00 | 최종 모델 학습 (2년) |
| TEST | `smp.csv`의 8,760시각 | 제출 |

- 검증 연도(VALID)는 조기 종료에 쓰지 않는다(검증 성능이 낙관적으로 나오지 않게).
- 최종 모델 트리 개수 = 검증 단계 best_iteration을 그대로 사용.

## 3. 전처리
### 3-1. 예보 결측 대체 (impute.py) — 절대 행을 버리지 않는다
1. 필요한 시각 전체로 완전 격자를 만든다: (관측지점 × round{10,17} × ts). 학습은 해당 기간, 테스트는 `smp.csv` 8,760시각.
2. 대상일 `target_date = (ts - 1h).date()`를 만든다.
3. 결측 채우기 순서:
   a. 같은 지점·같은 ts의 **다른 라운드** 값
   b. 같은 지점·같은 라운드·같은 target_date 안에서 시간 선형보간(`limit_direction="both"`) — **target_date를 넘지 않게 groupby**
   c. 그래도 남으면 같은 라운드·같은 ts의 **다른 지점들 중앙값**(같은 시각 예보끼리라 누수 없음)
   d. 강수·강설류는 남은 결측을 0으로
4. 대체 전/후 결측 수를 단계별로 로그 출력하고, 마지막에 `assert df.isna().sum().sum() == 0`.
5. 풍향은 보간 전에 sin/cos로 바꿔서 다룬다(359°→1° 보간 문제).

### 3-2. 발전량 정제 (학습 표본에만 적용)
- `cf = energy / capacity_kw` (설비이용률). `energy < 0` 행 제거, `cf > 1.2` 행 제거.
- **하루 종일 0인 발전소-일**(태양 고도 > 0인 시간대 발전량 합이 0)은 장애로 보고 학습에서 제외. 단, 로컬 채점(VALID)에서는 제외하지 않는다(평가 실적에도 장애·출력제어가 그대로 들어 있으므로).
- 제외한 행 수를 로그로 남긴다.

## 4. 피처 (features.py)
`(site_id, ts)` 격자에 `sites.csv`를 붙이고, `wa_station`/`wb_station` + `ts`로 해당 라운드 예보를 붙인다.

- 예보 A: `wa_cloud, wa_temp, wa_humidity, wa_rain, wa_snow, wa_wind_speed, wa_ground_press` (temp_max/min은 temp와 같으므로 제외)
- 예보 B: `wb_cloud, wb_temp, wb_rel_hum, wb_dew_point, wb_uv_idx, wb_vis, wb_ceiling, wb_precip_prob, wb_total_liq, wb_snow, wb_wind_speed`
- 태양: `elev, azim, cosz`, 대기권 밖 일사 근사 `toa = 1361 * (1 + 0.033*cos(2π·doy/365)) * max(cosz, 0)`
- 청천 대비 구름 효과: `toa_x_clear_a = toa * (1 - wa_cloud/100)`, `toa_x_clear_b = toa * (1 - wb_cloud/100)`
- 같은 대상일 안 시차/선행: `wa_cloud`, `wb_cloud`, `wb_uv_idx`의 ±1시간 값 (groupby `site_id, target_date`로 shift, 경계는 자기 값으로 채움)
- 같은 대상일 일 단위 집계: `wa_cloud`, `wb_cloud`의 주간(elev>0) 평균, `wb_total_liq` 일합
- 달력: `hour`, `doy_sin`, `doy_cos`, `month`
- 발전소: `site_id`(LightGBM categorical), `capacity_kw`, `lat`, `lon`, `wa_dist`, `wb_dist`
- 모든 수치형 float32.
- **금지 확인**: SMP, 실측 기상, 발전량 시차 피처 없음. 테스트 예보로 어떤 통계도 fit 하지 않음.

## 5. 모델 (model.py)
- 학습 대상: 태양 고도 `elev > 0`인 행만. 야간(elev ≤ 0)은 예측 0으로 고정.
- 타깃: `cf` (설비이용률).
- LightGBM 파라미터 (config 상수):
  ```python
  dict(objective="l1", learning_rate=0.05, num_leaves=63, min_child_samples=100,
       feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
       n_estimators=3000, seed=42, deterministic=True, force_row_wise=True, verbose=-1)
  ```
  - `objective="l1"`인 이유: 페널티가 절대오차이므로 중앙값 예측이 최적.
- 표본 가중치: `capacity_kw × k(ts)` — kWh 단위 오차와 11~15시 3배 가중을 학습에 반영. (SMP는 가중치에 넣지 않음, v1 단순화)
- 조기 종료: VAL_ES 구간, `early_stopping(200)`, 평가지표 l1.
- 라운드 10 예보 모델과 라운드 17 예보 모델을 **따로** 학습하고, 예측을 평균(kWh 기준)한다.
- 예측 후처리: `cf_pred.clip(0, 1.1)` → `kwh = cf_pred × capacity_kw`.
- 피처 중요도(gain) 상위 20개를 `outputs/feature_importance_r{10,17}.csv`로 저장.

## 6. 로컬 채점 (scoring.py)
- 입력: 발전소별 예측·실적 (site_id, ts, pred_kwh, energy), 구성(site→group), SMP(ts→smp), k.
- VALID는 학습 구간이므로 SMP는 `train/smp.csv`를 쓰고, 이 파일에는 k 열이 없으니 `ts.hour ∈ {11..15} → 3, 그 외 1`로 계산한다. 계산한 k를 `smp.csv`(평가 구간)의 k 열과 같은 규칙인지 한 번 대조해 assert.
- 실적이 없는 (site, ts)는 그 시각 그룹 합계에서 예측·실적 모두 빼고 계산(있는 발전소만 비교).
- 출력: 그룹별·전체 `S`, `W`(페널티 = 0.10·Σ SMP·k·|오차|), `매전수익`, 가중하지 않은 절대오차 합, 용량 대비 nMAE(주간).
- 하한값: `W_LB = 0.10·Σ_t SMP·k·|E_total(t)|` (E_total = 두 그룹 오차 합). 구성으로는 이 값 아래로 내려갈 수 없다(삼각부등식). 리포트에 항상 같이 출력.

## 7. 01_validate.py
1. VAL_TRAIN으로 라운드별 모델 학습(VAL_ES로 조기 종료) → VALID 예측.
2. 비교군 2개를 같은 VALID에서 채점:
   - (a) `baseline.py` 방식 근사(동일 피처셋이 아니라 baseline의 피처·파라미터, objective l2) — 참고용
   - (b) 우리 v1 모델
   - 둘 다 참조 구성(current_group A→G1, B→G2)으로 채점.
3. 발전소별 VALID 예측·실적·잔차를 `outputs/oof_valid.parquet`로 저장 (02에서 사용).
4. 결과표를 `outputs/01_validation_report.md`로 저장: S, W, W_LB, 절대오차 합, 월별 W, 시간대별 W.
5. best_iteration을 `outputs/best_iter.json`에 저장.

## 8. 02_optimize_composition.py (OR 파트)
목적: 발전소별 예측을 고정한 상태에서 G1/G2 배정을 바꿔 W를 최소화.

- 잔차 행렬 `R[i, t] = pred − actual` (VALID 연도, 실적 결측은 0), 가중치 `w[t] = 0.10·SMP_train(t)·k(t)`.
- 목적함수 `W(x) = Σ_t w[t]·(|Σ_{i∈G1} R[i,t]| + |Σ_{i∈G2} R[i,t]|)`
- 제약: `cap(G1) ≥ 20000`, `cap(G2) ≥ 20000` (합계 42,655.4이므로 각 그룹 20,000 ~ 22,655.4 kW).
- 알고리즘: **1-이동 + 2-교환 지역 탐색 (best-improvement)**
  - 그룹 잔차합 E1(t), E2(t)를 유지하고, 발전소 i를 옮길 때의 변화량을 numpy로 전 발전소 동시 계산.
  - 2-교환은 개선되는 1-이동이 없을 때만 시도하고, 계산량을 줄이려 각 그룹에서 1-이동 이득 상위 30개끼리만 쌍을 본다(30×30 쌍, 8,760시각 → 블록 단위 numpy 계산).
  - 개선 없을 때까지 반복. 시작점: 참조 구성 1개 + 무작위 실현가능 구성 19개(seed 42~60). 최선 결과 채택.
- 과적합 점검: VALID를 **홀수 주 / 짝수 주**로 나눠, 홀수 주로 최적화한 구성을 짝수 주에서 채점 → 참조 구성 대비 개선이 유지되는지 확인. 유지되면 VALID 전체로 다시 최적화한 구성을 최종 채택, 유지되지 않으면 참조 구성 유지.
- 저장: `outputs/composition_opt.csv`, `outputs/02_composition_report.md` (참조 vs 최적, 홀수/짝수 주 교차 결과, W_LB 대비 격차, 그룹 용량·발전소 수).
- 그룹 구성은 학습 구간 데이터(잔차·train SMP)로만 정한다. 테스트 예보·테스트 SMP 사용 금지.

## 9. 03_train_final_predict.py
1. FINAL_TRAIN으로 라운드별 모델 재학습 (n_estimators = best_iteration, 조기 종료 없음).
2. TEST 예보 결측 대체 → 피처 → 예측 → 라운드 평균 → 야간 0 → 그룹 합산.
3. 그룹 구성은 02의 최종 채택안.
4. 저장: `submissions/v1/composition.csv`, `submissions/v1/forecast.csv`
   - ts 문자열 형식 정확히 `YYYY-MM-DD HH:MM:SS+09:00`, 행 순서 group→ts.
   - `forecast_kwh` 소수 3자리 반올림.
5. 발전소별 예측도 `outputs/test_site_pred.parquet`로 저장(다음 버전 구성 실험용).

## 10. 04_check_submission.py (하나라도 실패하면 exit 1)
- composition: 헤더 `site_id,group`, 253행, `sites.csv`와 집합 일치, 중복 0, group ∈ {G1,G2}, 그룹별 용량 ≥ 20,000 출력.
- forecast: 헤더 `group,ts,forecast_kwh`, 17,520행, 그룹별 8,760행, ts 집합 == `smp.csv` ts 집합, 중복 0, 모든 ts 문자열이 `+09:00`로 끝남, NaN/inf/음수 0개.
- 상식 점검(경고만): 야간(ts 시 20~05) 예측 합이 전체의 1% 미만인지, 그룹별 연간 예측 합 / 용량이 900~1,600 kWh/kW 범위인지.

## 11. 마무리 보고
- `run_all.py`로 처음부터 끝까지 한 번 재실행해 결과가 같은지(시드 고정) 확인.
- `설명서_v1_방법론.md`의 "결과" 절을 실제 수치로 채워라(검증 S·W·W_LB, 구성 최적화 전후, 최종 그룹 용량, 소요 시간).
- 나에게 보고할 것: (1) 검증 연도 S/W 표 (baseline 근사 vs v1, 참조 구성 vs 최적 구성), (2) 04 검증 통과 여부, (3) 규칙 위반 가능성이 있는 부분과 근거, (4) 다음 개선 후보 3개.
- **대회 사이트 제출은 하지 마라.** 제출은 내가 직접 한다.

## 작업 방식
- 단계마다 짧게 진행 상황을 알려주고, 모르는 규칙은 `CLAUDE.md`를 근거로 판단하되 애매하면 멈추고 질문해라.
- 메모리가 부족하면 라운드별로 나눠 처리하고 `del` + `gc.collect()`로 정리해라.
- 에러가 나면 원인을 설명하고 고친 뒤 해당 단계부터 다시 실행해라.

---
