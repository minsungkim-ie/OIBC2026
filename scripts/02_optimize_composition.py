"""검증 잔차(학습 구간 데이터)로 G1/G2 구성 최적화. 테스트 예보·테스트 SMP는 쓰지 않는다."""
import numpy as np
import pandas as pd

from _bootstrap import log
from oibc.composition import local_search, lower_bound, objective, random_feasible
from oibc.config import OPT_SEED0, OPT_STARTS, OUT, VALID
from oibc.io import hourly_index, load_sites, load_smp_train
from oibc.scoring import residual_matrix

sites = load_sites()
order = sites["site_id"].tolist()
cap = sites["capacity_kw"].to_numpy(dtype="float64")
ref_x = (sites["current_group"] == "B").astype(np.int8).to_numpy()      # A→G1(0), B→G2(1)

oof = pd.read_parquet(OUT / "oof_valid.parquet")
smp = load_smp_train().set_index("ts")["smp"]
ts_index = hourly_index(*VALID)
R, w = residual_matrix(oof, order, ts_index, smp)
log(f"잔차 행렬 R {R.shape}, 실적 있는 셀 {len(oof):,}")

tdate = (ts_index - pd.Timedelta(hours=1)).floor("D")
week = ((tdate - tdate[0]).days // 7).to_numpy()
odd = (week % 2) == 1
even = ~odd


def cap_of(x):
    return cap[x == 0].sum(), cap[x == 1].sum()


def multi_start(Rm, wm, tag):
    best_x, best_W = None, np.inf
    for s in range(OPT_STARTS):
        if s == 0:
            x0 = ref_x.copy()
        else:
            x0 = random_feasible(cap, np.random.default_rng(OPT_SEED0 + s - 1))    # seed 42~60
        x, W = local_search(Rm, wm, cap, x0)
        if W < best_W:
            best_x, best_W = x, W
        log(f"  [{tag}] start {s:2d}: W={W:,.0f}  (best {best_W:,.0f})")
    return best_x, best_W


W_ref = objective(R, w, ref_x)
W_lb = lower_bound(R, w)
log(f"VALID 전체: 참조 구성 W={W_ref:,.0f}, W_LB={W_lb:,.0f}")

# 과적합 점검: 홀수 주로 최적화 → 짝수 주 채점 (반대 방향도 함께 보고)
cross = {}
for a_name, a, b_name, b in (("홀수", odd, "짝수", even), ("짝수", even, "홀수", odd)):
    x_a, _ = multi_start(R[:, a], w[a], f"{a_name}주 최적화")
    ref_b = objective(R[:, b], w[b], ref_x)
    opt_b = objective(R[:, b], w[b], x_a)
    lb_b = lower_bound(R[:, b], w[b])
    cross[(a_name, b_name)] = (ref_b, opt_b, lb_b)
    log(f"  {a_name}주 최적화 → {b_name}주 채점: 참조 W={ref_b:,.0f} → 최적 W={opt_b:,.0f} (W_LB {lb_b:,.0f})")

ref_b, opt_b, _ = cross[("홀수", "짝수")]
keeps = opt_b < ref_b                       # 지침: 홀수→짝수에서 개선이 유지될 때만 채택
log(f"과적합 점검(홀수→짝수) {'통과: 최적 구성 재최적화' if keeps else '실패: 참조 구성 유지'}")

x_full, W_full = multi_start(R, w, "VALID 전체")
x_final = x_full if keeps else ref_x
if keeps and W_full >= W_ref:
    x_final = ref_x
    keeps = False
W_final = objective(R, w, x_final)

comp = pd.DataFrame({"site_id": order, "group": np.where(x_final == 0, "G1", "G2")})
comp.to_csv(OUT / "composition_opt.csv", index=False)
c1, c2 = cap_of(x_final)
cr1, cr2 = cap_of(ref_x)

md = ["# 02 구성 최적화 리포트", "",
      f"- 잔차 기준: VALID 1년 (v2 모델 OOF), 가중치 0.10·SMP_train·k",
      f"- W_LB (구성과 무관한 하한) = {W_lb:,.0f}", "",
      "| 구성 | W | W_LB 대비 격차 | G1 용량(kW) | G2 용량(kW) | G1 개소 | G2 개소 |", "|---|---|---|---|---|---|---|",
      f"| 참조 (A→G1, B→G2) | {W_ref:,.0f} | {W_ref - W_lb:,.0f} | {cr1:,.1f} | {cr2:,.1f} | {(ref_x==0).sum()} | {(ref_x==1).sum()} |",
      f"| VALID 전체 최적 | {W_full:,.0f} | {W_full - W_lb:,.0f} | {cap_of(x_full)[0]:,.1f} | {cap_of(x_full)[1]:,.1f} | {(x_full==0).sum()} | {(x_full==1).sum()} |",
      "", "## 과적합 점검 (주 단위 교차)", "",
      "| 최적화 구간 → 채점 구간 | 참조 W | 최적 구성 W | 개선률 | 구간 W_LB |", "|---|---|---|---|---|"]
for (a, b), (rb, ob, lb) in cross.items():
    md.append(f"| {a}주 → {b}주 | {rb:,.0f} | {ob:,.0f} | {(1 - ob / rb) * 100:.2f}% | {lb:,.0f} |")
md += ["", f"- 지침 판정(홀수→짝수 개선 유지): **{'통과 → 최적 구성 채택' if keeps else '미통과 → 참조 구성 유지'}**",
       f"- 최종 채택: W={W_final:,.0f} (참조 대비 {(1 - W_final / W_ref) * 100:.2f}% 개선), "
       f"G1={c1:,.1f} kW ({(x_final==0).sum()}개소), G2={c2:,.1f} kW ({(x_final==1).sum()}개소)"]
(OUT / "02_composition_report.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
log("완료")
