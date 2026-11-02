"""그룹 구성 최적화: 1-이동 + 2-교환 지역 탐색 (best-improvement)."""
import numpy as np

from .config import MIN_GROUP_CAP, SWAP_TOP


def objective(R, w, x):
    """x[i]=0→G1, 1→G2. W = Σ w·(|ΣG1 R| + |ΣG2 R|)."""
    e1 = R[x == 0].sum(0)
    e2 = R[x == 1].sum(0)
    return float((w * (np.abs(e1) + np.abs(e2))).sum())


def lower_bound(R, w):
    return float((w * np.abs(R.sum(0))).sum())


def random_feasible(cap, rng, lo=MIN_GROUP_CAP):
    tot = cap.sum()
    while True:
        x = np.ones(len(cap), dtype=np.int8)
        c1 = 0.0
        for i in rng.permutation(len(cap)):
            if c1 >= lo:
                break
            x[i] = 0
            c1 += cap[i]
        if c1 >= lo and tot - c1 >= lo:
            return x


def _move_deltas(R, w, x, e1, e2):
    """모든 발전소를 반대편으로 옮길 때의 W 변화량(벡터화)."""
    sign = np.where(x == 0, -1.0, 1.0)[:, None]          # G1 원소는 e1에서 빼고 e2에 더함
    n1 = np.abs(e1[None] + sign * R)
    n2 = np.abs(e2[None] - sign * R)
    cur = (np.abs(e1) + np.abs(e2))[None]
    return (w[None] * (n1 + n2 - cur)).sum(1)


def local_search(R, w, cap, x0, lo=MIN_GROUP_CAP, top=SWAP_TOP, max_iter=10000):
    x = x0.copy()
    tot = cap.sum()
    e1, e2 = R[x == 0].sum(0), R[x == 1].sum(0)
    c1 = cap[x == 0].sum()
    for _ in range(max_iter):
        d = _move_deltas(R, w, x, e1, e2)
        # 실현가능성: G1→G2이면 c1-cap≥lo, G2→G1이면 tot-c1-cap≥lo
        feas = np.where(x == 0, c1 - cap >= lo, (tot - c1) - cap >= lo)
        dm = np.where(feas, d, np.inf)
        i = int(np.argmin(dm))
        if dm[i] < -1e-9:
            s = -1.0 if x[i] == 0 else 1.0
            e1 = e1 + s * R[i]
            e2 = e2 - s * R[i]
            c1 += -cap[i] if x[i] == 0 else cap[i]
            x[i] = 1 - x[i]
            continue
        # 2-교환: 각 그룹에서 1-이동 이득 상위 top개끼리
        g1 = np.where(x == 0)[0]
        g2 = np.where(x == 1)[0]
        a = g1[np.argsort(d[g1])[:top]]
        b = g2[np.argsort(d[g2])[:top]]
        best, bp = 0.0, None
        base = np.abs(e1) + np.abs(e2)
        for ia in a:
            ok = (c1 - cap[ia] + cap[b] >= lo) & ((tot - c1) + cap[ia] - cap[b] >= lo)
            if not ok.any():
                continue
            bb = b[ok]
            n1 = np.abs(e1[None] - R[ia][None] + R[bb])
            n2 = np.abs(e2[None] + R[ia][None] - R[bb])
            dd = (w[None] * (n1 + n2 - base[None])).sum(1)
            j = int(np.argmin(dd))
            if dd[j] < best - 1e-9:
                best, bp = float(dd[j]), (int(ia), int(bb[j]))
        if bp is None:
            break
        ia, ib = bp
        e1 = e1 - R[ia] + R[ib]
        e2 = e2 + R[ia] - R[ib]
        c1 += -cap[ia] + cap[ib]
        x[ia], x[ib] = 1, 0
    return x, objective(R, w, x)
