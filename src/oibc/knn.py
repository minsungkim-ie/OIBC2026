"""v5: 아날로그 앙상블(kNN).

각 (발전소, 시각) 질의마다 학습 풀에서 **같은 시(hour)**의 과거 주간 표본 가운데
예보·태양 위치가 가장 비슷한 K개를 찾고, 그 이웃들의 정규화 발전량 cf / max(cosz, 0.1)의
평균에 질의 시각의 cosz를 곱해 설비이용률을 예측한다.

- 거리 계산용 표준화(평균·표준편차)는 학습 풀에서만 구한다 (평가 기간 통계 사용 없음).
- 질의에는 그 행의 피처(같은 대상일 예보로 만든 것)만 쓰므로 다른 날짜 예보가 섞이지 않는다.
- 발전소 효율 보정(site_eff): 학습 풀 안에서 같은 시각 표본을 뺀 이웃 예측 대비 실적 비율.
  이력이 짧은 발전소는 수축(shrink)으로 1에 가깝게 둔다. 이력이 없으면 1.
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

COSZ_MIN = 0.1


class AnalogKNN:
    def __init__(self, feats, k=100, site_eff=True, shrink=0, leaf_size=40):
        self.feats = dict(feats)
        self.k = k
        self.site_eff = site_eff
        self.shrink = shrink
        self.leaf_size = leaf_size

    def _x(self, d):
        cols = list(self.feats)
        return ((d[cols].to_numpy(dtype="float64") - self.mu) / self.sd * self.w).astype("float32")

    def fit(self, d, cf, sid=None):
        """d: 주간 학습 표본(피처 + hour, cosz, ts). cf: 설비이용률. sid: 발전소 키(효율 보정용)."""
        cols = list(self.feats)
        self.w = np.array([self.feats[c] for c in cols], dtype="float64")
        self.mu = d[cols].mean().to_numpy(dtype="float64")
        self.sd = d[cols].std().to_numpy(dtype="float64")
        self.sd[self.sd == 0] = 1.0
        X = self._x(d)
        hour = d["hour"].to_numpy()
        y = np.asarray(cf, dtype="float64") / np.maximum(d["cosz"].to_numpy(dtype="float64"), COSZ_MIN)
        self.pool = {}
        for h in np.unique(hour):
            i = np.where(hour == h)[0]
            self.pool[h] = (KDTree(X[i], leaf_size=self.leaf_size), y[i], i)
        self.eff = None
        if self.site_eff and sid is not None:
            self.eff = self._site_eff(d, X, hour, np.asarray(cf, dtype="float64"), np.asarray(sid))
        return self

    def _site_eff(self, d, X, hour, cf, sid):
        """학습 풀 표본마다 같은 시각(ts) 이웃을 뺀 kNN 예측을 만들고, 발전소별 Σ실적/Σ예측."""
        ts = d["ts"].to_numpy()
        cosz = np.maximum(d["cosz"].to_numpy(dtype="float64"), COSZ_MIN)
        pred = np.empty(len(d))
        for h, (tree, y, i) in self.pool.items():
            _, idx = tree.query(X[i], k=min(self.k + 20, len(i)))
            same = ts[i][idx] == ts[i][:, None]           # 자기 자신과 같은 시각의 다른 발전소(같은 예보) 제외
            yy = np.where(same, np.nan, y[idx])
            cnt = (~same).sum(axis=1)
            pred[i] = np.where(cnt > 0, np.nansum(yy, axis=1) / np.maximum(cnt, 1), np.nan) * cosz[i]
        ok = np.isfinite(pred)                             # 이웃이 전부 같은 시각이면 비율 계산에서 뺀다(분자·분모 함께)
        g = pd.DataFrame({"sid": sid[ok], "a": cf[ok], "p": pred[ok]}).groupby("sid")
        s = g.sum()
        n = g.size()
        ratio = s["a"] / s["p"]
        return (ratio * n + 1.0 * self.shrink) / (n + self.shrink)

    def predict(self, d, sid=None):
        """d: 주간 질의 행. 반환: 설비이용률 예측(클립 전)."""
        X = self._x(d)
        hour = d["hour"].to_numpy()
        cosz = np.maximum(d["cosz"].to_numpy(dtype="float64"), COSZ_MIN)
        out = np.zeros(len(d))
        for h in np.unique(hour):
            j = np.where(hour == h)[0]
            if h not in self.pool:                          # 학습 풀에 없는 시(극히 드묾) → 가장 가까운 시
                h = min(self.pool, key=lambda q: abs(q - h))
            tree, y, _ = self.pool[h]
            _, idx = tree.query(X[j], k=min(self.k, len(y)))
            out[j] = y[idx].mean(axis=1) * cosz[j]
        if self.eff is not None and sid is not None:
            out *= pd.Series(np.asarray(sid)).map(self.eff).fillna(1.0).to_numpy()
        return out
