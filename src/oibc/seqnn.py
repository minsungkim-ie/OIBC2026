"""v17 후보: 하루 단위 · 전 발전소 동시 · 그룹 합 손실 신경망 (PyTorch CPU).

표본 하나 = 대상일 하루. 입력 X[s, t, f] = 발전소 s(253), 대상일의 24시각(ts 01:00 … 다음날 00:00), 예보 피처 f
(10시·17시 두 라운드 + 태양 위치 + 달력). 대상일 안의 예보만 쓴다.
모델: 발전소마다 같은 1차원 CNN(시간축) + 발전소 임베딩 + 하루 문맥(시간 평균) + 지역 문맥(발전소 평균) → 시각별 cf.
손실: 채점식 그대로 — 그룹(G1·G2)별 시각별 |Σ 예측 − Σ 실적| × SMP(학습 구간) × k, 실적 있는 발전소만 합산.
      안정화를 위해 발전소 단위 가중 L1을 SITE_LAMBDA 비중으로 더한다.
SMP는 손실 가중치로만 쓰고 입력에는 넣지 않는다 (CLAUDE.md 규칙 5).
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import K_HIGH, K_HOURS, SEED

R_COLS = ["wa_cloud", "wa_temp", "wa_humidity", "wa_rain", "wa_snow", "wa_wind_speed", "wb_cloud", "wb_temp",
          "wb_rel_hum", "wb_dew_point", "wb_uv_idx", "wb_vis", "wb_ceiling", "wb_precip_prob", "wb_total_liq",
          "wb_snow", "wb_wind_speed"]
COMMON = ["elev", "cosz", "toa", "doy_sin", "doy_cos", "lat", "lon", "wa_dist", "wb_dist"]
NN_PARAMS = dict(hidden=64, emb=16, lr=2e-3, wd=1e-4, batch_days=4, max_epochs=40, patience=6, site_lambda=0.3,
                 sched=None)
# v18: sched="cos" → 조기종료 없이 max_epochs 고정, 학습률 코사인 감쇠(배치마다), 마지막 상태 사용.

# v18: 앞선 대상일 예보 흐름 (CLAUDE.md 규칙 4: D−1 대상일 예보는 D−2 17시 발행 → 허용). 17시 라운드 열만.
PREV_COLS = ["wa_cloud", "wa_humidity", "wa_temp", "wb_cloud", "wb_uv_idx", "wb_vis", "wb_precip_prob"]


def add_prev(D, src=None):
    """X 뒤에 붙인다: D−1 같은 시각 값, D−1 하루 평균, D−2 하루 평균 (PREV_COLS, 17시 라운드).

    앞선 날은 D 자체 또는 src(dict days·X, 예: 평가 구간 첫날의 앞날을 담은 학습 텐서)에서 찾는다.
    앞선 날이 없으면 그날 값으로 대체(변화 0). 다음 대상일 값은 쓰지 않는다.
    """
    ci = [R_COLS.index(c) for c in PREV_COLS]
    days, X = D["days"], D["X"]
    pool_days = days if src is None else days.append(src["days"])
    pool_X = X[..., ci] if src is None else np.concatenate([X[..., ci], src["X"][..., ci]])
    pos = pd.Series(np.arange(len(pool_days)), index=pool_days)
    pos = pos[~pos.index.duplicated()]
    cur = X[..., ci].astype("float32")

    def lag(k):
        j = pos.reindex(days - pd.Timedelta(days=k)).to_numpy()
        out = cur.copy()
        ok = ~np.isnan(j)
        out[ok] = pool_X[j[ok].astype(int)].astype("float32")
        return out

    l1, l2 = lag(1), lag(2)
    m1 = np.broadcast_to(l1.mean(2, keepdims=True), l1.shape)
    m2 = np.broadcast_to(l2.mean(2, keepdims=True), l2.shape)
    new = np.concatenate([X, l1.astype("float16"), m1.astype("float16"), m2.astype("float16")], -1)
    assert np.isfinite(new.astype("float32")).all()
    return dict(D, X=new)


def day_tensors(f17, f10, n_sites):
    """f17/f10: build_frame 결과(같은 (site_id, ts) 행, site_id 정수). energy·keep 있으면 타깃도.

    반환 dict: days(DatetimeIndex), X float16 [D,S,24,F], present bool [D,S], y float32 [D,S,24](cf, 없으면 nan),
    elev float32 [D,S,24], cap float32 [S], ts [D,24].
    """
    a = f17.sort_values(["site_id", "ts"]).reset_index(drop=True)
    b = f10.set_index(["site_id", "ts"]).reindex(pd.MultiIndex.from_frame(a[["site_id", "ts"]]))
    td = (a["ts"] - pd.Timedelta(hours=1)).dt.floor("D")
    days = pd.DatetimeIndex(np.sort(td.unique()))
    di = days.get_indexer(td)
    ti = ((a["ts"] - pd.Timedelta(hours=1)).dt.hour).to_numpy()
    si = a["site_id"].to_numpy()
    feats = [a[c].to_numpy("float32") for c in R_COLS] + [b[c].to_numpy("float32") for c in R_COLS] + \
            [a[c].to_numpy("float32") for c in COMMON]
    F = len(feats)
    X = np.zeros((len(days), n_sites, 24, F), dtype="float16")
    for j, v in enumerate(feats):
        X[di, si, ti, j] = v
    present = np.zeros((len(days), n_sites), bool)
    present[di, si] = True
    elev = np.zeros((len(days), n_sites, 24), "float32")
    elev[di, si, ti] = a["elev"].to_numpy("float32")
    y = np.full((len(days), n_sites, 24), np.nan, "float32")
    if "energy" in a.columns:
        ok = a["energy"].notna().to_numpy() & a["keep"].astype(bool).to_numpy()
        y[di[ok], si[ok], ti[ok]] = (a["energy"] / a["capacity_kw"]).to_numpy("float32")[ok]
    ts = np.array([[d + pd.Timedelta(hours=h + 1) for h in range(24)] for d in days])
    return dict(days=days, X=X, present=present, y=y, elev=elev, ts=ts)


class DayNet(nn.Module):
    def __init__(self, n_feat, n_sites, hidden=64, emb=16):
        super().__init__()
        self.emb = nn.Embedding(n_sites, emb)
        self.c1 = nn.Conv1d(n_feat + emb, hidden, 3, padding=1)
        self.c2 = nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2)
        self.c3 = nn.Conv1d(hidden * 3, hidden, 3, padding=1)
        self.out = nn.Conv1d(hidden, 1, 1)
        self.act = nn.GELU()

    def forward(self, x, present):
        # x [B,S,T,F], present [B,S]
        B, S, T, F = x.shape
        e = self.emb.weight[None, :, None, :].expand(B, S, T, -1)
        h = torch.cat([x, e], -1).reshape(B * S, T, -1).transpose(1, 2)        # [BS, C, T]
        h = self.act(self.c1(h))
        h = h + self.act(self.c2(h))
        C = h.shape[1]
        h4 = h.reshape(B, S, C, T)
        day = h4.mean(-1, keepdim=True).expand(-1, -1, -1, T)                  # 하루 문맥
        pm = present[:, :, None, None].float()
        reg = (h4 * pm).sum(1, keepdim=True) / pm.sum(1, keepdim=True).clamp(min=1)   # 지역 문맥(같은 날 발전소 평균)
        reg = reg.expand(-1, S, -1, -1)
        h = torch.cat([h4, day, reg], 2).reshape(B * S, 3 * C, T)
        h = self.act(self.c3(h))
        z = self.out(h).reshape(B, S, T)
        return 1.1 * torch.sigmoid(z)


def _weights(ts, smp):
    k = np.where(pd.DatetimeIndex(ts.ravel()).hour.isin(list(K_HOURS)), K_HIGH, 1).reshape(ts.shape)
    s = pd.Series(ts.ravel()).map(smp).to_numpy("float32").reshape(ts.shape)
    return (np.nan_to_num(s, nan=float(np.nanmean(s))) * k).astype("float32")


class Trainer:
    def __init__(self, gid, cap, mean, std, seed=SEED, params=None):
        self.p = dict(NN_PARAMS, **(params or {}))
        self.gid = torch.tensor(gid, dtype=torch.long)       # [S] 0/1
        self.cap = torch.tensor(cap, dtype=torch.float32)    # [S]
        self.mean, self.std = torch.tensor(mean), torch.tensor(std)
        self.seed = seed

    def _x(self, X):
        x = (torch.from_numpy(X.astype("float32")) - self.mean) / self.std
        return x.clamp(-6, 6)

    def _loss(self, cf, y, w, elev, present):
        # cf/y [B,S,T], w [B,T]; 실적 있는 칸만
        has = ~torch.isnan(y)
        yv = torch.nan_to_num(y)
        cap = self.cap[None, :, None]
        pk, ak = cf * cap * has, yv * cap * has
        G = torch.stack([(self.gid == g).float() for g in (0, 1)])            # [2,S]
        pg = torch.einsum("gs,bst->bgt", G, pk)
        ag = torch.einsum("gs,bst->bgt", G, ak)
        capg = torch.einsum("gs,bst->bgt", G, cap * has)
        lg = (w[:, None, :] * (pg - ag).abs()).sum() / (w[:, None, :] * capg).sum().clamp(min=1)
        ls = (w[:, None, :] * cap * (cf - yv).abs() * has).sum() / (w[:, None, :] * cap * has).sum().clamp(min=1)
        return lg + self.p["site_lambda"] * ls, lg

    def predict(self, model, D):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(D["days"]), 8):
                x = self._x(D["X"][i:i + 8])
                pr = torch.from_numpy(D["present"][i:i + 8])
                cf = model(x, pr) * (torch.from_numpy(D["elev"][i:i + 8]) > 0)
                out.append(cf.numpy())
        return np.concatenate(out)

    def fit(self, D, smp, idx_tr, idx_es=None, epochs=None, log=print):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        model = DayNet(D["X"].shape[-1], len(self.cap), self.p["hidden"], self.p["emb"])
        opt = torch.optim.AdamW(model.parameters(), lr=self.p["lr"], weight_decay=self.p["wd"])
        W = torch.from_numpy(_weights(D["ts"], smp))
        n_ep = epochs or self.p["max_epochs"]
        cos = self.p["sched"] == "cos"
        if cos:
            idx_es = None
            steps = n_ep * int(np.ceil(len(idx_tr) / self.p["batch_days"]))
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=self.p["lr"] * 0.02)
        best, best_ep, best_state, bad = np.inf, 0, None, 0
        for ep in range(1, n_ep + 1):
            model.train()
            perm = rng.permutation(idx_tr)
            for i in range(0, len(perm), self.p["batch_days"]):
                b = np.sort(perm[i:i + self.p["batch_days"]])
                x = self._x(D["X"][b])
                pr = torch.from_numpy(D["present"][b])
                cf = model(x, pr) * (torch.from_numpy(D["elev"][b]) > 0)
                loss, _ = self._loss(cf, torch.from_numpy(D["y"][b]), W[b], None, pr)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if cos:
                    sch.step()
            if cos and ep % 5 == 0:
                log(f"    ep {ep}: 마지막 배치 손실 {float(loss):.5f}")
            if idx_es is not None:
                cf = torch.from_numpy(self.predict(model, {k: (v[idx_es] if k != "days" else v[idx_es])
                                                           for k, v in D.items()}))
                _, lg = self._loss(cf, torch.from_numpy(D["y"][idx_es]), W[idx_es], None, None)
                v = float(lg)
                if v < best - 1e-6:
                    best, best_ep, bad = v, ep, 0
                    best_state = {k: t.clone() for k, t in model.state_dict().items()}
                else:
                    bad += 1
                log(f"    ep {ep}: es 그룹손실 {v:.5f} (best {best:.5f} @ {best_ep})")
                if bad >= self.p["patience"]:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.best_epoch = best_ep or n_ep
        return model


def norm_stats(X, idx):
    """학습 표본(idx)의 피처 평균·표준편차 (해당 폴드 학습 구간만)."""
    sub = X[idx].reshape(-1, X.shape[-1]).astype("float32")
    sub = sub[np.abs(sub).sum(1) > 0]
    m, s = sub.mean(0), sub.std(0)
    return m.astype("float32"), np.where(s < 1e-6, 1.0, s).astype("float32")


def group_sum_kwh(cf, D, gid, cap):
    """cf [D,S,24] → (g, ts) 시리즈 kWh. 발전소 전부(행이 있는 발전소) 합산."""
    pk = cf * cap[None, :, None] * D["present"][:, :, None]
    rows = []
    for g in (0, 1):
        v = pk[:, gid == g, :].sum(1)
        rows.append(pd.Series(v.ravel(), index=pd.MultiIndex.from_arrays(
            [np.full(v.size, f"G{g + 1}"), pd.DatetimeIndex(D["ts"].ravel())], names=["g", "ts"])))
    return pd.concat(rows)
