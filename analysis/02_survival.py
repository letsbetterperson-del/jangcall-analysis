# -*- coding: utf-8 -*-
"""2단계 — 포기한 호출을 넣어 대기시간을 다시 잰다 (생존분석).
실행: python analysis/02_survival.py   (01_prep.py 먼저)

왜: 서울시설공단 공개 '시간대별 대기시간'은 메타데이터에 "탑승 완료한 데이터 기준 집계"라고 적혀 있다.
기다리다 취소한 호출은 평균에서 빠진다 → 대기가 길수록 더 많이 빠지는 생존 편향.
방법:
 (a) 공식 집계 재현 검증: 공단 시간대별 대기시간(일×시) vs 탑승내역에서 같은 정의로 다시 계산한 값
 (b) Kaplan-Meier: 사건 = 탑승, 취소 = 그 시각까지는 못 탔다는 정보(우측 중도절단).
     "끝까지 기다렸다면 언제 탔을까"의 분포 → 60분 안에 탈 확률, 중앙값, 180분 제한 평균(RMST)
     가정: 취소 결정이 남은 대기와 독립(비정보적 중도절단). 위반 방향은 한계 절에 적는다
 (c) Aalen-Johansen 누적발생: 실제로 60분 안에 '탄' 비율과 '포기한' 비율(경쟁위험, 가정 없음)
 (d) 인내 분포: 취소까지 시간(사건 = 취소, 배차 = 중도절단) → 시뮬레이터 입력
 (e) 자치구·행정동별: 완료 기준 순위 vs 포기 포함 순위 어긋남
"""
import numpy as np, pandas as pd, os
from scipy.stats import spearmanr
from common import CACHE, OUT, RAW, HORIZON_MIN, save_summary

base = pd.read_parquet(os.path.join(CACHE, "base.parquet"))
GRID = np.arange(0, 241)      # 분 단위 격자 0..240


def km(T, E, grid=GRID):
    """Kaplan-Meier 생존함수 S(t)=P(아직 못 탐). T: 관측시간(분), E: 1=탑승, 0=중도절단."""
    T = np.clip(np.asarray(T, float), 0, None); E = np.asarray(E, bool)
    order_bins = np.floor(T).astype(int)
    maxb = grid[-1] + 1
    ev = np.bincount(np.minimum(order_bins[E], maxb), minlength=maxb + 1)[:maxb + 1]
    out_all = np.bincount(np.minimum(order_bins, maxb), minlength=maxb + 1)[:maxb + 1]
    at_risk = len(T) - np.concatenate([[0], np.cumsum(out_all)[:-1]])
    h = np.divide(ev, at_risk, out=np.zeros_like(ev, float), where=at_risk > 0)
    S = np.cumprod(1 - h)                      # S[k] = P(T > k+1분)
    S = np.concatenate([[1.0], S])[: len(grid)]  # S at grid minute t
    return S


def summarize(S):
    p60 = 1 - S[HORIZON_MIN]
    med = int(np.argmax(S <= 0.5)) if (S <= 0.5).any() else np.nan
    rmst = float(S[:180].sum())                # 180분 제한 평균(분)
    p90 = int(np.argmax(S <= 0.1)) if (S <= 0.1).any() else 241
    return p60, med, rmst, p90


def cif(T_b, T_c, grid=GRID):
    """경쟁위험 누적발생(탑승/취소). 중도절단 없음(모든 건 결말 관측) → 경험 누적비율."""
    n = len(T_b)
    b = np.searchsorted(np.sort(T_b[~np.isnan(T_b)]), grid, side="right") / n
    c = np.searchsorted(np.sort(T_c[~np.isnan(T_c)]), grid, side="right") / n
    return b, c


def prep(d):
    T = np.where(d["탑승"], d["T_탑승"], d["T_취소"]).astype(float)
    E = d["탑승"].values
    ok = ~np.isnan(T) & (T >= 0)
    return T[ok], E[ok]


out = {}
# ---------- (a) 공식 집계 재현 검증 ----------
off = pd.read_csv(os.path.join(RAW, "서울시설공단_장애인콜택시 시간대별 대기시간_2025.csv"), encoding="utf-8-sig")
cnt = pd.read_csv(os.path.join(RAW, "서울시설공단_장애인콜택시 시간대별 탑승건수_2025.csv"), encoding="utf-8-sig")
off["시"] = off["시간대"].str[:2].astype(int); cnt["시"] = cnt["시간대"].str[:2].astype(int)
off = off.merge(cnt[["일자", "시"] + [c for c in cnt.columns if c not in ("시설명", "일자", "시간대", "시")]],
                on=["일자", "시"], how="left")
wcol = [c for c in cnt.columns if c not in ("시설명", "일자", "시간대", "시")][0]
off["대기시간평균"] = pd.to_numeric(off["대기시간평균"], errors="coerce")
off[wcol] = pd.to_numeric(off[wcol], errors="coerce")
out["official_wait_weighted_mean"] = round(float(np.average(off["대기시간평균"], weights=off[wcol].fillna(0))), 1)
out["official_wait_simple_mean"] = round(float(off["대기시간평균"].mean()), 1)
done = base[base["탑승"] & (base["T_탑승"] >= 0)].copy()
done["일자s"] = done["일자"].dt.strftime("%Y-%m-%d")
mine = done.groupby(["일자s", "시"]).agg(w=("T_탑승", "mean"), n=("T_탑승", "size")).reset_index()
mm = off.merge(mine, left_on=["일자", "시"], right_on=["일자s", "시"])
r = np.corrcoef(mm["대기시간평균"], mm["w"])[0, 1]
out["official_vs_mine_corr"] = round(float(r), 3)
out["official_vs_mine_n_cells"] = int(len(mm))
out["official_vs_mine_mae"] = round(float((mm["대기시간평균"] - mm["w"]).abs().mean()), 1)
out["official_count_vs_mine_corr"] = round(float(np.corrcoef(mm[wcol], mm["n"])[0, 1]), 3)
mm[["일자", "시", "대기시간평균", "w", wcol, "n"]].to_csv(f"{OUT}/02_official_check.csv", index=False, encoding="utf-8-sig")

# ---------- (b)(c) 전체·즉시호출·차량구분 ----------
rows = []
for name, d in [("전체", base), ("즉시", base[~base["예약"]]), ("특장차_즉시", base[(~base["예약"]) & (base["차량구분"] == "특장차")]),
                ("임차택시_즉시", base[(~base["예약"]) & (base["차량구분"] == "임차택시")])]:
    T, E = prep(d)
    S = km(T, E)
    p60, med, rmst, p90 = summarize(S)
    Tb = np.where(E, T, np.nan); Tc = np.where(~E, T, np.nan)
    b, c = cif(Tb, Tc)
    done_d = T[E]
    rows.append(dict(구분=name, 호출=len(T), 취소율=round((~E).mean() * 100, 1),
                     완료기준_평균=round(done_d.mean(), 1), 완료기준_중앙=round(np.median(done_d), 1),
                     완료기준_60분내비율=round((done_d <= HORIZON_MIN).mean() * 100, 1),
                     KM_60분내탑승확률=round(p60 * 100, 1), KM_중앙=med, KM_RMST180=round(rmst, 1), KM_90분위=p90,
                     실제_60분내탑승=round(b[HORIZON_MIN] * 100, 1), 실제_60분내포기=round(c[HORIZON_MIN] * 100, 1)))
    if name in ("즉시", "특장차_즉시"):
        pd.DataFrame({"분": GRID, "KM_미탑승확률": S, "누적탑승": b, "누적포기": c}).to_csv(
            f"{OUT}/02_curve_{name}.csv", index=False, encoding="utf-8-sig")
tab = pd.DataFrame(rows); tab.to_csv(f"{OUT}/02_overall.csv", index=False, encoding="utf-8-sig")
print(tab.to_string())
t = tab.set_index("구분")
for k in ["즉시", "특장차_즉시", "임차택시_즉시"]:
    for col, key in [("완료기준_평균", "done_mean"), ("완료기준_60분내비율", "done_p60"), ("KM_60분내탑승확률", "km_p60"),
                     ("KM_RMST180", "km_rmst"), ("실제_60분내탑승", "cif_board60"), ("실제_60분내포기", "cif_cancel60"),
                     ("취소율", "cancel"), ("호출", "n"), ("KM_중앙", "km_median"), ("완료기준_중앙", "done_median")]:
        out[f"surv_{k}_{key}"] = float(t.loc[k, col])
# 완료 기준 평균 대비 포기 포함(RMST) 증가폭
out["surv_특장차_즉시_gap_min"] = round(out["surv_특장차_즉시_km_rmst"] - out["surv_특장차_즉시_done_mean"], 1)
out["surv_특장차_즉시_gap_pct"] = round(out["surv_특장차_즉시_gap_min"] / out["surv_특장차_즉시_done_mean"] * 100, 1)
# 포기한 사람이 그냥 흘려보낸 대기 시간(연간)
cc = base[base["취소"] & (base["T_취소"] >= 0)]
out["surv_cancel_waited_hours_total"] = round(float(cc["T_취소"].sum() / 60), 0)
out["surv_cancel_waited_median"] = round(float(cc["T_취소"].median()), 1)
out["surv_cancel_over30_n"] = int((cc["T_취소"] >= 30).sum())
out["surv_cancel_over60_n"] = int((cc["T_취소"] >= 60).sum())

# ---------- 시간대별 (특장차 즉시) ----------
sp = base[(~base["예약"]) & (base["차량구분"] == "특장차")]
hr = []
for h, d in sp.groupby("시"):
    T, E = prep(d)
    if len(T) < 500: continue
    S = km(T, E); p60, med, rmst, _ = summarize(S)
    hr.append(dict(시=h, 호출=len(T), 취소율=round((~E).mean() * 100, 1), 완료기준_평균=round(T[E].mean(), 1),
                   완료기준_60분내비율=round((T[E] <= 60).mean() * 100, 1),
                   KM_60분내탑승확률=round(p60 * 100, 1), KM_RMST180=round(rmst, 1)))
hr = pd.DataFrame(hr); hr.to_csv(f"{OUT}/02_hour.csv", index=False, encoding="utf-8-sig")
print(hr.to_string())
worst = hr.sort_values("KM_60분내탑승확률").iloc[0]
out["surv_hour_worst"] = int(worst["시"]); out["surv_hour_worst_p60"] = float(worst["KM_60분내탑승확률"])
best = hr.sort_values("KM_60분내탑승확률").iloc[-1]
out["surv_hour_best"] = int(best["시"]); out["surv_hour_best_p60"] = float(best["KM_60분내탑승확률"])
for h in (7, 8, 12, 13, 15, 16, 20, 21):
    if (hr["시"] == h).any():
        out[f"surv_hour{h}_p60"] = float(hr.loc[hr["시"] == h, "KM_60분내탑승확률"].iloc[0])
        out[f"surv_hour{h}_cancel"] = float(hr.loc[hr["시"] == h, "취소율"].iloc[0])
        out[f"surv_hour{h}_done_mean"] = float(hr.loc[hr["시"] == h, "완료기준_평균"].iloc[0])

# ---------- (d) 인내 분포: 취소까지 시간, 배차되면 중도절단 ----------
T = np.where(sp["취소"] & sp["배차일시"].isna(), sp["T_취소"], sp["T_배차"]).astype(float)
E = (sp["취소"] & sp["배차일시"].isna()).values
ok = ~np.isnan(T) & (T >= 0)
Sp = km(T[ok], E[ok])
pd.DataFrame({"분": GRID, "인내_생존": Sp}).to_csv(f"{OUT}/02_patience.csv", index=False, encoding="utf-8-sig")
out["patience_giveup_by30_pct"] = round((1 - Sp[30]) * 100, 1)
out["patience_giveup_by60_pct"] = round((1 - Sp[60]) * 100, 1)
out["patience_giveup_by90_pct"] = round((1 - Sp[90]) * 100, 1)

# ---------- (e) 자치구 순위 어긋남 ----------
spS = sp[sp["서울출발"]]
gu = []
for g, d in spS.groupby("출발구"):
    T, E = prep(d); S = km(T, E); p60, med, rmst, _ = summarize(S)
    gu.append(dict(자치구=g, 호출=len(T), 취소율=round((~E).mean() * 100, 1), 완료기준_평균=round(T[E].mean(), 1),
                   KM_60분내탑승확률=round(p60 * 100, 1), KM_RMST180=round(rmst, 1)))
gu = pd.DataFrame(gu)
gu["순위_완료기준"] = gu["완료기준_평균"].rank(ascending=False, method="min").astype(int)    # 1=가장 오래 기다림
gu["순위_포기포함"] = gu["KM_60분내탑승확률"].rank(ascending=True, method="min").astype(int)  # 1=60분 내 탈 확률 최저
gu["순위차"] = gu["순위_완료기준"] - gu["순위_포기포함"]
gu = gu.sort_values("순위_포기포함"); gu.to_csv(f"{OUT}/02_gu.csv", index=False, encoding="utf-8-sig")
print(gu.to_string())
out["gu_rank_spearman"] = round(float(spearmanr(gu["순위_완료기준"], gu["순위_포기포함"])[0]), 3)
out["gu_p60_min"] = float(gu["KM_60분내탑승확률"].min()); out["gu_p60_min_name"] = gu.iloc[0]["자치구"]
out["gu_p60_max"] = float(gu["KM_60분내탑승확률"].max()); out["gu_p60_max_name"] = gu.iloc[-1]["자치구"]
out["gu_max_rankshift"] = int(gu["순위차"].abs().max())

# ---------- 행정동 ----------
dg = []
for (g, dn), d in spS.groupby(["출발구", "출발동"]):
    if len(d) < 300: continue
    T, E = prep(d); S = km(T, E); p60, med, rmst, _ = summarize(S)
    dg.append(dict(자치구=g, 행정동=dn, 호출=len(T), 취소율=round((~E).mean() * 100, 1), 완료기준_평균=round(T[E].mean(), 1),
                   KM_60분내탑승확률=round(p60 * 100, 1)))
dg = pd.DataFrame(dg).sort_values("KM_60분내탑승확률")
dg["순위_완료기준"] = dg["완료기준_평균"].rank(ascending=False, method="min").astype(int)
dg["순위_포기포함"] = dg["KM_60분내탑승확률"].rank(ascending=True, method="min").astype(int)
dg.to_csv(f"{OUT}/02_dong.csv", index=False, encoding="utf-8-sig")
out["dong_n"] = int(len(dg))
out["dong_p60_p10"] = round(float(dg["KM_60분내탑승확률"].quantile(0.1)), 1)
out["dong_p60_p90"] = round(float(dg["KM_60분내탑승확률"].quantile(0.9)), 1)
out["dong_p60_min"] = float(dg["KM_60분내탑승확률"].min())
out["dong_p60_min_name"] = dg.iloc[0]["자치구"] + " " + dg.iloc[0]["행정동"]
out["dong_p60_max"] = float(dg["KM_60분내탑승확률"].max())
out["dong_p60_max_name"] = dg.iloc[-1]["자치구"] + " " + dg.iloc[-1]["행정동"]
top30_done = set(dg.nsmallest(30, "순위_완료기준")[["자치구", "행정동"]].apply(tuple, axis=1))
top30_km = set(dg.nsmallest(30, "순위_포기포함")[["자치구", "행정동"]].apply(tuple, axis=1))
out["dong_top30_overlap"] = len(top30_done & top30_km)
out["dong_top30_new"] = 30 - len(top30_done & top30_km)
out["dong_rank_spearman"] = round(float(spearmanr(dg["순위_완료기준"], dg["순위_포기포함"])[0]), 3)
save_summary(out)
print({k: v for k, v in out.items()})
