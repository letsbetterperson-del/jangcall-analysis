# -*- coding: utf-8 -*-
"""3단계 — 호출 기록에서 '지금 움직이는 차량 수'와 '배차를 기다리는 호출 수'를 10분 단위로 다시 만든다.
실행: python analysis/03_supply.py   (01_prep.py 먼저)

공개 자료에는 차량 ID·운전원 근무표가 없다. 대신
 - 운행 중 차량 B(t) = 배차 시각 ≤ t < 하차(배차 후 취소면 취소) 시각인 호출 수
 - 배차 대기 Q(t)   = 대기 시계 시작(t0) ≤ t < min(배차, 취소) 인 호출 수
대기 호출이 쌓여 있는 순간(Q ≥ 10)에는 쉬는 차가 없다고 볼 수 있으므로 B(t) ≈ 그 시각 근무 중 차량 수.
→ 특장차(휠체어) 풀의 평일 근무 차량 곡선을 역산한다. 시뮬레이터(04)의 기준 공급 곡선이 된다.
산출: out/03_slots.parquet(일×10분), out/03_profile.csv(평일 10분 평균), summary.json(supply_*)
"""
import numpy as np, pandas as pd, os
from common import CACHE, OUT, save_summary, holidays_2025

base = pd.read_parquet(os.path.join(CACHE, "base.parquet"))
sp = base[base["차량구분"] == "특장차"].copy()
SAT_Q = 10                                           # 포화 판정: 배차 대기 호출 10건 이상(설정값)

grid = pd.date_range("2025-01-01", "2026-01-01", freq="10min", inclusive="left")
grid1 = pd.date_range("2025-01-01", "2026-01-01", freq="1min", inclusive="left")   # 1분 격자로 세고 10분 슬롯으로 묶는다
g = grid1.values.astype("datetime64[ns]").astype(np.int64)
g10 = grid.values.astype("datetime64[ns]").astype(np.int64)


def count_active(start, end):
    s = np.sort(start[~np.isnan(start)]); e = np.sort(end[~np.isnan(end)])
    return np.searchsorted(s, g, side="right") - np.searchsorted(e, g, side="right")


ns = lambda c: sp[c].values.astype("datetime64[ns]").astype(np.int64).astype(float)
disp, drop, canc, t0 = ns("배차일시"), ns("하차일시"), ns("취소일시"), ns("t0")
NaT = pd.NaT.value
for a in (disp, drop, canc, t0):
    a[a == NaT] = np.nan
# 운행 중: 배차~하차, 배차 후 취소는 배차~취소
b_end = np.where(~np.isnan(drop), drop, canc)
ok = ~np.isnan(disp) & ~np.isnan(b_end) & (b_end >= disp)
B = count_active(np.where(ok, disp, np.nan), np.where(ok, b_end, np.nan))
# 배차 대기: t0 ~ min(배차, 취소)
q_end = np.fmin(disp, canc)
okq = ~np.isnan(q_end) & (q_end >= t0)
Q = count_active(np.where(okq, t0, np.nan), np.where(okq, q_end, np.nan))
# 10분 안 신규 호출(대기 시계 시작 기준)
arr = np.histogram(t0[~np.isnan(t0)], bins=np.append(g10, g10[-1] + 600e9))[0]
B10 = B.reshape(-1, 10); Q10 = Q.reshape(-1, 10)
# 운행중 = 슬롯 안 1분 값의 최댓값(그 10분 동안 동시에 움직인 최대 차량 수), 배차대기 = 슬롯 안 최솟값(10분 내내 쌓여 있던 호출)
slots = pd.DataFrame({"t": grid, "운행중": B10.max(1), "운행중_평균": B10.mean(1), "배차대기": Q10.min(1),
                      "배차대기_평균": Q10.mean(1), "신규호출": arr})
slots["일자"] = slots["t"].dt.normalize(); slots["슬롯"] = slots["t"].dt.hour * 6 + slots["t"].dt.minute // 10
hol = holidays_2025()
slots["평일"] = (slots["t"].dt.dayofweek < 5) & ~slots["일자"].isin(hol)
slots["포화"] = slots["배차대기"] >= SAT_Q
slots.to_parquet(f"{OUT}/03_slots.parquet")

wd = slots[slots["평일"]]
prof = wd.groupby("슬롯").agg(운행중_평균=("운행중_평균", "mean"), 운행중_최대평균=("운행중", "mean"), 운행중_p90=("운행중", lambda s: s.quantile(.9)),
                             배차대기_평균=("배차대기_평균", "mean"), 신규호출_평균=("신규호출", "mean"), 포화비율=("포화", "mean"))
prof["포화시_운행중"] = wd[wd["포화"]].groupby("슬롯")["운행중"].median()
prof["시각"] = [f"{s // 6:02d}:{(s % 6) * 10:02d}" for s in prof.index]
prof.round(2).to_csv(f"{OUT}/03_profile.csv", encoding="utf-8-sig")
print(prof.iloc[::3].round(1).to_string())

out = {}
out["supply_sat_threshold"] = SAT_Q
out["supply_weekdays"] = int(wd["일자"].nunique())
out["supply_peak_busy_mean"] = round(float(prof["운행중_평균"].max()), 0)
out["supply_peak_busy_time"] = prof["운행중_평균"].idxmax()
out["supply_peak_busy_time_str"] = prof.loc[prof["운행중_평균"].idxmax(), "시각"]
out["supply_max_busy_year"] = int(slots["운행중"].max())
out["supply_queue_peak_mean"] = round(float(prof["배차대기_평균"].max()), 0)
out["supply_queue_peak_time_str"] = prof.loc[prof["배차대기_평균"].idxmax(), "시각"]
for hhmm in ["06:30", "07:00", "07:30", "08:00", "09:00", "12:00", "14:00", "15:00", "16:00", "16:30", "17:00", "20:00"]:
    s = int(hhmm[:2]) * 6 + int(hhmm[3:]) // 10
    k = hhmm.replace(":", "")
    out[f"supply_busy_{k}"] = round(float(prof.loc[s, "운행중_평균"]), 0)
    out[f"supply_queue_{k}"] = round(float(prof.loc[s, "배차대기_평균"]), 0)
    out[f"supply_arr_{k}"] = round(float(prof.loc[s, "신규호출_평균"]), 1)
# 14:30 → 16:30 운행중 감소폭 (교대 공백)
a, b = int(14.5 * 6), int(16.5 * 6)
out["supply_drop_1430_1630"] = round(float(prof.loc[a:b, "운행중_평균"].max() - prof.loc[b, "운행중_평균"]), 0)
# 포화 시간 비율(평일 06~22시)
day = wd[(wd["슬롯"] >= 36) & (wd["슬롯"] < 132)]
out["supply_saturated_share_0622"] = round(float(day["포화"].mean()) * 100, 1)
# 차량 대비 동시 운행 (차량정보 698대 = 공단 관용 차량)
out["supply_fleet_public"] = 698
out["supply_peak_util_pct"] = round(out["supply_peak_busy_mean"] / 698 * 100, 1)
save_summary(out)
print(out)
