# -*- coding: utf-8 -*-
"""1단계 — 정제·기본 통계. 실행: python analysis/01_prep.py
정의(설정값은 common.py):
- 즉시 호출: 접수→예정 10분 미만. 예약 호출: 10분 이상. 대기 시계 t0 = 즉시는 접수, 예약은 예정 시각
- 탑승: 승차일시 있음 / 취소: 취소일시 있고 승차 없음 / 둘 다 없음: 상태 불명(제외)
- 포기(abandonment) 분석 대상: 즉시 호출 + (예약 호출 중 예정 시각 이후 취소)
  · 예약 호출을 예정 시각 전에 취소한 건은 '일정 변경'으로 보고 포기에서 뺀다
산출: out/01_basic.csv, summary.json(basic_*)
"""
import numpy as np, pandas as pd
from common import load_trips, save_summary, OUT

df = load_trips()
n = len(df)
unk = ~df["탑승"] & ~df["취소"]
pre_cancel = df["예약"] & df["취소"] & (df["T_취소"] < 0)
base = df[~unk & ~pre_cancel].copy()

out = {}
out["basic_n_all"] = int(n)
out["basic_n_unknown"] = int(unk.sum())
out["basic_n_reserve_precancel"] = int(pre_cancel.sum())
out["basic_n_analysis"] = int(len(base))
out["basic_share_immediate"] = round(float((~df["예약"]).mean()) * 100, 1)
out["basic_cancel_rate_all"] = round(float(df["취소"].mean()) * 100, 1)
out["basic_cancel_rate_analysis"] = round(float(base["취소"].mean()) * 100, 1)
out["basic_n_cancel_analysis"] = int(base["취소"].sum())
for pool in ["특장차", "임차택시"]:
    b = base[base["차량구분"] == pool]
    out[f"basic_n_{pool}"] = int(len(b))
    out[f"basic_cancel_{pool}"] = round(float(b["취소"].mean()) * 100, 1)
# 배차 전/후 취소
c = base[base["취소"]]
out["basic_cancel_before_dispatch_pct"] = round(float(c["배차일시"].isna().mean()) * 100, 1)
out["basic_cancel_after_dispatch_n"] = int(c["배차일시"].notna().sum())
# 완료 기준 대기(공식 집계 방식 재현): 탑승 건만, t0→승차
done = base[base["탑승"] & (base["T_탑승"] >= 0)]
out["basic_wait_done_mean"] = round(float(done["T_탑승"].mean()), 1)
out["basic_wait_done_median"] = round(float(done["T_탑승"].median()), 1)
imm = done[~done["예약"]]
out["basic_wait_done_imm_mean"] = round(float(imm["T_탑승"].mean()), 1)
out["basic_wait_done_imm_median"] = round(float(imm["T_탑승"].median()), 1)
out["basic_days"] = int(base["일자"].nunique())
out["basic_daily_requests"] = round(len(base) / base["일자"].nunique(), 0)
out["basic_seoul_origin_pct"] = round(float(base["서울출발"].mean()) * 100, 2)
out["basic_n_dong"] = int(base.loc[base["서울출발"], ["출발구", "출발동"]].drop_duplicates().shape[0])
out["basic_km_total"] = round(float(done["승차거리_km"].sum()), 0)
out["basic_trip_km_median"] = round(float(done["승차거리_km"].median()), 2)
# 장애유형별
g = base.groupby("장애유형").agg(n=("취소", "size"), 취소율=("취소", "mean")).sort_values("n", ascending=False)
g["취소율"] = (g["취소율"] * 100).round(1)
g.head(10).to_csv(f"{OUT}/01_disability.csv", encoding="utf-8-sig")
for k in ["자폐", "지적", "뇌병", "지체", "신장"]:
    out[f"basic_cancel_type_{k}"] = float(g.loc[k, "취소율"])
# 시간대별 기본표
h = base.groupby("시").agg(호출=("취소", "size"), 취소율=("취소", "mean"))
h["완료기준_대기평균"] = done.groupby("시")["T_탑승"].mean()
h["취소율"] = (h["취소율"] * 100).round(1)
h.round(1).to_csv(f"{OUT}/01_basic_hour.csv", encoding="utf-8-sig")
base.to_parquet(f"{OUT}/../../data/raw/_cache/base.parquet")
save_summary(out)
print(pd.Series(out))
