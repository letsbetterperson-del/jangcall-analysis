# -*- coding: utf-8 -*-
"""8단계 — 지역 격차의 원인 점검: 자치구별 차고지 주차면 수 vs 빈 차 접근 시간 vs 60분 내 탑승 확률.
실행: python analysis/08_garage.py   (02 먼저)
차고지 정보(44곳·주차대수·주소)의 주소 앞 '구'로 자치구에 붙인다. 접근 시간 = 배차→승차(빈 차가 오는 데 걸린 시간).
"""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
from common import CACHE, OUT, RAW, save_summary, SEOUL_GU

gar = pd.read_csv(os.path.join(RAW, "서울시설공단_장애인콜택시 차고지 정보_20260714.csv"), encoding="utf-8-sig")
gar.columns = [c.replace("﻿", "") for c in gar.columns]
gar["자치구"] = gar["상세주소"].str.split().str[0]
gar["주차대수"] = pd.to_numeric(gar["주차대수"], errors="coerce")
g1 = gar.groupby("자치구")["주차대수"].sum().reindex(SEOUL_GU).fillna(0)
base = pd.read_parquet(os.path.join(CACHE, "base.parquet"))
sp = base[(~base["예약"]) & (base["차량구분"] == "특장차") & base["서울출발"]]
appr = sp[sp["탑승"] & (sp["접근분"] >= 0) & (sp["접근분"] < 180)].groupby("출발구")["접근분"].median()
calls = sp.groupby("출발구").size()
gu = pd.read_csv(os.path.join(OUT, "02_gu.csv")).set_index("자치구")
t = pd.DataFrame({"차고지_주차면": g1, "호출": calls, "접근_중앙분": appr, "60분내탑승확률": gu["KM_60분내탑승확률"]})
t["호출1만건당_주차면"] = t["차고지_주차면"] / t["호출"] * 10000
t = t.sort_values("60분내탑승확률"); t.round(2).to_csv(f"{OUT}/08_garage_gu.csv", encoding="utf-8-sig")
print(t.round(1).to_string())
out = {
    "garage_n": int(len(gar)), "garage_spaces": int(gar["주차대수"].sum()),
    "garage_rho_spaces_p60": round(float(spearmanr(t["호출1만건당_주차면"], t["60분내탑승확률"])[0]), 2),
    "garage_rho_approach_p60": round(float(spearmanr(t["접근_중앙분"], t["60분내탑승확률"])[0]), 2),
    "garage_rho_spaces_approach": round(float(spearmanr(t["호출1만건당_주차면"], t["접근_중앙분"])[0]), 2),
    "garage_gu_zero": int((t["차고지_주차면"] == 0).sum()),
    "garage_approach_max_gu": t["접근_중앙분"].idxmax(), "garage_approach_max": round(float(t["접근_중앙분"].max()), 1),
    "garage_approach_min_gu": t["접근_중앙분"].idxmin(), "garage_approach_min": round(float(t["접근_중앙분"].min()), 1),
    "garage_seocho_spaces": int(t.loc["서초구", "차고지_주차면"]), "garage_nowon_spaces": int(t.loc["노원구", "차고지_주차면"]),
    "garage_south4_spaces": int(t.loc[["서초구", "강남구", "관악구", "금천구"], "차고지_주차면"].sum()),
    "garage_northeast4_spaces": int(t.loc[["노원구", "도봉구", "중랑구", "강북구"], "차고지_주차면"].sum()),
}
save_summary(out); print(out)
