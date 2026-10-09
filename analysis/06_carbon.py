# -*- coding: utf-8 -*-
"""6단계 — 특장차 운행 탄소: 연간 운행거리·빈 차 접근·배차 후 취소로 버려진 접근, 유종 구성.
실행: python analysis/06_carbon.py   (01 먼저)
- 승객 탑승 거리 = 탑승내역 '승차거리'(m) 합계 [실측]
- 빈 차 접근 거리 = 배차→승차 시간 × 탑승 구간 평균 속도(승차거리 ÷ 승차시간의 중앙값) [추정: 접근 구간도 같은 속도라고 가정]
- 배차 후 취소로 버려진 접근 = 배차→취소 시간 × 같은 속도 [추정]
- 유종 구성 = 차량정보 698대(LPG·휘발유·경유) [실측]
- 연비·배출계수는 가정값(범위로 민감도): 연비 LPG 6.6 km/L(스타리아 LPG 3.5 복합, 판매 제원), 휘발유 8 km/L, 경유 10 km/L(가정)
  배출계수 kgCO2/L: LPG 1.5~1.8, 휘발유 2.1~2.3, 경유 2.6~2.7 (국제 기본값 범위)
"""
import os, numpy as np, pandas as pd
from common import CACHE, OUT, RAW, save_summary

base = pd.read_parquet(os.path.join(CACHE, "base.parquet"))
sp = base[base["차량구분"] == "특장차"]
veh = pd.read_csv(os.path.join(RAW, "서울시설공단_장애인콜택시 차량정보_20260806.csv"), encoding="cp949")
mix = veh["유종"].value_counts()
share = (mix / mix.sum()).to_dict()

done = sp[sp["탑승"] & (sp["승차분"] > 3) & (sp["승차거리_km"] > 0.3)]
speed = float((done["승차거리_km"] / (done["승차분"] / 60)).median())          # km/h
km_ride = float(sp.loc[sp["탑승"], "승차거리_km"].sum())
appr = sp.loc[sp["탑승"] & (sp["접근분"] >= 0) & (sp["접근분"] < 180), "접근분"]
km_appr = float(appr.sum() / 60 * speed)
ca = sp[sp["취소"] & sp["배차일시"].notna()]
w = ((ca["취소일시"] - ca["배차일시"]).dt.total_seconds() / 60).clip(0, 180)
km_waste = float(w.sum() / 60 * speed)
km_total = km_ride + km_appr + km_waste

FE = {"LPG": 6.6, "휘발유": 8.0, "경유": 10.0}                  # km/L (가정)
EF_LO = {"LPG": 1.5, "휘발유": 2.1, "경유": 2.6}; EF_HI = {"LPG": 1.8, "휘발유": 2.3, "경유": 2.7}
def co2(km, ef):
    return sum(km * share.get(f, 0) / FE[f] * ef[f] for f in FE) / 1000          # tCO2
out = dict(
    carbon_speed_kmh=round(speed, 1), carbon_km_ride=round(km_ride, 0), carbon_km_approach=round(km_appr, 0),
    carbon_km_waste=round(km_waste, 0), carbon_km_total=round(km_total, 0),
    carbon_approach_share_pct=round(km_appr / km_total * 100, 1), carbon_waste_share_pct=round(km_waste / km_total * 100, 2),
    carbon_n_aftercancel=int(len(ca)), carbon_aftercancel_median_min=round(float(w.median()), 1),
    carbon_fleet_lpg=int(mix.get("LPG", 0)), carbon_fleet_gasoline=int(mix.get("휘발유", 0)), carbon_fleet_diesel=int(mix.get("경유", 0)),
    carbon_t_total_lo=round(co2(km_total, EF_LO), 0), carbon_t_total_hi=round(co2(km_total, EF_HI), 0),
    carbon_t_waste_lo=round(co2(km_waste, EF_LO), 0), carbon_t_waste_hi=round(co2(km_waste, EF_HI), 0),
    carbon_t_approach_lo=round(co2(km_appr, EF_LO), 0), carbon_t_approach_hi=round(co2(km_appr, EF_HI), 0),
)
# 접근 시간의 시간대 차이(빈 차가 멀리서 오는 정도)
h = sp[sp["탑승"] & (sp["접근분"] >= 0) & (sp["접근분"] < 180)].groupby("시")["접근분"].median()
out["carbon_approach_med_peak16"] = round(float(h.loc[16]), 1); out["carbon_approach_med_12"] = round(float(h.loc[12]), 1)
out["carbon_approach_med_all"] = round(float(appr.median()), 1)
h.round(1).to_csv(f"{OUT}/06_approach_hour.csv", encoding="utf-8-sig")
save_summary(out); print(out)
