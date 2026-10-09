# -*- coding: utf-8 -*-
"""4d단계 — 투입량 곡선·배치안 비교에서 보고서용 파생 값(목표 도달 운전시간, A안 대비 건수)을 계산한다.
실행: python analysis/04d_curve_keys.py   (04_sim.py, 04c_hybrid.py 먼저)"""
import json, os
import pandas as pd
from common import OUT, save_summary
s = json.load(open(os.path.join(OUT, "summary.json"), encoding="utf-8"))
cv = pd.read_csv(os.path.join(OUT, "04_curve.csv"))
def need(col, t):
    ok = cv[cv[col] >= t]; return int(ok["추가운전시간"].iloc[0]) if len(ok) else None
out = {"curve_need_mean90_opt_h": need("최적", 90), "curve_need_mean90_even_h": need("균등", 90),
       "curve_need_worst85_opt_h": need("최적_최저시간대", 85), "curve_need_worst85_even_h": need("균등_최저시간대", 85),
       "curve_1500_even_worst": float(cv["균등_최저시간대"].iloc[-1]), "curve_1500_opt_worst": float(cv["최적_최저시간대"].iloc[-1]),
       "curve_1000_opt_worst": float(cv.loc[cv["추가운전시간"] == 1000, "최적_최저시간대"].iloc[0]),
       "curve_1000_even_worst": float(cv.loc[cv["추가운전시간"] == 1000, "균등_최저시간대"].iloc[0]),
       "curve_750_opt": float(cv.loc[cv["추가운전시간"] == 750, "최적"].iloc[0]),
       "curve_1000_even": float(cv.loc[cv["추가운전시간"] == 1000, "균등"].iloc[0]),
       "curve_1000_drivers": 1000 // 5, "curve_750_drivers": 750 // 5}
per = s["sim_test_requests_per_day"]
out["sim_HY_vs_A_trips_per_day"] = round(s["sim_HY_vs_A_pp"] / 100 * per, 0)
out["sim_A_vs_B_pp"] = round(s["sim_S1_b60"] - s["sim_S2_b60"], 2)
out["sim_A_vs_B_trips_per_day"] = round(out["sim_A_vs_B_pp"] / 100 * per, 0)
out["sim_S1_gain_pp_r"] = round(s["sim_S1_gain_pp"], 1)
b = pd.read_csv(os.path.join(OUT, "04_scenarios_byhour.csv")).set_index("h") * 100
for k in ["S0", "S1", "S2", "HY"]:
    for h in (7, 8, 15, 16, 17, 20, 21):
        out[f"byh_{k}_{h}"] = round(float(b.loc[h, k]), 1)
save_summary(out); print(out)

# 절충안 격자 민감도(학습일): 오후조·오전조 시작 시각별 최고 하루 평균
g = pd.read_csv(os.path.join(OUT, "04c_hybrid_grid.csv"))
pm = g.groupby("오후시작")["학습_평균"].max(); am = g.groupby("오전시작")["학습_평균"].max()
sens = {"grid_pm1230_best": float(pm["12:30"]), "grid_pm1400_best": float(pm["14:00"]),
        "grid_pm_gain_pp": round(float(pm["12:30"] - pm["14:00"]), 2),
        "grid_am_range_pp": round(float(am.max() - am.min()), 2),
        "grid_am0530_best": float(am["05:30"]), "grid_am0700_best": float(am["07:00"])}
save_summary(sens); print(sens)
