# -*- coding: utf-8 -*-
"""4c단계 — 절충안 탐색: 오전조 시작 시각·인원 × 오후조 시작 시각 격자(55명 고정)를 학습일에서 평가하고,
'하루 평균'과 '가장 나쁜 시간대'를 둘 다 A안 이상으로 만드는 조합 중 평균 최고를 검증일에서 보고한다(E안).
실행: python analysis/04c_hybrid.py   (04_sim.py 먼저)
"""
import importlib, json, os, sys, itertools
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, save_summary
M = importlib.import_module("04_sim")

if __name__ == "__main__":
    sp, pat, slots = M.load_inputs()
    days = M.build_days(sp, pat, slots)
    train, test = days[0::2], days[1::2]
    eta = pd.read_csv(os.path.join(OUT, "04_eta.csv"))["eta"].values
    AM = [330, 360, 390, 420]; PM = [690, 720, 750, 780, 840]; NAM = [15, 20, 25, 30]
    grid = [((a, n), (p, 55 - n)) for a, p, n in itertools.product(AM, PM, NAM)]
    with Pool(max(1, os.cpu_count() - 2), initializer=M._init, initargs=({"train": train, "test": test}, eta)) as pool:
        res = pool.map(M._eval, [("train", "OPT", [g[0], g[1]]) for g in grid])
        a_tr = pool.map(M._eval, [("train", "S1", None)])[0]
        rows = []
        for g, r in zip(grid, res):
            rows.append(dict(오전시작=f"{g[0][0] // 60:02d}:{g[0][0] % 60:02d}", 오전인원=g[0][1],
                             오후시작=f"{g[1][0] // 60:02d}:{g[1][0] % 60:02d}", 오후인원=g[1][1],
                             학습_평균=round(r["b60"] * 100, 2), 학습_최저=round(float(r["byh"].loc[6:21, "b60"].min()) * 100, 1)))
        tab = pd.DataFrame(rows)
        a_mean, a_min = a_tr["b60"] * 100, float(a_tr["byh"].loc[6:21, "b60"].min()) * 100
        ok = tab[(tab["학습_평균"] >= a_mean) & (tab["학습_최저"] >= a_min)].sort_values("학습_평균", ascending=False)
        tab.sort_values("학습_평균", ascending=False).to_csv(os.path.join(OUT, "04c_hybrid_grid.csv"), index=False, encoding="utf-8-sig")
        best = ok.iloc[0] if len(ok) else tab.sort_values("학습_평균", ascending=False).iloc[0]
        i = tab.index.get_loc(best.name); g = grid[i]
        r = pool.map(M._eval, [("test", "OPT", [g[0], g[1]])])[0]
    w = r["byh"].loc[6:21, "b60"]
    s = json.load(open(os.path.join(OUT, "summary.json"), encoding="utf-8"))
    out = {"sim_HY_b60": round(r["b60"] * 100, 2), "sim_HY_ab": round(r["ab"] * 100, 2),
           "sim_HY_worst_b60": round(float(w.min()) * 100, 1), "sim_HY_worst_hour": int(w.idxmin()),
           "sim_HY_am_start": best["오전시작"], "sim_HY_am_n": int(best["오전인원"]),
           "sim_HY_pm_start": best["오후시작"], "sim_HY_pm_n": int(best["오후인원"]),
           "sim_HY_n_ok_train": int(len(ok)), "sim_HY_grid_n": int(len(tab)),
           "sim_A_train_b60": round(a_mean, 2), "sim_A_train_worst": round(a_min, 1)}
    out["sim_HY_gain_pp"] = round(out["sim_HY_b60"] - s["sim_S0_b60"], 2)
    out["sim_HY_gain_trips_per_day"] = round(out["sim_HY_gain_pp"] / 100 * s["sim_test_requests_per_day"], 0)
    out["sim_HY_ab_drop_per_day"] = round((s["sim_S0_ab"] - out["sim_HY_ab"]) / 100 * s["sim_test_requests_per_day"], 0)
    out["sim_HY_vs_A_pp"] = round(out["sim_HY_b60"] - s["sim_S1_b60"], 2)
    byh = pd.read_csv(os.path.join(OUT, "04_scenarios_byhour.csv")).set_index("h")
    byh["HY"] = r["byh"]["b60"].round(4); byh.to_csv(os.path.join(OUT, "04_scenarios_byhour.csv"), encoding="utf-8-sig")
    sc = pd.read_csv(os.path.join(OUT, "04_scenarios.csv")); sc = sc[sc["시나리오"] != "HY"]
    sc.loc[len(sc)] = ["HY", out["sim_HY_b60"], out["sim_HY_ab"], out["sim_HY_worst_b60"], out["sim_HY_worst_hour"]]
    sc.to_csv(os.path.join(OUT, "04_scenarios.csv"), index=False, encoding="utf-8-sig")
    save_summary(out); print(out); print(ok.head(8).to_string())
