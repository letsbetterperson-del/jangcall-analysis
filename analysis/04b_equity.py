# -*- coding: utf-8 -*-
"""4b단계 — 형평 목표 배치(D안): 같은 55명×5시간을 '가장 나쁜 시간대(06~21시)'를 끌어올리는 쪽으로 고른다.
실행: python analysis/04b_equity.py   (04_sim.py 먼저 — η는 out/04_eta.csv를 그대로 쓴다)
목표 점수 = 06~21시 시간대별 60분 내 탑승 비율의 최솟값 + 0.2 × 하루 평균 (학습일), 평가는 검증일.
"""
import importlib, json, os, sys
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, save_summary
M = importlib.import_module("04_sim")


def score(r):
    return float(r["byh"].loc[6:21, "b60"].min()) + 0.2 * r["b60"]


if __name__ == "__main__":
    sp, pat, slots = M.load_inputs()
    days = M.build_days(sp, pat, slots)
    train, test = days[0::2], days[1::2]
    eta = pd.read_csv(os.path.join(OUT, "04_eta.csv"))["eta"].values
    with Pool(max(1, os.cpu_count() - 2), initializer=M._init, initargs=({"train": train, "test": test}, eta)) as pool:
        sched = []
        for it in range(M.ADD_DRIVERS // 5):
            res = pool.map(M._eval, [("train", "OPT", sched + [(st, 5)]) for st in M.CANDS])
            k = int(np.argmax([score(r) for r in res]))
            sched.append((M.CANDS[k], 5))
            print(f"  equity {it + 1}: {M.CANDS[k] // 60:02d}:{M.CANDS[k] % 60:02d} score={score(res[k]):.4f}", flush=True)
        r = pool.map(M._eval, [("test", "OPT", sched)])[0]
    w = r["byh"].loc[6:21, "b60"]
    out = {"sim_EQ_b60": round(r["b60"] * 100, 2), "sim_EQ_ab": round(r["ab"] * 100, 2),
           "sim_EQ_worst_b60": round(float(w.min()) * 100, 1), "sim_EQ_worst_hour": int(w.idxmin()),
           "sim_EQ_schedule": M.fmt_sched(sched)}
    s = json.load(open(os.path.join(OUT, "summary.json"), encoding="utf-8"))
    out["sim_EQ_gain_pp"] = round(out["sim_EQ_b60"] - s["sim_S0_b60"], 2)
    out["sim_EQ_gain_trips_per_day"] = round(out["sim_EQ_gain_pp"] / 100 * s["sim_test_requests_per_day"], 0)
    out["sim_EQ_ab_drop_per_day"] = round((s["sim_S0_ab"] - out["sim_EQ_ab"]) / 100 * s["sim_test_requests_per_day"], 0)
    byh = pd.read_csv(os.path.join(OUT, "04_scenarios_byhour.csv")).set_index("h")
    byh["EQ"] = r["byh"]["b60"].round(4)
    byh.to_csv(os.path.join(OUT, "04_scenarios_byhour.csv"), encoding="utf-8-sig")
    sc = pd.read_csv(os.path.join(OUT, "04_scenarios.csv"))
    sc = sc[sc["시나리오"] != "EQ"]
    sc.loc[len(sc)] = ["EQ", out["sim_EQ_b60"], out["sim_EQ_ab"], out["sim_EQ_worst_b60"], out["sim_EQ_worst_hour"]]
    sc.to_csv(os.path.join(OUT, "04_scenarios.csv"), index=False, encoding="utf-8-sig")
    json.dump({"schedule": out["sim_EQ_schedule"]}, open(os.path.join(OUT, "04_eq_schedule.json"), "w", encoding="utf-8"), ensure_ascii=False)
    save_summary(out); print(out)
