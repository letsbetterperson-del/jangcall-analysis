# -*- coding: utf-8 -*-
"""4단계 — 실제 호출 기록을 다시 흘리는 추적형(trace-driven) 시뮬레이터로 운전원 시간대 배치를 비교한다.
실행: python analysis/04_sim.py   (01·02·03 먼저. 16코어 약 15~30분)

모형 (특장차 = 휠체어 풀, 2025년 평일 244일 하루씩):
 - 호출: 그날 실제 호출 전부(대기 시계 t0 순). 실제로 탄 호출은 실제 '배차→승차(접근)'·'배차→하차(점유)' 시간을 그대로 쓴다.
   실제로 포기한 호출은 같은 시간대에 탄 호출의 (접근, 점유)를 무작위로 빌린다(고정 난수).
 - 인내: 실제 포기한 호출은 실제 포기 시각. 탄 호출은 02단계 인내 분포에서 '실제 배차까지 기다린 시간보다 길다'는 조건부로 뽑는다.
   배차 뒤 취소한 호출은 실제대로(차량 점유 후 미탑승) 둔다.
 - 공급: 10분마다 근무 차량 수 N(t). 기준 N은 03단계에서 그날 실제 '운행 중 차량'으로 역산한다
   (배차 대기 10건 이상 = 포화 → N=운행 중, 비포화 → N=운행 중 + 2).
 - 배차: 1분 단위. 예약 호출은 예정 시각이 되면 먼저(실제 07:00·08:00 일괄 배차 관찰), 즉시 호출은 절반은 먼저 온 순서,
   절반은 무작위(자동배차가 '가까운 차'를 고르는 효과의 대리 — 검증에서 순수 선착순보다 실제에 가까웠다).
   근무 차량이 줄면 운행 중인 차는 운행을 마친 뒤 빠진다.
 - 시간대별 유효 공급 계수 η(h): 차량 수만 세는 모형은 공간 마찰(차가 멀리 있음)을 모른다. 학습일에서 시간대별
   60분 내 탑승 비율이 실제와 맞도록 η를 보정하고, 검증일에서 맞는지 본다. 추가 운전원에도 같은 η를 곱한다.
 - 지표: 호출 대비 '60분 안에 탑승' 비율(대기 시계 t0 → 승차), 포기율, 06~21시 중 최저 시간대 값.
학습일 = 홀수 번째 평일(122일), 검증일 = 짝수 번째 평일(122일). 보정·최적화는 학습일, 보고 수치는 검증일.
시나리오(추가 운전원 55명 × 5시간 = 2026년 단시간 운전원 120→175명 증가분과 같은 운전시간 275시간/일):
 S1 출퇴근 집중: 07~12시 28명 + 14~19시 27명 (서울시 설명 '출퇴근 5시간조'를 가정해 배치, 시각은 가정)
 S2 16시간 균등: 같은 275 운전시간을 06~22시에 고르게(시간당 약 17.2대)
 OPT 시작시각 최적화: 05:00~19:00 30분 간격 시작 후보 중 5명씩 탐욕적으로 고름(학습일) → 검증일 평가
추가: 투입 운전시간(0~1,500시간/일)별 최적 배치 vs 균등 배치 곡선 → "몇 시간이 더 필요한가"
"""
import numpy as np, pandas as pd, os, heapq, json, sys
from multiprocessing import Pool
from common import CACHE, OUT, save_summary, holidays_2025

SLOT = 10
RNG_SEED = 20251231
PT_HOURS = 5
ADD_DRIVERS = 55
Q_OLD = 0.5            # 즉시 호출 중 '먼저 온 순서'로 배차하는 비율(나머지는 무작위) — 검증으로 고른 값
FLEET_CAP = 818        # 특장차 총 대수(서울시 2026-01-10 해명자료) — 근무 차량 상한


def load_inputs():
    base = pd.read_parquet(os.path.join(CACHE, "base.parquet"))
    sp = base[base["차량구분"] == "특장차"].copy()
    hol = holidays_2025()
    sp = sp[(sp["요일"] < 5) & ~sp["일자"].isin(hol)]
    pat = pd.read_csv(os.path.join(OUT, "02_patience.csv"))["인내_생존"].values
    slots = pd.read_parquet(os.path.join(OUT, "03_slots.parquet"))
    return sp, pat, slots


def build_days(sp, pat, slots):
    rng = np.random.default_rng(RNG_SEED)
    served = sp[sp["탑승"] & sp["접근분"].notna() & sp["점유분"].notna() & (sp["접근분"] >= 0) & (sp["점유분"] >= sp["접근분"])]
    pool = {h: g[["접근분", "점유분"]].values for h, g in served.groupby("시")}
    days = []
    for d, g in sp.groupby("일자"):
        g = g.sort_values("t0")
        t0 = ((g["t0"] - d).dt.total_seconds() / 60).values
        n = len(g)
        A = g["접근분"].values.astype(float); D = g["점유분"].values.astype(float)
        P = np.full(n, np.inf)
        board = g["탑승"].values; canc = g["취소"].values; disp = g["배차일시"].notna().values
        ab = canc & ~disp                                  # 배차 전 포기
        P[ab] = np.maximum(g["T_취소"].values[ab], 0)
        hrs = g["시"].values
        for i in np.where(ab | (board & (np.isnan(A) | np.isnan(D))))[0]:
            pr = pool[hrs[i]]; A[i], D[i] = pr[rng.integers(len(pr))]
        tb = np.clip(g["T_배차"].values, 0, 239)            # 탄 호출: 인내 > 실제 배차까지 기다린 시간
        for i in np.where(board)[0]:
            s_cond = pat[int(tb[i])] if not np.isnan(tb[i]) else 1.0
            u = rng.uniform(0, s_cond)
            k = int(np.argmax(pat <= u)) if (pat <= u).any() else -1
            P[i] = np.inf if k <= 0 else max(float(k), tb[i] + 1)
        ca = canc & disp                                   # 배차 후 취소: 배차~취소 점유, 탑승 없음
        D[ca] = np.clip(((g["취소일시"] - g["배차일시"]).dt.total_seconds() / 60).values[ca], 0, None)
        A[ca] = np.nan
        A = np.where(np.isnan(A) & ~ca, 15.0, A)
        D = np.clip(np.nan_to_num(D, nan=30.0), 1, 600)
        sd = slots[slots["일자"] == d].sort_values("슬롯")
        N = np.where(sd["포화"].values, sd["운행중"].values, sd["운행중"].values + 2)
        obs_b60 = board & (g["T_탑승"].values <= 60)
        days.append(dict(day=d, t0=t0, A=A, D=D, P=P, aftercancel=ca, res=g["예약"].values, N=N.astype(float),
                         hour=hrs, obs_b60=obs_b60, obs_ab=canc))
    return days


def simulate(day, N):
    """N: 길이 144(10분 슬롯) 유효 근무 차량 수. 반환: 60분 내 탑승 여부, 포기 여부 (호출별)."""
    from collections import deque
    rng = np.random.default_rng(int(day["day"].strftime("%Y%m%d")))
    t0, A, D, P, ca, res = day["t0"], day["A"], day["D"], day["P"], day["aftercancel"], day["res"]
    n = len(t0)
    assign = np.full(n, np.nan)
    busy = []
    qi = 0
    qr, qm = deque(), []
    for m in range(0, 24 * 60 + 240):
        while qi < n and t0[qi] <= m:
            (qr.append(qi) if res[qi] else qm.append(qi)); qi += 1
        while busy and busy[0] <= m:
            heapq.heappop(busy)
        cap = N[min(m // SLOT, 143)] if m < 1440 else N[143] + 50
        while (qr or qm) and len(busy) < cap:
            if qr:
                i = qr.popleft()
            else:
                j = 0 if rng.random() < Q_OLD else int(rng.integers(len(qm)))
                i = qm.pop(j)
            if m - t0[i] > P[i]:        # 이미 포기
                continue
            assign[i] = m
            heapq.heappush(busy, m + D[i])
        if qi >= n and not qr and not qm and m >= 1440:
            break
    wait_board = assign - t0 + A
    b60 = ~np.isnan(assign) & ~ca & (wait_board <= 60)
    ab = np.isnan(assign) | ca
    return b60, ab


def eff(N_raw, eta):
    return np.minimum(N_raw, FLEET_CAP) * np.repeat(eta, 6)


def add_shift(N, start_min, drivers, hours=PT_HOURS):
    N = N.copy()
    s = int(start_min // SLOT); e = int(min(144, (start_min + hours * 60) // SLOT))
    N[s:e] += drivers
    return N


def scenario_raw(day, name, extra=None):
    N = day["N"].astype(float)
    if name == "S0":
        return N
    if name == "S1":
        return add_shift(add_shift(N, 7 * 60, 28), 14 * 60, 27)
    if name == "S2":
        N2 = N.copy(); N2[36:132] += ADD_DRIVERS * PT_HOURS / 16.0
        return N2
    if name == "OPT":
        for st, k in extra:
            N = add_shift(N, st, k)
        return N
    if name == "EVEN":
        N2 = N.copy(); N2[36:132] += extra / 16.0
        return N2


G = {}


def _init(days_map, eta):
    G["days"] = days_map; G["eta"] = eta


def _eval(args):
    which, name, extra = args
    days = G["days"][which]; eta = G["eta"]
    b = []; a = []; h = []
    for d in days:
        b60, ab = simulate(d, eff(scenario_raw(d, name, extra), eta))
        b.append(b60); a.append(ab); h.append(d["hour"])
    b = np.concatenate(b); a = np.concatenate(a); h = np.concatenate(h)
    byh = pd.DataFrame({"h": h, "b60": b, "ab": a}).groupby("h").agg(b60=("b60", "mean"), ab=("ab", "mean"), n=("b60", "size"))
    return dict(b60=float(b.mean()), ab=float(a.mean()), byh=byh)


def _eval_eta(args):
    which, eta = args
    G["eta"] = eta
    return _eval((which, "S0", None))


CANDS = list(range(5 * 60, 19 * 60 + 1, 30))


def greedy(pool, n_batches, batch=5, start=None):
    sched = list(start or [])
    hist = []
    for it in range(n_batches):
        res = pool.map(_eval, [("train", "OPT", sched + [(st, batch)]) for st in CANDS])
        best = int(np.argmax([r["b60"] for r in res]))
        sched.append((CANDS[best], batch))
        hist.append((CANDS[best], res[best]["b60"]))
        print(f"  greedy {it + 1}/{n_batches}: start {CANDS[best] // 60:02d}:{CANDS[best] % 60:02d}  b60={res[best]['b60']:.4f}",
              flush=True)
    return sched, hist


def fmt_sched(sched):
    agg = {}
    for st, k in sched:
        agg[st] = agg.get(st, 0) + k
    return {f"{st // 60:02d}:{st % 60:02d}": k for st, k in sorted(agg.items())}


def obs_byhour(days):
    return pd.DataFrame({"h": np.concatenate([d["hour"] for d in days]),
                         "b60": np.concatenate([d["obs_b60"] for d in days]),
                         "ab": np.concatenate([d["obs_ab"] for d in days])}).groupby("h").mean()


if __name__ == "__main__":
    sp, pat, slots = load_inputs()
    days = build_days(sp, pat, slots)
    train = days[0::2]; test = days[1::2]
    chunks = {f"tr{k}": train[k::4] for k in range(4)}
    days_map = {"train": train, "test": test, **chunks}
    print("weekdays", len(days), "train", len(train), "test", len(test), flush=True)
    out = {}
    obs_tr, obs_te = obs_byhour(train), obs_byhour(test)
    eta = np.ones(24)
    NPROC = max(1, os.cpu_count() - 2)
    with Pool(NPROC, initializer=_init, initargs=(days_map, eta)) as pool:
        # 시간 순 좌표 탐색: 앞 시간대의 대기열이 뒤로 넘어가므로 0시부터 차례로 η(h)를 고르고 두 번 반복
        log = []
        GRIDV = np.round(np.arange(0.40, 1.61, 0.05), 2)
        for ps in range(2):
            for h in range(24):
                cands = []
                for v in GRIDV:
                    e = eta.copy(); e[h] = v; cands.append(e)
                res = pool.map(_eval_eta, [("train", e) for e in cands])
                def loss(r):
                    sb = r["byh"]["b60"].reindex(range(24))
                    l = abs(obs_tr["b60"].get(h, 0) - sb.get(h, 0))
                    if h < 23:
                        l += 0.5 * abs(obs_tr["b60"].get(h + 1, 0) - sb.get(h + 1, 0))
                    return l
                k = int(np.argmin([loss(r) for r in res]))
                eta = cands[k]
            r = pool.map(_eval_eta, [("train", eta)])[0]
            err = (obs_tr["b60"] - r["byh"]["b60"]).reindex(range(24)).fillna(0)
            mae = float(err.loc[6:23].abs().mean())
            log.append(dict(반복=ps, 시간대MAE_pp=round(mae * 100, 2)))
            print(f"  eta pass {ps}: hourly MAE {mae * 100:.2f}pp  eta={np.round(eta, 2).tolist()}", flush=True)
        pd.DataFrame(log).to_csv(f"{OUT}/04_eta_log.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"시": range(24), "eta": eta.round(3)}).to_csv(f"{OUT}/04_eta.csv", index=False, encoding="utf-8-sig")
    with Pool(NPROC, initializer=_init, initargs=(days_map, eta)) as pool:
        s0 = pool.map(_eval, [("test", "S0", None)])[0]   # 검증(검증일)
        val = pd.DataFrame({"실제_60분내탑승": obs_te["b60"], "모형_60분내탑승": s0["byh"]["b60"],
                            "실제_포기": obs_te["ab"], "모형_포기": s0["byh"]["ab"], "호출": s0["byh"]["n"]})
        val.round(4).to_csv(f"{OUT}/04_validation.csv", encoding="utf-8-sig")
        print(val.round(3).to_string(), flush=True)
        vv = val[val["호출"] > 1000]
        out["sim_obs_b60"] = round(float(np.concatenate([d["obs_b60"] for d in test]).mean()) * 100, 1)
        out["sim_obs_ab"] = round(float(np.concatenate([d["obs_ab"] for d in test]).mean()) * 100, 1)
        out["sim_s0_b60_test"] = round(s0["b60"] * 100, 1)
        out["sim_s0_ab_test"] = round(s0["ab"] * 100, 1)
        out["sim_val_hour_mae_pp"] = round(float((vv["실제_60분내탑승"] - vv["모형_60분내탑승"]).abs().mean()) * 100, 1)
        out["sim_val_hour_corr"] = round(float(np.corrcoef(vv["실제_60분내탑승"], vv["모형_60분내탑승"])[0, 1]), 3)
        out["sim_val_ab_mae_pp"] = round(float((vv["실제_포기"] - vv["모형_포기"]).abs().mean()) * 100, 1)
        print("greedy on train days", flush=True)        # 최적화(학습일)
        sched, hist = greedy(pool, ADD_DRIVERS // 5, 5)
        json.dump({"schedule": fmt_sched(sched), "history": hist}, open(f"{OUT}/04_opt_schedule.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        names = [("S0", None), ("S1", None), ("S2", None), ("OPT", sched)]   # 시나리오(검증일)
        resl = pool.map(_eval, [("test", nm, ex) for nm, ex in names])
        res = {nm: r for (nm, _), r in zip(names, resl)}
        rows = []
        for nm, m in res.items():
            w = m["byh"].loc[6:21, "b60"]
            rows.append(dict(시나리오=nm, 검증일_60분내탑승=round(m["b60"] * 100, 2), 검증일_포기=round(m["ab"] * 100, 2),
                             최저시간대_60분내탑승=round(float(w.min()) * 100, 1), 최저시간대=int(w.idxmin())))
        sc = pd.DataFrame(rows); sc.to_csv(f"{OUT}/04_scenarios.csv", index=False, encoding="utf-8-sig")
        print(sc.to_string(), flush=True)
        byh = pd.concat({k: v["byh"]["b60"] for k, v in res.items()}, axis=1)
        byh["실제"] = obs_te["b60"]
        byh.round(4).to_csv(f"{OUT}/04_scenarios_byhour.csv", encoding="utf-8-sig")
        per_day = sum(len(d["t0"]) for d in test) / len(test)
        t = sc.set_index("시나리오")
        for k in ["S0", "S1", "S2", "OPT"]:
            out[f"sim_{k}_b60"] = float(t.loc[k, "검증일_60분내탑승"]); out[f"sim_{k}_ab"] = float(t.loc[k, "검증일_포기"])
            out[f"sim_{k}_worst_b60"] = float(t.loc[k, "최저시간대_60분내탑승"]); out[f"sim_{k}_worst_hour"] = int(t.loc[k, "최저시간대"])
        for k in ["S1", "S2", "OPT"]:
            out[f"sim_{k}_gain_pp"] = round(out[f"sim_{k}_b60"] - out["sim_S0_b60"], 2)
            out[f"sim_{k}_gain_trips_per_day"] = round((out[f"sim_{k}_b60"] - out["sim_S0_b60"]) / 100 * per_day, 0)
            out[f"sim_{k}_ab_drop_per_day"] = round((out["sim_S0_ab"] - out[f"sim_{k}_ab"]) / 100 * per_day, 0)
        out["sim_opt_vs_s1_ratio"] = round(out["sim_OPT_gain_pp"] / out["sim_S1_gain_pp"], 2) if out["sim_S1_gain_pp"] > 0 else None
        out["sim_opt_vs_s2_ratio"] = round(out["sim_OPT_gain_pp"] / out["sim_S2_gain_pp"], 2) if out["sim_S2_gain_pp"] > 0 else None
        out["sim_test_requests_per_day"] = round(per_day, 0)
        out["sim_opt_schedule"] = fmt_sched(sched)
        out["sim_n_train"] = len(train); out["sim_n_test"] = len(test)
        print("curve", flush=True)                       # 투입량 곡선(검증일)
        curve = [dict(추가운전시간=0, 최적=out["sim_S0_b60"], 균등=out["sim_S0_b60"],
                      최적_최저시간대=out["sim_S0_worst_b60"], 균등_최저시간대=out["sim_S0_worst_b60"])]
        sched_c = []
        for step in range(1, 7):
            sched_c, _ = greedy(pool, 5, 10, start=sched_c)
            hrs = sum(k for _, k in sched_c) * PT_HOURS
            m_opt, m_even = pool.map(_eval, [("test", "OPT", sched_c), ("test", "EVEN", hrs)])
            curve.append(dict(추가운전시간=hrs, 최적=round(m_opt["b60"] * 100, 2), 균등=round(m_even["b60"] * 100, 2),
                              최적_최저시간대=round(float(m_opt["byh"].loc[6:21, "b60"].min()) * 100, 1),
                              균등_최저시간대=round(float(m_even["byh"].loc[6:21, "b60"].min()) * 100, 1)))
            print(curve[-1], flush=True)
        cv = pd.DataFrame(curve); cv.to_csv(f"{OUT}/04_curve.csv", index=False, encoding="utf-8-sig")
        json.dump({"schedule": fmt_sched(sched_c)}, open(f"{OUT}/04_curve_schedule.json", "w", encoding="utf-8"), ensure_ascii=False)

        def need(col, target):
            ok = cv[cv[col] >= target]
            return int(ok["추가운전시간"].iloc[0]) if len(ok) else None
        for tg in (85, 90):
            out[f"sim_curve_need{tg}_opt_h"] = need("최적", tg); out[f"sim_curve_need{tg}_even_h"] = need("균등", tg)
        out["sim_curve_max_h"] = int(cv["추가운전시간"].max())
        out["sim_curve_max_opt"] = float(cv["최적"].iloc[-1]); out["sim_curve_max_even"] = float(cv["균등"].iloc[-1])
        out["sim_curve_max_opt_worst"] = float(cv["최적_최저시간대"].iloc[-1])
        out["sim_curve_max_even_worst"] = float(cv["균등_최저시간대"].iloc[-1])
    out["sim_eta_min"] = round(float(eta.min()), 2); out["sim_eta_max"] = round(float(eta.max()), 2)
    save_summary(out)
    print(out)
