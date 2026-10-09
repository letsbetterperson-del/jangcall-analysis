# -*- coding: utf-8 -*-
"""5단계 — 다음 날 출퇴근 시간대 호출 수 예측(머신러닝)과 '수요 많은 날' 사전 탐지.
실행: python analysis/05_forecast.py   (01 먼저)

왜: 단시간 운전원 근무표를 요일마다 같게 두면 수요가 몰리는 날에 모자라고 한산한 날에 남는다.
전날 오후에 다음 날 07~10시·14~18시 호출 수를 예측하면, 예비 단시간조(호출형 근무)를 그날만 부를 수 있다.
방법: 특장차 호출을 일×시간대 묶음(오전 07~10시, 오후 14~18시)으로 센다.
 특징 = 요일, 월, 공휴일·연휴 전후, 기온(최고·최저), 강수량, 적설, 1주 전 같은 요일 값, 직전 4주 같은 요일 평균
 (기상은 Open-Meteo 과거 기상 재분석값 — 실제 운영에서는 기상청 단기예보로 바꾼다)
 모형 = 그래디언트 부스팅 회귀(scikit-learn HistGradientBoostingRegressor), 학습 1~9월 / 평가 10~12월(시간 순 분할)
 비교 기준 = 1주 전 같은 요일 값(계절 단순 예측), 직전 4주 평균
 '수요 많은 날' = 평가 기간 상위 20% 날. 모형 예측 상위 20%로 맞힌 비율(정밀도·재현율)
 그날 실제 포기율: 수요 많은 날 vs 나머지
"""
import json, os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from common import CACHE, OUT, RAW, save_summary, holidays_2025

base = pd.read_parquet(os.path.join(CACHE, "base.parquet"))
sp = base[base["차량구분"] == "특장차"]
hol = holidays_2025()
W = {"오전": (7, 10), "오후": (14, 18)}

met = json.load(open(os.path.join(RAW, "open-meteo_seoul_2025_daily.json"), encoding="utf-8"))["daily"]
wx = pd.DataFrame(met).rename(columns={"time": "일자"}); wx["일자"] = pd.to_datetime(wx["일자"])

days = pd.DataFrame({"일자": pd.date_range("2025-01-01", "2025-12-31")})
for k, (a, b) in W.items():
    s = sp[(sp["시"] >= a) & (sp["시"] < b)]
    c = s.groupby("일자").agg(n=("t0", "size"), ab=("취소", "mean"))
    days = days.merge(c.rename(columns={"n": f"호출_{k}", "ab": f"포기율_{k}"}), left_on="일자", right_index=True, how="left")
days = days.merge(wx, on="일자", how="left")
days["요일"] = days["일자"].dt.dayofweek; days["월"] = days["일자"].dt.month
days["공휴일"] = days["일자"].isin(hol).astype(int)
days["연휴전날"] = days["일자"].shift(-1).isin(hol).astype(int)
days["연휴다음날"] = days["일자"].shift(1).isin(hol).astype(int)
days["일련"] = np.arange(len(days))
rows = []; out = {}
for k in W:
    y = f"호출_{k}"
    days[f"lag7_{k}"] = days[y].shift(7)
    days[f"ma4_{k}"] = (days[y].shift(7) + days[y].shift(14) + days[y].shift(21) + days[y].shift(28)) / 4
feats_common = ["요일", "월", "공휴일", "연휴전날", "연휴다음날", "temperature_2m_max", "temperature_2m_min",
                "precipitation_sum", "snowfall_sum", "일련"]
d = days.dropna(subset=[f"ma4_{k}" for k in W]).copy()
train = d[d["일자"] < "2025-10-01"]; test = d[d["일자"] >= "2025-10-01"]
imp_rows = []
for k in W:
    y = f"호출_{k}"
    X = feats_common + [f"lag7_{k}", f"ma4_{k}"]
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=8,
                                      l2_regularization=1.0, random_state=0)
    m.fit(train[X], train[y])
    pred = m.predict(test[X])
    test = test.assign(**{f"pred_{k}": pred})
    mape = lambda p: float(np.mean(np.abs(p - test[y]) / test[y]) * 100)
    mae = lambda p: float(np.mean(np.abs(p - test[y])))
    r = dict(시간대=k, 평가일=len(test), 모형_MAPE=round(mape(pred), 1), 모형_MAE=round(mae(pred), 1),
             주전값_MAPE=round(mape(test[f"lag7_{k}"]), 1), 주전값_MAE=round(mae(test[f"lag7_{k}"]), 1),
             평균4주_MAPE=round(mape(test[f"ma4_{k}"]), 1), 평균4주_MAE=round(mae(test[f"ma4_{k}"]), 1))
    # 수요 많은 날(평가기간 상위 20%) 탐지
    thr = test[y].quantile(0.8); hi = test[y] >= thr
    ph = pred >= np.quantile(pred, 0.8); pl = test[f"lag7_{k}"] >= test[f"lag7_{k}"].quantile(0.8)
    r["상위20_정밀도_모형"] = round(float((hi & ph).sum() / ph.sum()) * 100, 0)
    r["상위20_정밀도_주전값"] = round(float((hi & pl).sum() / pl.sum()) * 100, 0)
    r["수요많은날_포기율"] = round(float(test.loc[hi, f"포기율_{k}"].mean()) * 100, 1)
    r["나머지날_포기율"] = round(float(test.loc[~hi, f"포기율_{k}"].mean()) * 100, 1)
    # 평일만
    wd = test["요일"] < 5
    r["평일_모형_MAPE"] = round(float(np.mean(np.abs(pred[wd] - test.loc[wd, y]) / test.loc[wd, y]) * 100), 1)
    r["평일_주전값_MAPE"] = round(float(np.mean(np.abs(test.loc[wd, f"lag7_{k}"] - test.loc[wd, y]) / test.loc[wd, y]) * 100), 1)
    rows.append(r)
    pi = permutation_importance(m, test[X], test[y], n_repeats=20, random_state=0)
    for f, v in zip(X, pi.importances_mean):
        imp_rows.append(dict(시간대=k, 특징=f, 중요도=round(float(v), 2)))
    for kk, vv in r.items():
        if kk != "시간대":
            out[f"fc_{k}_{kk}"] = vv
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/05_forecast.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(imp_rows).to_csv(f"{OUT}/05_importance.csv", index=False, encoding="utf-8-sig")
test[["일자", "요일"] + [f"호출_{k}" for k in W] + [f"pred_{k}" for k in W] + [f"lag7_{k}" for k in W]].to_csv(
    f"{OUT}/05_test_pred.csv", index=False, encoding="utf-8-sig")
# 평일 호출 수 변동폭(같은 요일인데도)
wdays = days[(days["요일"] < 5) & (days["공휴일"] == 0)]
for k in W:
    y = f"호출_{k}"
    out[f"fc_{k}_weekday_p10"] = round(float(wdays[y].quantile(0.1)), 0)
    out[f"fc_{k}_weekday_p90"] = round(float(wdays[y].quantile(0.9)), 0)
    out[f"fc_{k}_weekday_corr_abandon"] = round(float(wdays[[y, f"포기율_{k}"]].corr().iloc[0, 1]), 2)
save_summary(out)
print(res.T.to_string()); print(pd.DataFrame(imp_rows).sort_values("중요도", ascending=False).head(12).to_string())
print({k: v for k, v in out.items() if "weekday" in k})
