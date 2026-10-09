# -*- coding: utf-8 -*-
"""공통: 경로·자료 적재·도표 양식. 모든 분석 스크립트가 import 한다."""
import json, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
CACHE = os.path.join(RAW, "_cache")          # data/raw 는 .gitignore 대상(용량)
OUT = os.path.join(ROOT, "analysis", "out")
FIG = os.path.join(ROOT, "analysis", "fig")
for d in (CACHE, OUT, FIG):
    os.makedirs(d, exist_ok=True)

TRIPS_CSV = os.path.join(RAW, "서울시설공단_장애인콜택시 탑승내역_20251231.csv")
SEOUL_GU = ["종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구", "강북구", "도봉구",
            "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구", "구로구", "금천구", "영등포구", "동작구",
            "관악구", "서초구", "강남구", "송파구", "강동구"]
IMMEDIATE_LEAD_MIN = 10      # 접수→예정 10분 미만 = 즉시 호출, 이상 = 예약 호출 (설정값)
HORIZON_MIN = 60             # "60분 안에 탈 확률"의 기준 (설정값)


def load_trips():
    """탑승내역 CSV → 정제 parquet 캐시. 열 의미는 01_prep.py 머리말."""
    pq = os.path.join(CACHE, "trips.parquet")
    if os.path.exists(pq) and os.path.getmtime(pq) > os.path.getmtime(TRIPS_CSV):
        return pd.read_parquet(pq)
    df = pd.read_csv(TRIPS_CSV, dtype=str, encoding="utf-8-sig")
    for c in ["접수일시", "예정일시", "배차일시", "승차일시", "하차일시", "취소일시"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["승차거리_km"] = pd.to_numeric(df["승차거리"], errors="coerce") / 1000
    df["요금"] = pd.to_numeric(df["요금"], errors="coerce")
    lead = (df["예정일시"] - df["접수일시"]).dt.total_seconds() / 60
    df["예약"] = lead >= IMMEDIATE_LEAD_MIN
    # 대기 시계 시작: 즉시 호출은 접수 시각, 예약 호출은 예정 시각
    df["t0"] = df["접수일시"].where(~df["예약"], df["예정일시"])
    df["탑승"] = df["승차일시"].notna()
    df["취소"] = df["취소일시"].notna() & ~df["탑승"]
    m = lambda a, b: (a - b).dt.total_seconds() / 60
    df["T_탑승"] = m(df["승차일시"], df["t0"])
    df["T_취소"] = m(df["취소일시"], df["t0"])
    df["T_배차"] = m(df["배차일시"], df["t0"])
    df["접근분"] = m(df["승차일시"], df["배차일시"])        # 배차→승차 (빈 차 접근)
    df["점유분"] = m(df["하차일시"], df["배차일시"])        # 배차→하차 (차량 점유)
    df["승차분"] = m(df["하차일시"], df["승차일시"])
    df["일자"] = df["t0"].dt.normalize()
    df["시"] = df["t0"].dt.hour
    df["요일"] = df["t0"].dt.dayofweek
    df["서울출발"] = df["출발구"].isin(SEOUL_GU)
    df.to_parquet(pq)
    return df


def holidays_2025():
    import holidays
    return set(pd.Timestamp(d) for d in holidays.KR(years=2025).keys())


def save_summary(updates: dict):
    p = os.path.join(OUT, "summary.json")
    s = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
    s.update(updates)
    json.dump(s, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)


# ---------- 도표 양식: 흑백 계열 + 강조 1색, 맑은 고딕, 제목 명사형 ----------
INK, INK2, MUTED, GRID = "#1a1a1a", "#4d4d4d", "#8c8c8c", "#e3e3e3"
ACCENT, ACCENT_L = "#9c2f2f", "#d9a3a3"     # 와인 1색


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 7.6,
        "axes.edgecolor": INK2, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "xtick.color": INK2,
        "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "legend.frameon": False, "savefig.dpi": 220, "figure.dpi": 110,
    })
    return plt


def finish(fig, name, title, source):
    """그림 파일에는 제목·출처를 넣지 않고(보고서 본문 폭에서 글자가 잘림) fig/captions.json에 남긴다.
    보고서 조립 스크립트가 그림 위에 제목, 아래에 출처를 본문 글꼴로 붙인다."""
    path = os.path.join(FIG, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white", pad_inches=0.03)
    cp = os.path.join(FIG, "captions.json")
    cap = json.load(open(cp, encoding="utf-8")) if os.path.exists(cp) else {}
    cap[name] = {"title": title, "source": source}
    json.dump(cap, open(cp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return path
