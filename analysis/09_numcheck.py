# -*- coding: utf-8 -*-
"""9단계 — 원고 숫자 대조: 원고.md의 숫자를 전부 뽑아 summary.json·out/*.csv 값과 맞춘다.
실행: python analysis/09_numcheck.py   → analysis/out/수치대조.csv (숫자 | 위치 | 판정 | 근거)
판정: 일치(summary 키) / 일치(CSV) / 허용(설정값·연도·절 번호·외부 출처 수치 — 사유 기재) / 미대조
허용 목록은 아래 ALLOW에 사유와 함께 둔다. 미대조가 0건이어야 제출본을 만든다.
"""
import json, os, re, glob, sys
import pandas as pd
from common import OUT, ROOT

MS = os.path.join(ROOT, "원고.md")
S = json.load(open(os.path.join(OUT, "summary.json"), encoding="utf-8"))

# 외부 출처·설정값 등 (값 → 사유)
ALLOW = {
    "2025": "분석 연도", "2026": "연도", "2024": "연도", "2023": "연도", "2021": "연도", "2019": "연도", "2015": "연도(자동배차 도입)",
    "2027": "연도(제안 일정)",
    "60": "설정값: 60분 기준", "10": "설정값(즉시 호출 판정 10분·포화 10건·10분 단위)", "5": "설정값: 단시간 운전원 5시간 근무",
    "55": "설정값: 2026년 단시간 운전원 증원분 175-120(서울시 해명자료)", "120": "외부: 2025 단시간 운전원(서울시 2026-01-10)",
    "175": "외부: 2026 단시간 운전원 계획(서울시 2026-01-10)", "275": "설정값: 55명×5시간", "818": "외부: 특장차 대수(서울시 2026-01-10)",
    "834": "외부: 정규 운전원(서울시 2026-01-10)", "33.5": "외부: 서울시 공표 2025 평균 대기시간", "47": "외부: 서울시 공표 2023 평균 대기",
    "40.8": "외부: 서울시 공표 2024 평균 대기", "40,387": "외부: 등록 회원 2024(서울시 미디어허브)", "152": "외부: 법정기준 대비(서울시)",
    "2.5": "외부: 전장연 요구 차량당 운전원", "16": "외부/설정: 16시간 운행(전장연)·06~22시 16시간", "20": "외부: 언론 인용 취소율 20% 이상(헤럴드경제)",
    "90": "외부/설정: 언론 90분 대기 사례·90% 목표", "0": "수치 0", "1": "순번", "2": "순번", "3": "순번", "4": "순번", "6": "순번", "7": "순번·시각",
    "8": "시각", "9": "순번", "11": "순번", "12": "시각", "13": "시각", "14": "시각", "15": "시각", "17": "시각", "18": "시각", "19": "시각",
    "21": "시각", "22": "시각", "23": "시각", "30": "시각(분)", "1,500": "설정값: 곡선 최대 운전시간", "250": "설정값: 곡선 단계(50명×5시간)",
    "500": "곡선 단계", "750": "곡선 단계", "1,000": "곡선 단계", "1,250": "곡선 단계", "122": "설정: 학습·검증일 수", "244": "설정: 평일 수",
    "6.6": "외부: 스타리아 LPG 연비(판매 제원)", "1.5": "가정: LPG 배출계수 하한", "1.8": "가정: LPG 배출계수 상한", "28": "설정값: A안 07시조 인원",
    "27": "설정값: A안 14시조 인원", "100": "단위(%)", "150": "외부: 법정기준 150명당 1대", "80": "설정/외부", "50": "설정값", "05": "시각", "00": "시각",
}


def flat_numbers():
    vals = {}
    for k, v in S.items():
        if isinstance(v, (int, float)):
            vals.setdefault(norm(v), []).append(k)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float)):
                    vals.setdefault(norm(vv), []).append(f"{k}.{kk}")
    return vals


def norm(x):
    try:
        f = float(str(x).replace(",", ""))
    except ValueError:
        return str(x)
    return f"{f:.4f}".rstrip("0").rstrip(".")


def csv_numbers():
    vals = {}
    for p in glob.glob(os.path.join(OUT, "*.csv")):
        if os.path.basename(p).startswith("수치대조"):
            continue
        df = pd.read_csv(p, encoding="utf-8-sig")
        for c in df.columns:
            for v in pd.to_numeric(df[c], errors="coerce").dropna().unique():
                for r in (v, round(v, 1), round(v, 0), round(v * 100, 1), round(v * 100, 0)):
                    vals.setdefault(norm(r), os.path.basename(p) + ":" + str(c))
    return vals


def main():
    text = open(MS, encoding="utf-8").read()
    sv, cv = flat_numbers(), csv_numbers()
    rows = []
    sec = ""
    for ln, line in enumerate(text.splitlines(), 1):
        if line.startswith("## "):
            sec = line[3:].strip()
        if line.startswith("<!--") or "http" in line and line.strip().startswith("|") and "URL" in line:
            pass
        clean = re.sub(r"https?://\S+", "", line)
        clean = re.sub(r"[\w./-]*\.(?:kr|com|org|io)[\w./-]*", "", clean)        # 주소(도메인·경로)
        clean = re.sub(r"(?:19|20)\d{2}\.\d{1,2}(?:\.\d{1,2})?", "", clean)    # 날짜 2026.1.10
        clean = re.sub(r"(?i)(?:python|pandas|numpy|scipy|scikit-learn|matplotlib)\s+\d+(?:\.\d+)+", "", clean)  # 버전
        clean = re.sub(r"\b\d{8,}\b", "", clean)          # 데이터 번호·코드
        for m in re.finditer(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\d,]|\.\d)", clean):
            raw = m.group(1); n = norm(raw)
            ctx = clean[max(0, m.start() - 12): m.end() + 8].strip()
            if n in sv:
                rows.append((raw, f"{sec} L{ln}", "일치", sv[n][0], ctx))
            elif raw in ALLOW or n in ALLOW:
                rows.append((raw, f"{sec} L{ln}", "허용", ALLOW.get(raw, ALLOW.get(n)), ctx))
            elif n in cv:
                rows.append((raw, f"{sec} L{ln}", "일치(CSV)", cv[n], ctx))
            else:
                rows.append((raw, f"{sec} L{ln}", "미대조", "", ctx))
    df = pd.DataFrame(rows, columns=["숫자", "위치", "판정", "근거", "문맥"])
    df.to_csv(os.path.join(OUT, "수치대조.csv"), index=False, encoding="utf-8-sig")
    print(df["판정"].value_counts().to_string())
    bad = df[df["판정"] == "미대조"]
    if len(bad):
        print(bad.to_string())
    return len(bad)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
