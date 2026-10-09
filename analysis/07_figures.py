# -*- coding: utf-8 -*-
"""7단계 — 보고서 그림. 숫자는 analysis/out/ 산출물에서만 읽는다(손 입력 없음).
실행: python analysis/07_figures.py [그림번호...]   (01~06 먼저)
양식: 흑백 계열 + 강조 1색(와인), 맑은 고딕, 제목 명사형(common.style/finish).
"""
import json, os, re, sys
import numpy as np, pandas as pd
from common import CACHE, OUT, RAW, FIG, style, finish, INK, INK2, MUTED, GRID, ACCENT, ACCENT_L

plt = style()
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

S = json.load(open(os.path.join(OUT, "summary.json"), encoding="utf-8"))
SRC = "자료: 서울시설공단 장애인콜택시 탑승내역 2025(공공데이터포털 15115859), 분석 이완"
W_IN = 4.9      # 서식 오른쪽 칸 폭(약 357pt) 안쪽 350pt — 1:1 인쇄 크기


def fig1():
    p = pd.read_csv(os.path.join(OUT, "03_profile.csv"))
    p = p[(p["슬롯"] >= 30) & (p["슬롯"] < 138)]
    x = p["슬롯"] / 6
    fig, ax = plt.subplots(figsize=(W_IN, 2.25))
    ax.fill_between(x, 0, p["배차대기_평균"], color=ACCENT_L, alpha=0.85, lw=0, label="배차를 기다리는 호출(건)")
    ax.plot(x, p["운행중_평균"], color=INK, lw=1.6, label="운행 중 특장차(대)")
    ax.plot(x, p["신규호출_평균"], color=MUTED, lw=0.9, ls="--", label="10분 동안 새로 들어온 호출(건)")
    ax.axvspan(6 + 20 / 60, 7, color="#bbbbbb", alpha=0.35, lw=0)
    ax.annotate(f"06:20~07:00\n운행 차량 0대", xy=(6.6, 5), xytext=(5.1, 300), fontsize=7.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    b1430 = p.loc[p["슬롯"] == 87, "운행중_평균"].iloc[0]; b1630 = p.loc[p["슬롯"] == 99, "운행중_평균"].iloc[0]
    q16 = p.loc[p["슬롯"] == 96, "배차대기_평균"].iloc[0]
    ax.annotate(f"14:30 {b1430:.0f}대 → 16:30 {b1630:.0f}대\n같은 시각 대기 호출 {q16:.0f}건", xy=(16.2, b1630), xytext=(17.2, 215),
                fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xlim(5, 23); ax.set_xticks(range(5, 24, 1)); ax.set_xticklabels([f"{h}" for h in range(5, 24)])
    ax.set_xlabel("시각(평일 평균, 10분 단위)"); ax.set_ylim(0, 380)
    ax.legend(loc="upper right", fontsize=6.8, ncol=1)
    return finish(fig, "그림1_대기열과운행차량.png", "그림 2. 평일 10분 단위 배차 대기 호출과 운행 중 특장차(2025년 평일 244일 평균)",
                  SRC + " · 운행 중 = 배차~하차, 배차 대기 = 대기 시작~배차·취소")


def fig2():
    base = pd.read_parquet(os.path.join(CACHE, "base.parquet"))
    sp = base[(~base["예약"]) & (base["차량구분"] == "특장차")]
    T = np.where(sp["탑승"], sp["T_탑승"], np.nan)
    sp = sp.assign(b60=(sp["탑승"] & (sp["T_탑승"] <= 60)))
    hr = pd.read_csv(os.path.join(OUT, "02_hour.csv")).set_index("시")
    cif = sp.groupby("시")["b60"].mean() * 100
    h = [x for x in range(5, 24) if x in hr.index]
    fig, ax = plt.subplots(figsize=(W_IN, 2.15))
    xs = np.arange(len(h))
    ax.bar(xs - 0.2, hr.loc[h, "완료기준_60분내비율"], width=0.4, color="#c8c8c8", label="탑승 완료 건 기준(공식 집계 방식)")
    ax.bar(xs + 0.2, cif.loc[h], width=0.4, color=ACCENT, label="호출 건 기준(포기 포함)")
    for i, hh in enumerate(h):
        if hh in (7, 8, 16, 17, 20, 21):
            ax.text(i + 0.2, cif.loc[hh] + 1.5, f"{cif.loc[hh]:.0f}", ha="center", fontsize=6.8, color=ACCENT)
            ax.text(i - 0.2, hr.loc[hh, "완료기준_60분내비율"] + 1.5, f"{hr.loc[hh, '완료기준_60분내비율']:.0f}", ha="center",
                    fontsize=6.8, color=INK2)
    ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in h]); ax.set_xlabel("호출 시각(시)")
    ax.set_ylabel("60분 안에 탄 비율(%)"); ax.set_ylim(0, 112)
    ax.legend(loc="upper center", fontsize=6.8, ncol=2, bbox_to_anchor=(0.5, 1.13))
    return finish(fig, "그림2_시간대별60분탑승.png", "그림 1. 시간대별 호출 후 60분 안에 탄 비율, 탑승 완료 기준과 호출 기준(특장차 즉시 호출)",
                  SRC)


def _seoul_polys():
    g = json.load(open(os.path.join(RAW, "HangJeongDong_ver20251231.geojson"), encoding="utf-8"))
    feats = {}
    for f in g["features"]:
        p = f["properties"]
        if not p["adm_nm"].startswith("서울특별시"):
            continue
        _, gu, dong = p["adm_nm"].split(" ", 2)
        geom = f["geometry"]
        rings = []
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            rings.append(np.array(poly[0]))
        feats[(gu, dong)] = rings
    return feats


def nz(s):
    return re.sub(r"(?<=[가-힣])제(?=\d)", "", s).replace(".", "·")


def fig3():
    feats = _seoul_polys()
    look = {}
    for (gu, dong) in feats:
        look[(gu, dong)] = (gu, dong); look.setdefault((gu, nz(dong)), (gu, dong))
    dg = pd.read_csv(os.path.join(OUT, "02_dong.csv"))
    val = {}
    miss = 0
    for r in dg.itertuples():
        k = look.get((r.자치구, r.행정동)) or look.get((r.자치구, nz(r.행정동)))
        if k: val[k] = r.KM_60분내탑승확률
        else: miss += 1
    bins = [0, 60, 70, 80, 90, 101]
    cols = ["#4a1212", "#9c2f2f", "#d08a8a", "#f0d9d9", "#ffffff"]
    labels = ["60% 미만", "60~70%", "70~80%", "80~90%", "90% 이상"]
    fig, ax = plt.subplots(figsize=(W_IN, 3.45))
    vs, cs = [], []
    for k, rings in feats.items():
        v = val.get(k)
        c = "#e9e9e9" if v is None else cols[np.digitize([v], bins)[0] - 1]
        for r in rings:
            vs.append(r); cs.append(c)
    ax.add_collection(PolyCollection(vs, facecolors=cs, edgecolors="#9a9a9a", linewidths=0.15))
    # 자치구 경계(굵게) — 같은 구의 동 외곽을 다시 그리는 대신 구 이름만 중심에 표기
    gu_c = {}
    for (gu, d), rings in feats.items():
        pts = np.vstack(rings); gu_c.setdefault(gu, []).append(pts.mean(0))
    for gu, pts in gu_c.items():
        c = np.mean(pts, 0); ax.text(c[0], c[1], gu, fontsize=5.4, ha="center", va="center", color=INK,
                                     bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.6))
    ax.autoscale_view(); ax.set_aspect(1.2); ax.axis("off"); ax.grid(False)
    hand = [Patch(fc=c, ec="#9a9a9a", lw=0.3, label=l) for c, l in zip(cols, labels)] + [Patch(fc="#e9e9e9", ec="#9a9a9a", lw=0.3, label="호출 300건 미만·경계 불일치")]
    ax.legend(handles=hand, loc="upper left", fontsize=5.8, title="60분 안에 탈 확률", title_fontsize=6.2, ncol=1,
              bbox_to_anchor=(-0.04, 1.02))
    lo, hi = S["dong_p60_min_name"], S["dong_p60_max_name"]
    ax.text(1.0, 0.98, f"최저 {lo} {S['dong_p60_min']:.1f}%\n최고 {hi} {S['dong_p60_max']:.1f}%\n분석 {S['dong_n']}개 동",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.3, color=INK2)
    return finish(fig, "그림3_행정동별60분탑승확률.png", "그림 3. 출발 행정동별 호출 후 60분 안에 탈 확률(특장차 즉시 호출, 포기 포함 생존분석)",
                  SRC + " · 경계: 통계청 행정동 경계(vuski/admdongkor ver20251231)")


def fig4():
    v = pd.read_csv(os.path.join(OUT, "04_validation.csv")).set_index("h")
    v = v.loc[5:23]
    fig, axs = plt.subplots(1, 2, figsize=(W_IN, 1.7))
    for ax, a, b, t in [(axs[0], "실제_60분내탑승", "모형_60분내탑승", "60분 안에 탄 비율"), (axs[1], "실제_포기", "모형_포기", "포기율")]:
        ax.plot(v.index, v[a] * 100, color=INK, lw=1.4, marker="o", ms=2.5, label="실제(검증일)")
        ax.plot(v.index, v[b] * 100, color=ACCENT, lw=1.4, ls="--", marker="s", ms=2.5, label="시뮬레이터")
        ax.set_title(t, fontsize=8.5, loc="left", color=INK2); ax.set_xticks(range(5, 24, 2)); ax.set_xlabel("호출 시각(시)")
        ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    axs[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    return finish(fig, "그림4_시뮬레이터검증.png",
                  f"그림 4. 시뮬레이터 재현 검증(학습에 쓰지 않은 평일 {S['sim_n_test']}일, 시간대 평균 오차 {S['sim_val_hour_mae_pp']}%p)",
                  SRC + " · 시간대별 유효 공급 계수는 다른 평일 122일로 보정")


def fig5():
    b = pd.read_csv(os.path.join(OUT, "04_scenarios_byhour.csv")).set_index("h").loc[5:23] * 100
    fig, axs = plt.subplots(2, 1, figsize=(W_IN, 2.95), gridspec_kw=dict(height_ratios=[2.5, 1]))
    ax = axs[0]
    sty = {"S0": ("현행(2025 공급 재현)", INK, "-", 1.5), "S2": ("B안 06~22시 균등", "#8a8a8a", ":", 1.5),
           "S1": ("A안 07시·14시 시작(가정)", MUTED, "--", 1.2), "HY": ("C안 05:30·12:30 시작", ACCENT, "-", 2.0)}
    for k, (lab, c, ls, lw) in sty.items():
        v = S[f"sim_{k}_b60"]
        ax.plot(b.index, b[k], color=c, ls=ls, lw=lw, label=f"{lab} — 하루 {v:.1f}%")
    ax.set_ylabel("60분 안에 탄 비율"); ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    ax.set_xticks(range(5, 24)); ax.set_ylim(30, 100); ax.legend(fontsize=6.3, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    ax2 = axs[1]
    rows = [("C안", [(5.5, S["sim_HY_am_n"]), (12.5, S["sim_HY_pm_n"])], ACCENT), ("A안", [(7, 28), (14, 27)], MUTED)]
    for yi, (lab, shifts, c) in enumerate(rows):
        for h0, k in shifts:
            ax2.barh(-yi, 5, left=h0, height=0.55, color=c, alpha=0.35, ec=c, lw=0.6)
            ax2.text(h0 + 0.1, -yi, f"{int(h0):02d}:{int(round((h0 % 1) * 60)):02d} 시작 {k}명", fontsize=6.4, va="center", color=INK)
    ax2.set_yticks([0, -1]); ax2.set_yticklabels(["C안", "A안"], fontsize=7); ax2.set_xlim(5, 24); ax2.set_xticks(range(5, 24))
    ax2.set_ylim(-1.6, 0.6); ax2.set_xlabel("단시간 운전원 5시간 근무조(55명)"); ax2.grid(axis="y", visible=False)
    fig.tight_layout(h_pad=0.6)
    return finish(fig, "그림5_운전원배치시나리오.png", f"그림 5. 같은 운전시간(55명×5시간) 배치안별 시간대 60분 내 탑승 비율(검증일 {S['sim_n_test']}일)",
                  SRC + " · A안 시각은 서울시 '출퇴근 5시간조' 설명을 가정한 값, C안은 학습일 122일에서 오전·오후 시작 시각 80개 조합 중 선택")


def fig6():
    c = pd.read_csv(os.path.join(OUT, "04_curve.csv"))
    fig, ax = plt.subplots(figsize=(W_IN, 2.15))
    ax.plot(c["추가운전시간"], c["최적"], color=ACCENT, lw=1.8, marker="o", ms=3, label="시작시각 최적화 — 하루 평균")
    ax.plot(c["추가운전시간"], c["균등"], color=INK, lw=1.2, ls="--", marker="s", ms=3, label="06~22시 균등 — 하루 평균")
    ax.plot(c["추가운전시간"], c["최적_최저시간대"], color=ACCENT, lw=0.9, ls=":", label="최적화 — 가장 나쁜 시간대")
    ax.plot(c["추가운전시간"], c["균등_최저시간대"], color=INK, lw=0.9, ls="-.", label="균등 — 가장 나쁜 시간대")
    ax.axvline(275, color=MUTED, lw=0.6); ax.text(285, 32, "2026 증원분\n(55명×5시간)", fontsize=6.8, color=INK2)
    ax.set_xlabel("추가 운전시간(시간/평일 하루)"); ax.set_ylabel("60분 안에 탄 비율")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0f}%"); ax.set_ylim(20, 100)
    ax.legend(fontsize=6.2, loc="lower right", ncol=1)
    return finish(fig, "그림6_운전시간투입곡선.png", "그림 6. 추가 운전시간 투입량별 60분 내 탑승 비율(최적 배치와 균등 배치)", SRC + " · 검증일 기준")


def fig7():
    t = pd.read_csv(os.path.join(OUT, "05_test_pred.csv"), parse_dates=["일자"])
    fig, axs = plt.subplots(1, 2, figsize=(W_IN, 1.85), sharey=False)
    for ax, k in zip(axs, ["오전", "오후"]):
        ax.plot(t["일자"], t[f"호출_{k}"], color=INK, lw=1.0, label="실제")
        ax.plot(t["일자"], t[f"pred_{k}"], color=ACCENT, lw=1.0, ls="--", label="예측(전날 기준)")
        ax.set_title(f"{k} {'07~10시' if k == '오전' else '14~18시'} 호출 수 — 오차 {S[f'fc_{k}_모형_MAPE']}%", fontsize=8, loc="left", color=INK2)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%m/%d")); ax.tick_params(axis="x", labelsize=6.5)
    axs[0].legend(fontsize=6.8, loc="lower left")
    fig.tight_layout()
    return finish(fig, "그림7_호출수예측.png", "그림 7. 다음 날 출퇴근 시간대 호출 수 예측(2025년 10~12월 평가)", SRC + " · 기상: Open-Meteo 재분석")


if __name__ == "__main__":
    want = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7"]
    for w in want:
        print(globals()[f"fig{w}"]())
