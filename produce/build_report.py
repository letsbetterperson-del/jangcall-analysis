# -*- coding: utf-8 -*-
"""붙임2 분석보고서 서식을 PDF로 조립한다 — run 1회용.

왜 run 폴더 스크립트인가(STATE 기록): 이 PC(DESKTOP-3JMF70J)에는 한컴오피스가 없어 hwp 서식을 열 수 없다.
대신 decode 때 한컴 PC에서 렌더링한 서식 PDF(원문/_render/붙임2-분석보고서최종.pdf)를 실측해
같은 표 구조(왼쪽 항목 칸 85.0~147.6pt 회색 #F2F2F2, 오른쪽 작성 칸 ~504.5pt, 선 0.36pt)를 HTML로 만들고
Chrome으로 A4 PDF를 찍은 뒤, 서식 머리(한겨레·숲과나눔 로고, 1쪽 제목 띠)는 서식 PDF에서 벡터 그대로 옮겨 얹는다.
칸 안 작성 지시문(※ …)과 칸 밖 '분석보고서 작성 시 유의사항'(작성자용 안내 — 분량·형식·제출 방법)은 지운다(제출안내.md 기록).

사용: python produce/build_report.py   → out/build/분석보고서.html, 제출/이완_분석보고서.pdf
"""
import html, io, json, os, re, subprocess, sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", write_through=True)
RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pymupdf as fitz  # noqa: E402

FORM = os.path.join(RUN, "원문", "_render", "붙임2-분석보고서최종.pdf")
MAN = os.path.join(RUN, "원고.md")
FIG = os.path.join(RUN, "analysis", "fig")
BUILD = os.path.join(RUN, "out", "build")
OUT_PDF = os.path.join(RUN, "제출", "이완_분석보고서.pdf")
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
CODE_LINK = os.environ.get("CODE_LINK", "")
os.makedirs(BUILD, exist_ok=True)

XL, XR, XM = 85.0, 504.5, 147.6           # 서식 표 왼쪽·오른쪽·항목 칸 경계(실측)
TOP1, TOPN, BOT = 170.1, 99.1, 772.0      # 1쪽 표 시작, 이어지는 쪽 표 시작, 아래 여백 경계
LABELS = {
    "이름/팀명": "이름/팀명", "주제": "주제", "분석 배경 및 목적": "분석 배경<br>및 목적", "활용 데이터": "활용<br>데이터",
    "분석과정 및 방법": "분석과정<br>및 방법", "AI 기술 활용": "AI 기술<br>활용", "분석 내용 및 결과": "분석 내용<br>및 결과",
    "정책제안 및 기대효과": "정책제안<br>및<br>기대효과", "분석도구 및 참고문헌": "분석도구<br>및<br>참고문헌",
    "분석결과 이미지(필수)": "분석결과<br>이미지<br>(필수)",
}
CAP = json.load(open(os.path.join(FIG, "captions.json"), encoding="utf-8"))

CSS = f"""
@page {{ size: 595pt 841pt; margin: {TOPN}pt {595 - XR}pt {841 - BOT}pt {XL}pt; }}
@page :first {{ margin-top: {TOP1}pt; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: 'Malgun Gothic'; font-size: 8.9pt; line-height: 1.47; color: #111; word-break: keep-all; }}
table.form {{ width: {XR - XL}pt; border-collapse: collapse; table-layout: fixed; }}
table.form > tbody > tr > td {{ border: 0.36pt solid #000; vertical-align: top; }}
td.lab {{ width: {XM - XL}pt; background: #F2F2F2; font-family: 'Batang'; font-size: 10.6pt; line-height: 1.55;
          text-align: center; vertical-align: middle !important; padding: 4pt 2pt; }}
td.lab.top {{ vertical-align: top !important; padding-top: 10pt; }}
td.val {{ padding: 4.5pt 6pt 5pt 6pt; }}
tr {{ break-inside: auto; }}
p {{ margin: 0 0 1.6pt 0; }}
p.h {{ font-weight: bold; margin: 5pt 0 1.8pt 0; font-size: 9.3pt; break-after: avoid; }}
div.fig p.cap {{ break-after: avoid; }}
p.h:first-child {{ margin-top: 0; }}
p.b {{ padding-left: 10pt; text-indent: -9pt; }}
p.title {{ font-size: 10.2pt; font-weight: bold; line-height: 1.45; }}
table.in {{ width: 100%; border-collapse: collapse; margin: 2.5pt 0 4pt 0; table-layout: auto; }}
table.in.fixed {{ table-layout: fixed; }}
table.in.fixed td:nth-child(4) {{ word-break: break-all; font-size: 7.3pt; }}
table.in {{ break-inside: auto; }}
table.in td, table.in th {{ border: 0.36pt solid #555; padding: 1.6pt 3pt; font-size: 7.9pt; line-height: 1.32; vertical-align: middle; }}
table.in th {{ background: #F2F2F2; font-weight: bold; text-align: center; }}
table.in tr {{ break-inside: avoid; }}
div.fig {{ margin: 4pt 0 6pt 0; break-inside: avoid; }}
div.fig p.cap {{ font-weight: bold; font-size: 8.4pt; margin: 0 0 2pt 0; line-height: 1.35; }}
div.fig img {{ width: 100%; display: block; }}
div.fig p.src {{ font-size: 6.9pt; color: #555; margin: 1.5pt 0 0 0; line-height: 1.3; }}
div.note {{ margin-top: 12pt; font-size: 10pt; line-height: 1.6; break-inside: avoid; }}
div.note p {{ margin: 0; padding-left: 20pt; text-indent: -10pt; }}
div.note p.nh {{ font-weight: bold; padding-left: 0; text-indent: 0; margin-bottom: 4pt; }}
"""


def inline(s):
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return s


def md_table(rows, sec):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows if not re.match(r"^\|\s*-", r)]
    head, body = cells[0], cells[1:]
    out = ['<table class="in fixed">' if sec == "활용 데이터" else '<table class="in">']
    if sec == "활용 데이터":       # 서식 머리행 그대로: 출처 아래 플랫폼(명)·URL 2단
        out.append('<colgroup><col style="width:31%"><col style="width:15%"><col style="width:19%"><col style="width:35%"></colgroup>')
        out.append('<tr><th rowspan="2">활용 데이터(명)</th><th rowspan="2">제공기관(명)</th><th colspan="2">출처</th></tr>'
                   '<tr><th>플랫폼(명)</th><th>URL</th></tr>')
    else:
        out.append("<tr>" + "".join(f"<th>{inline(h)}</th>" for h in head) + "</tr>")
    for r in body:
        out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def figure(name):
    c = CAP.get(name, {"title": name, "source": ""})
    path = os.path.join(FIG, name).replace("\\", "/")
    return (f'<div class="fig"><p class="cap">{html.escape(c["title"])}</p><img src="file:///{path}">'
            f'<p class="src">{html.escape(c["source"])}</p></div>')


def section_html(sec, lines):
    out, buf = [], []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.strip().startswith("|"):
            tb = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tb.append(lines[i]); i += 1
            out.append(md_table(tb, sec)); continue
        m = re.match(r"^\[\[(.+?)\]\]$", ln.strip())
        if m:
            out.append(figure(m.group(1)))
        elif ln.startswith("□"):
            out.append(f'<p class="h">{inline(ln)}</p>')
        elif ln.strip().startswith("○"):
            out.append(f'<p class="b">{inline(ln.strip())}</p>')
        elif ln.strip():
            cls = ' class="title"' if sec == "주제" else ""
            out.append(f"<p{cls}>{inline(ln.strip())}</p>")
        i += 1
    return "\n".join(out)


def build_html():
    text = open(MAN, encoding="utf-8").read().replace("{CODE_LINK}", CODE_LINK or "(링크 기재 예정)")
    secs, cur = [], None
    for ln in text.splitlines():
        if ln.startswith("## "):
            cur = [ln[3:].strip(), []]; secs.append(cur)
        elif cur is not None:
            cur[1].append(ln)
    rows = []
    for name, lines in secs:
        lab = LABELS[name]
        long = sum(len(l) for l in lines) > 900
        rows.append(f'<tr><td class="lab{" top" if long else ""}">{lab}</td><td class="val">{section_html(name, lines)}</td></tr>')
    note = """<div class="note"><p class="nh">※ 분석보고서 작성 시 유의사항</p>
<p>- 본문 5장 내외로 작성(표지, 참고문헌 제외)하되, 위 필수항목은 분석보고서에 반드시 명시하여 작성</p>
<p>- 제시한 항목 외 추가 내용이 있을 경우 별도 제목을 기재하여 작성</p>
<p>- 분석 결과 이미지(시각화, 인포그래픽 등)는 문서 내 포함 필수</p>
<p>- 프로그래밍 언어를 이용한 분석 사례의 경우, 분석 코드를 Github/클라우드 링크를 분석보고서에 기재, 결선 진출 시 실행 가능한 전체 코드를 별도 안내에 따라 제출하며, 결선 발표 평가 시 코드 실행을 확인함.</p>
<p>- 제출 형식 : PDF / 파일명: 팀명(또는 이름)_분석보고서.pdf / 이메일 접수: yaerumjung@koreashe.org</p></div>"""
    doc = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><style>{CSS}</style></head>
<body><table class="form"><tbody>{''.join(rows)}</tbody></table></body></html>"""
    p = os.path.join(BUILD, "분석보고서.html")
    open(p, "w", encoding="utf-8").write(doc)
    return p


def chrome_pdf(html_path, pdf_path):
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--allow-file-access-from-files",
                    f"--print-to-pdf={pdf_path}", "file:///" + html_path.replace("\\", "/")], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)


def decorate(raw_pdf, out_pdf):
    form = fitz.open(FORM)
    doc = fitz.open(raw_pdf)
    font = fitz.Font(fontfile="C:/Windows/Fonts/malgun.ttf")
    for i, page in enumerate(doc):
        # 서식 머리: 1쪽은 로고+제목 띠(y 40~160), 이후 쪽은 로고(y 40~92)
        vs = [it for g in page.get_drawings() for it in g["items"]
              if it[0] == "l" and abs(it[1].x - it[2].x) < 0.2 and abs(it[1].x - XL) < 1.5]
        clip = fitz.Rect(0, 40, 595, 160 if i == 0 else 92)
        page.show_pdf_page(clip, form, 0, clip=clip)
        # 쪽 경계에서 잘린 표의 위·아래 닫는 선을 그린다(Chrome은 쪽에서 나뉜 칸의 테두리를 열어 둔다)
        if vs:
            y0 = min(min(it[1].y, it[2].y) for it in vs); y1 = max(max(it[1].y, it[2].y) for it in vs)
            for y in (y0, y1):
                page.draw_line(fitz.Point(XL, y), fitz.Point(XR, y), color=(0, 0, 0), width=0.36)
        # 쪽 번호(서식과 같은 위치)
        tw = fitz.TextWriter(page.rect)
        s = f"- {i + 1} -"
        w = font.text_length(s, fontsize=9.5)
        tw.append(fitz.Point(297.5 - w / 2, 797), s, font=font, fontsize=9.5)
        tw.write_text(page)
    doc.subset_fonts()
    doc.save(out_pdf, garbage=4, deflate=True)
    return len(doc)


if __name__ == "__main__":
    h = build_html()
    raw = os.path.join(BUILD, "chrome_raw.pdf")
    chrome_pdf(h, raw)
    os.makedirs(os.path.dirname(OUT_PDF), exist_ok=True)
    n = decorate(raw, OUT_PDF)
    print("pages", n, OUT_PDF, round(os.path.getsize(OUT_PDF) / 1e6, 2), "MB")
