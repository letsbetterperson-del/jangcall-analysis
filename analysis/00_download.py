# -*- coding: utf-8 -*-
"""원자료 내려받기 (로그인·API 키 불필요). 실행: python analysis/00_download.py
- 공공데이터포털 파일데이터 6종(서울시설공단·한국교통안전공단)
- 행정동 경계(vuski/admdongkor, 통계청 행정동 경계 가공본, ver20251231)
- 일별 기상(Open-Meteo 과거 기상 API, 서울 시청 좌표)
받은 파일은 data/raw/ 에 원래 파일명 그대로 둔다.
"""
import json, os, re, sys, urllib.parse, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
os.makedirs(RAW, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"}

# (공공데이터포털 데이터 번호, 설명)
DATAGO = [
    ("15115859", "서울시설공단_장애인콜택시 탑승내역(2025, 호출 단위)"),
    ("15147613", "서울시설공단_장애인콜택시 시간대별 대기시간(2025)"),
    ("15147614", "서울시설공단_장애인콜택시 시간대별 탑승건수(2025)"),
    ("15144781", "서울시설공단_장애인콜택시 차고지 정보"),
    ("15160788", "서울시설공단_장애인콜택시 차량정보"),
    ("15066774", "한국교통안전공단_전국 대중교통 특별교통수단운행현황(2024)"),
]


def datago(pk):
    page = urllib.request.urlopen(urllib.request.Request(
        f"https://www.data.go.kr/data/{pk}/fileData.do", headers=UA), timeout=60).read().decode("utf8", "ignore")
    uddi = re.search(rf"fn_fileDataDown\('{pk}', '(uddi:[^']+)'", page).group(1)
    body = urllib.parse.urlencode({"publicDataDetailPk": uddi, "publicDataPk": pk, "atchFileId": "",
                                   "fileDetailSn": "1", "publicDataTyCode": "PR0051"}).encode()
    j = json.loads(urllib.request.urlopen(urllib.request.Request(
        "https://www.data.go.kr/tcs/dss/selectFileDataDownload.do", data=body, headers=UA), timeout=60).read())
    url = f"https://www.data.go.kr/cmm/cmm/fileDownload.do?atchFileId={j['atchFileId']}&fileDetailSn={j['fileDetailSn']}"
    resp = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=1200)
    fn = re.search(r'filename="?([^";]+)', resp.headers.get("Content-Disposition", "")).group(1)
    try:
        fn = fn.encode("latin1").decode("utf8")
    except Exception:
        pass
    path = os.path.join(RAW, urllib.parse.unquote(fn))
    with open(path, "wb") as f:
        f.write(resp.read())
    return path


def get(url, name):
    path = os.path.join(RAW, name)
    with open(path, "wb") as f:
        f.write(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=1200).read())
    return path


if __name__ == "__main__":
    skip = "--skip-existing" in sys.argv
    for pk, desc in DATAGO:
        print(pk, desc, "->", datago(pk))
    print(get("https://raw.githubusercontent.com/vuski/admdongkor/master/ver20251231/HangJeongDong_ver20251231.geojson",
              "HangJeongDong_ver20251231.geojson"))
    print(get("https://archive-api.open-meteo.com/v1/archive?latitude=37.5665&longitude=126.978"
              "&start_date=2025-01-01&end_date=2025-12-31"
              "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum&timezone=Asia%2FSeoul",
              "open-meteo_seoul_2025_daily.json"))
