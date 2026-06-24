"""
news.py
키 없는 RSS 피드에서 매크로/시장 뉴스 헤드라인을 수집한다 (서버에서 수집 → CORS 무관).
피드가 막히거나 형식이 바뀌면 graceful 하게 건너뛴다. 실패 시 빈 리스트.
"""
import re
import requests
import xml.etree.ElementTree as ET

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"}

# (소스명, RSS URL) — 키 불필요 공개 피드
FEEDS = [
    ("WSJ", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
    ("CNBC", "https://www.cnbc.com/id/20910258/device/rss/rss.html"),   # Economy
    ("CNBC", "https://www.cnbc.com/id/15839135/device/rss/rss.html"),   # Markets
    ("MarketWatch", "http://feeds.marketwatch.com/marketwatch/topstories/"),
]


def _clean(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", t, flags=re.S)
    t = re.sub(r"<[^>]+>", "", t)
    return t.strip()


def fetch_news(limit: int = 8) -> list:
    items = []
    for src, url in FEEDS:
        try:
            r = requests.get(url, headers=UA, timeout=8)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            for it in root.iter("item"):
                title = _clean(it.findtext("title"))
                link = (it.findtext("link") or "").strip()
                pub = (it.findtext("pubDate") or "").strip()
                if title and link:
                    items.append({"title": title[:150], "link": link, "source": src, "pub": pub})
            if len(items) >= limit * 3:
                break
        except Exception:
            continue
    # 제목 앞부분 기준 중복 제거
    seen, out = set(), []
    for it in items:
        k = it["title"][:48].lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out[:limit]
