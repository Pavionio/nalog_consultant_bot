#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
import time
import html as _html
from typing import Optional, Dict, List
from urllib.parse import urljoin
import datetime as dt
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET


UA = "nalog-consultant-bot/0.1"
BASE = "https://www.nalog.gov.ru/rn77/opendata/7707329152-kalendar/"
TIMEOUT = 60.0
SLEEP = 0.3

_MONTH_NUM: Dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}


def session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Accept-Language": "ru,en;q=0.8"})
    return s


def fetch(s: requests.Session, url: str) -> bytes:
    r = s.get(url, timeout=TIMEOUT)
    time.sleep(SLEEP)
    r.raise_for_status()
    return r.content


DATE_IN_NAME = re.compile(r"data-(\d{8})-structure-\d+\.xml", re.I)

def _parse_d8(d8: str) -> dt.date | None:
    if not (d8 and len(d8) == 8 and d8.isdigit()):
        return None
    # YYYYMMDD vs DDMMYYYY
    try:
        y = int(d8[:4])
        if 2000 <= y <= 2099:
            return dt.datetime.strptime(d8, "%Y%m%d").date()
        return dt.datetime.strptime(d8, "%d%m%Y").date()
    except Exception:
        return None

def pick_latest_data_link(page_html: bytes, base_url: str) -> str:
    soup = BeautifulSoup(page_html, "lxml")

    # 1) Самый надежный способ для nalog.gov.ru: строка таблицы с №=8
    for tr in soup.select("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        left = tds[0].get_text(" ", strip=True).replace("\xa0", " ").strip()
        if left == "8":
            a = tds[1].select_one("a[href]")
            if a and (a.get("href") or "").strip():
                return urljoin(base_url, a["href"].strip())

    # 2) Fallback: выбираем max по дате среди всех data-XXXXXXXX-structure-*.xml
    best_url = None
    best_dt = None

    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        abs_url = urljoin(base_url, href)

        m = DATE_IN_NAME.search(abs_url)
        if not m:
            continue

        d = _parse_d8(m.group(1))
        if not d:
            continue

        if best_dt is None or d > best_dt:
            best_dt = d
            best_url = abs_url

    if not best_url:
        # быстрый дебаг: покажем 20 первых href (потом уберёшь)
        hrefs = [x.get("href") for x in soup.select("a[href]")[:20]]
        raise RuntimeError(f"Не нашёл ссылку на data-*-structure-*.xml. Примеры href: {hrefs}")

    return best_url

def parse_calendar_xml(xml_bytes: bytes) -> tuple[Optional[str], List[dict]]:
    root = ET.fromstring(xml_bytes)
    if root.tag != "calendar":
        raise ValueError(f"Unexpected root tag: {root.tag}")

    title_el = root.find("title")
    title = (title_el.text or "").strip() if title_el is not None and title_el.text else None
    if title == "":
        title = None

    out: List[dict] = []
    for y in root.findall("year"):
        idx = (y.attrib.get("index") or "").strip()
        if not idx:
            continue
        year = int(idx)

        for m in y.findall("month"):
            mname = (m.attrib.get("name") or "").strip().lower()
            if not mname:
                continue
            mnum = _MONTH_NUM.get(mname)
            if not mnum:
                continue

            for d in m.findall("day"):
                num_s = (d.attrib.get("num") or "").strip()
                typ = (d.attrib.get("type") or "").strip().lower()
                if not num_s or not typ:
                    continue
                day_num = int(num_s)

                raw = "".join(d.itertext()).strip()
                raw_html = _html.unescape(raw).strip()
                txt = BeautifulSoup(raw_html, "lxml").get_text(" ", strip=True) if raw_html else ""

                out.append(
                    dict(
                        year=year, month=mnum, month_name=mname,
                        day=day_num, type=typ,
                        text=txt, html=raw_html,
                    )
                )

    out.sort(key=lambda x: (x["year"], x["month"], x["day"]))
    return title, out


def main() -> None:
    s = session()

    print("1) Fetch page:", BASE)
    page = fetch(s, BASE)

    print("2) Pick latest XML link…")
    xml_url = pick_latest_data_link(page, BASE)
    print("   latest:", xml_url)

    print("3) Download XML…")
    xml = fetch(s, xml_url)
    print("   bytes:", len(xml))

    print("4) Parse XML…")
    title, days = parse_calendar_xml(xml)
    types = sorted({d["type"] for d in days})
    print("   title:", title)
    print("   total day nodes:", len(days))
    print("   types:", types)

    events = [d for d in days if d["type"] == "event" and d["text"]]
    holidays = [d for d in days if d["type"] == "holiday"]
    print("   events:", len(events))
    print("   holidays:", len(holidays))

    print("\n5) First 10 events:")
    for e in events[:10]:
        print(f'   {e["year"]:04d}-{e["month"]:02d}-{e["day"]:02d} [{e["type"]}] {e["text"][:180]}')

    print("\nOK")


if __name__ == "__main__":
    main()
