from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict
import html as _html
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

_MONTH_NUM: Dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

@dataclass(frozen=True)
class CalendarDay:
    year: int
    month_name: str
    month: int
    day: int
    day_type: str
    html: str         # исходный HTML (после unescape)
    text: str         # текст без HTML

def parse_nalog_calendar_xml(xml_bytes: bytes) -> tuple[Optional[str], List[CalendarDay]]:
    """
    Строго по structure-20140228.xsd:
      <calendar>
        <title?>...</title>
        <year index="2026">
          <month name="january">
            <day num="12" type="event"> &lt;p&gt;...&lt;/p&gt; </day>
    """
    root = ET.fromstring(xml_bytes)
    if root.tag != "calendar":
        raise ValueError(f"Unexpected root tag: {root.tag}")

    title_el = root.find("title")
    title = (title_el.text or "").strip() if title_el is not None and title_el.text else None
    if title == "":
        title = None

    out: List[CalendarDay] = []

    for y in root.findall("year"):
        idx = (y.attrib.get("index") or "").strip()
        if not idx:
            continue
        # xs:gYear -> "2026"
        year = int(idx)

        for m in y.findall("month"):
            mname = (m.attrib.get("name") or "").strip().lower()
            if not mname:
                continue
            mnum = _MONTH_NUM.get(mname)
            if not mnum:
                # на всякий: если появятся нетипичные значения
                continue

            for d in m.findall("day"):
                num_s = (d.attrib.get("num") or "").strip()
                typ = (d.attrib.get("type") or "").strip().lower()
                if not num_s or not typ:
                    continue
                day_num = int(num_s)

                # Внутри <day> лежит ЭКРАНИРОВАННЫЙ HTML (&lt;p&gt;...).
                raw = "".join(d.itertext()).strip()
                raw_html = _html.unescape(raw).strip()

                # Чистим HTML -> текст
                txt = ""
                if raw_html:
                    txt = BeautifulSoup(raw_html, "lxml").get_text(" ", strip=True)

                out.append(
                    CalendarDay(
                        year=year,
                        month_name=mname,
                        month=mnum,
                        day=day_num,
                        day_type=typ,
                        html=raw_html,
                        text=txt,
                    )
                )

    # сортировка (на всякий)
    out.sort(key=lambda x: (x.year, x.month, x.day))
    return title, out


def calendar_days_to_rag_text(title: Optional[str], days: List[CalendarDay]) -> str:
    """
    Делает “плоский” текст:
      YYYY-MM-DD [type] <text>
    По умолчанию индексируем только дни, где есть текст (обычно это event).
    """
    lines: List[str] = []
    if title:
        lines.append(title)
        lines.append("")  # пустая строка

    for d in days:
        if not d.text:
            continue
        lines.append(f"{d.year:04d}-{d.month:02d}-{d.day:02d} [{d.day_type}] {d.text}")

    return "\n".join(lines).strip()