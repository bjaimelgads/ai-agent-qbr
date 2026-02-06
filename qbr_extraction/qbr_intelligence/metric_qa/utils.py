"""Utility helpers for metric QA."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import re


_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


@dataclass(frozen=True)
class DateRange:
    start: date | None
    end: date | None


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower()) if text else []


def parse_year(value: str) -> int | None:
    if not value:
        return None
    try:
        year = int(value)
    except ValueError:
        return None
    if year < 100:
        return 2000 + year
    return year


def start_of_month(year: int, month: int) -> date:
    return date(year, month, 1)


def end_of_month(year: int, month: int) -> date:
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def add_months(anchor: date, months: int) -> date:
    year = anchor.year + (anchor.month - 1 + months) // 12
    month = (anchor.month - 1 + months) % 12 + 1
    day = min(anchor.day, end_of_month(year, month).day)
    return date(year, month, day)


def quarter_bounds(fiscal_year: int, quarter: int, fiscal_start_month: int) -> DateRange:
    fiscal_start_year = fiscal_year if fiscal_start_month == 1 else fiscal_year - 1
    start_month = ((quarter - 1) * 3 + fiscal_start_month - 1) % 12 + 1
    start_year = fiscal_start_year + ((quarter - 1) * 3 + fiscal_start_month - 1) // 12
    start = start_of_month(start_year, start_month)
    end = end_of_month(*_add_month_year(start_year, start_month, 2))
    return DateRange(start=start, end=end)


def half_bounds(fiscal_year: int, half: int, fiscal_start_month: int) -> DateRange:
    fiscal_start_year = fiscal_year if fiscal_start_month == 1 else fiscal_year - 1
    start_month = ((half - 1) * 6 + fiscal_start_month - 1) % 12 + 1
    start_year = fiscal_start_year + ((half - 1) * 6 + fiscal_start_month - 1) // 12
    start = start_of_month(start_year, start_month)
    end = end_of_month(*_add_month_year(start_year, start_month, 5))
    return DateRange(start=start, end=end)


def year_bounds(fiscal_year: int, fiscal_start_month: int) -> DateRange:
    fiscal_start_year = fiscal_year if fiscal_start_month == 1 else fiscal_year - 1
    start = start_of_month(fiscal_start_year, fiscal_start_month)
    end_month = ((fiscal_start_month - 1) + 11) % 12 + 1
    end_year = fiscal_start_year + ((fiscal_start_month - 1) + 11) // 12
    end = end_of_month(end_year, end_month)
    return DateRange(start=start, end=end)


def _add_month_year(year: int, month: int, offset: int) -> tuple[int, int]:
    total = (month - 1) + offset
    new_year = year + total // 12
    new_month = total % 12 + 1
    return new_year, new_month
