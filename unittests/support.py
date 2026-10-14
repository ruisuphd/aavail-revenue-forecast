import json
import os
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_fixture(directory):
    start = date(2018, 1, 1)
    uk_rows = []
    eire_rows = []
    for offset in range(120):
        current = start + timedelta(days=offset)
        uk_rows.append(
            {
                "country": "United Kingdom",
                "customer_id": 100,
                "invoice": f"A{5000 + offset}B",
                "price": 8 + (offset % 5),
                "stream_id": "s1",
                "times_viewed": 3,
                "year": current.year,
                "month": current.month,
                "day": current.day,
            }
        )
        if offset % 3 == 0:
            eire_rows.append(
                {
                    "country": "EIRE",
                    "customer_id": 7,
                    "invoice": str(8000 + offset),
                    "total_price": 2.5,
                    "StreamID": "e1",
                    "TimesViewed": 1,
                    "year": str(current.year),
                    "month": str(current.month),
                    "day": str(current.day),
                }
            )
    with open(os.path.join(directory, "invoices-uk.json"), "w", encoding="utf-8") as handle:
        json.dump(uk_rows, handle)
    with open(os.path.join(directory, "invoices-eire.json"), "w", encoding="utf-8") as handle:
        json.dump(eire_rows, handle)


def snapshot(directory):
    if not os.path.isdir(directory):
        return None
    found = {}
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            with open(path, "rb") as handle:
                found[name] = handle.read()
    return found
