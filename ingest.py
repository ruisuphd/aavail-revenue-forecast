import os
import re

import pandas as pd

REQUIRED = [
    "country",
    "customer_id",
    "invoice",
    "price",
    "stream_id",
    "times_viewed",
    "year",
    "month",
    "day",
]

ALIASES = {
    "uk": "united_kingdom",
    "all_countries": "all",
}


def country_slug(name):
    slug = re.sub(r"\s+", "_", str(name).strip().lower())
    slug = ALIASES.get(slug, slug)
    if not re.fullmatch(r"[a-z0-9_]+", slug):
        raise ValueError(f"invalid country: {name}")
    return slug


def _normalize(frame, source):
    frame = frame.copy()
    if "price" in frame.columns and "total_price" in frame.columns:
        frame = frame.drop(columns=["total_price"])
    if "stream_id" in frame.columns and "StreamID" in frame.columns:
        frame = frame.drop(columns=["StreamID"])
    if "times_viewed" in frame.columns and "TimesViewed" in frame.columns:
        frame = frame.drop(columns=["TimesViewed"])
    frame = frame.rename(
        columns={
            "StreamID": "stream_id",
            "TimesViewed": "times_viewed",
            "total_price": "price",
        }
    )
    missing = [column for column in REQUIRED if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing columns: {', '.join(missing)}")

    keep = frame.loc[:, REQUIRED].copy()
    keep["price"] = pd.to_numeric(keep["price"], errors="coerce").fillna(0.0)
    keep["times_viewed"] = pd.to_numeric(keep["times_viewed"], errors="coerce").fillna(0.0)
    keep["invoice"] = keep["invoice"].astype(str).str.replace(r"\D+", "", regex=True)
    keep["country"] = keep["country"].astype(str)
    keep["stream_id"] = keep["stream_id"].astype(str)
    keep["invoice_date"] = pd.to_datetime(
        {
            "year": keep["year"].astype(int),
            "month": keep["month"].astype(int),
            "day": keep["day"].astype(int),
        }
    )
    return keep


def load_invoices(data_dir):
    """Load every invoice JSON file in ``data_dir`` into one table."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data directory does not exist: {data_dir}")
    names = sorted(name for name in os.listdir(data_dir) if name.endswith(".json"))
    if not names:
        raise FileNotFoundError(f"no JSON invoices in {data_dir}")

    frames = []
    for name in names:
        path = os.path.join(data_dir, name)
        frames.append(_normalize(pd.read_json(path), name))
    invoices = pd.concat(frames, ignore_index=True)
    return invoices.sort_values("invoice_date").reset_index(drop=True)


def select_country(invoices, country):
    slug = country_slug(country)
    if slug == "all":
        return invoices, slug
    slugs = invoices["country"].map(country_slug)
    chosen = invoices.loc[slugs == slug].copy()
    if chosen.empty:
        known = ", ".join(sorted(slugs.unique())[:12])
        raise ValueError(f"unknown country '{country}'. known examples: {known}")
    return chosen, slug
