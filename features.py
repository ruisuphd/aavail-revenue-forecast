import pandas as pd

FEATURE_COLUMNS = [
    "previous_7",
    "previous_14",
    "previous_28",
    "previous_70",
    "previous_30",
    "previous_year",
    "recent_invoices",
    "recent_views",
]
HORIZON = 30


def to_daily(invoices):
    """Aggregate transactions onto a continuous daily index."""
    if invoices.empty:
        raise ValueError("no invoices to aggregate")
    grouped = invoices.groupby("invoice_date", sort=True)
    daily = pd.DataFrame(
        {
            "purchases": grouped.size(),
            "revenue": grouped["price"].sum(),
            "total_views": grouped["times_viewed"].sum(),
            "unique_invoices": grouped["invoice"].nunique(),
            "unique_streams": grouped["stream_id"].nunique(),
        }
    )
    index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(index)
    daily.index.name = "date"
    for column in daily.columns:
        daily[column] = daily[column].fillna(0.0)
    return daily


def build_supervised(daily):
    """Build lagged features and a 30-day forward revenue target.

    The target on day ``t`` is revenue from ``t`` through ``t+29``.
    Lag features use only days before ``t``.
    """
    revenue = daily["revenue"].astype(float)
    frame = pd.DataFrame(index=daily.index)
    frame.index.name = "date"
    for window in (7, 14, 28, 70, 30):
        frame[f"previous_{window}"] = revenue.shift(1).rolling(window, min_periods=window).sum()
    # Same 30-day span one year earlier: t-365 through t-336.
    frame["previous_year"] = revenue.shift(336).rolling(30, min_periods=30).sum()
    frame["recent_invoices"] = daily["unique_invoices"].shift(1).rolling(30, min_periods=1).mean()
    frame["recent_views"] = daily["total_views"].shift(1).rolling(30, min_periods=1).mean()
    frame["target"] = revenue.iloc[::-1].rolling(HORIZON, min_periods=HORIZON).sum().iloc[::-1]
    return frame


def trainable(frame):
    labeled = frame.dropna(subset=["previous_7", "target"]).copy()
    labeled.loc[:, FEATURE_COLUMNS] = labeled.loc[:, FEATURE_COLUMNS].fillna(0.0)
    return labeled


def predictable(frame):
    rows = frame.dropna(subset=["previous_7"]).copy()
    rows.loc[:, FEATURE_COLUMNS] = rows.loc[:, FEATURE_COLUMNS].fillna(0.0)
    return rows
