import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from features import to_daily
from ingest import load_invoices
from paths import data_dir, figure_dir
from plots import write_daily_revenue, write_monthly_activity, write_top_countries


def main():
    invoices = load_invoices(data_dir())
    directory = figure_dir()
    totals = invoices.groupby("country")["price"].sum().sort_values(ascending=False).head(10)
    write_top_countries(totals, os.path.join(directory, "eda_top_countries.png"))

    daily = to_daily(invoices)
    write_daily_revenue(daily, os.path.join(directory, "eda_daily_revenue.png"))

    year = invoices.loc[invoices["invoice_date"].dt.year == 2018].copy()
    year["month"] = year["invoice_date"].dt.month
    monthly = year.groupby("month").agg(
        revenue=("price", "sum"),
        purchases=("invoice", "size"),
        total_views=("times_viewed", "sum"),
    )
    write_monthly_activity(monthly, os.path.join(directory, "eda_2018_monthly.png"))
    print(f"wrote figures to {directory}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
