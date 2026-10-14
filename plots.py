import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def _panel(ax, frame, title, learned_name):
    dates = pd.to_datetime(frame["date"])
    ax.plot(dates, frame["actual"], color="#1f2933", lw=1.6, label="Actual")
    ax.plot(dates, frame["baseline"], color="#9aa5b1", lw=1.4, label="Baseline (previous 30 days)")
    ax.plot(dates, frame["learned"], color="#c2410c", lw=1.4, label=learned_name.replace("_", " "))
    ax.set_title(title)
    ax.set_ylabel("Next 30-day revenue")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def write_model_comparison(holdouts, path, learned_names=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    names = list(holdouts)
    fig, axes = plt.subplots(len(names), 1, figsize=(9.2, 3.3 * len(names)), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        learned = (learned_names or {}).get(name, "learned model")
        title = name.replace("_", " ")
        _panel(ax, holdouts[name], title, learned)
    fig.suptitle("Holdout: learned model against the previous-30-day baseline", y=1.02)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def write_top_countries(totals, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ordered = totals.sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(ordered.index, ordered.values, color="#1f4e79")
    ax.set_xlabel("Revenue")
    ax.set_title("Total revenue by country")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_daily_revenue(daily, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rolling = daily["revenue"].rolling(7, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    ax.plot(daily.index, daily["revenue"], color="#d9e2ec", lw=0.8, label="Daily revenue")
    ax.plot(daily.index, rolling, color="#1f4e79", lw=1.6, label="7-day mean")
    ax.set_title("Daily revenue, all countries")
    ax.set_ylabel("Revenue")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def write_monthly_activity(monthly, path):
    """``monthly`` is indexed by month number 1-12 and has revenue, purchases, total_views."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    colors = ["#9aa5b1"] * 10 + ["#c2410c", "#c2410c"]
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 7.2), sharex=True)
    series = (
        ("revenue", "Revenue"),
        ("purchases", "Purchases"),
        ("total_views", "Views"),
    )
    positions = list(range(1, 13))
    for ax, (column, title) in zip(axes, series):
        values = [float(monthly.loc[month, column]) if month in monthly.index else 0 for month in positions]
        ax.bar(labels, values, color=colors)
        ax.set_ylabel(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_title("2018 activity by month (November and December highlighted)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
