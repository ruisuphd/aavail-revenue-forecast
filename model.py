import json
import os
import re
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from baseline import PreviousRevenueBaseline
from features import FEATURE_COLUMNS, build_supervised, predictable, to_daily, trainable
from ingest import load_invoices, select_country
from logging_utils import write_log
from paths import comparison_path, data_dir, feature_path, figure_dir, holdout_path, model_dir, model_path

MODEL_VERSION = "1.0"
HOLDOUT_FRACTION = 0.25


def regression_scores(y_true, y_pred):
    error = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return {
        "rmse": round(float(np.sqrt(np.mean(error ** 2))), 2),
        "mae": round(float(np.mean(np.abs(error))), 2),
    }


def make_estimator(kind, alpha=1.0):
    if kind == "baseline":
        return PreviousRevenueBaseline()
    if kind == "ridge":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ])
    if kind == "random_forest":
        return RandomForestRegressor(
            n_estimators=80,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=1,
        )
    raise ValueError(f"unknown model: {kind}")


def _select_ridge_alpha(x_train, y_train):
    """Pick the penalty using only the training window, in time order."""
    alphas = np.logspace(0, 5, 13)
    if len(y_train) < 20:
        return 10.0
    splits = 4 if len(y_train) >= 40 else 2
    splitter = TimeSeriesSplit(n_splits=splits)
    best_alpha = float(alphas[0])
    best_score = float("inf")
    for alpha in alphas:
        fold_scores = []
        for train_idx, valid_idx in splitter.split(x_train):
            estimator = make_estimator("ridge", alpha=float(alpha))
            estimator.fit(x_train[train_idx], y_train[train_idx])
            predicted = estimator.predict(x_train[valid_idx])
            error = y_train[valid_idx] - predicted
            fold_scores.append(float(np.sqrt(np.mean(error ** 2))))
        score = float(np.mean(fold_scores))
        if score < best_score:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha


def _split(labeled):
    if len(labeled) < 8:
        raise ValueError(f"need at least 8 labeled days, found {len(labeled)}")
    cut = int(len(labeled) * (1.0 - HOLDOUT_FRACTION))
    cut = min(max(cut, 1), len(labeled) - 1)
    return labeled.iloc[:cut], labeled.iloc[cut:]


def compare_models(labeled):
    train_rows, test_rows = _split(labeled)
    x_train = train_rows[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_rows["target"].to_numpy(dtype=float)
    x_test = test_rows[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test = test_rows["target"].to_numpy(dtype=float)

    alpha = _select_ridge_alpha(x_train, y_train)
    scores = {}
    predictions = {}
    fitted = {}
    for kind in ("baseline", "ridge", "random_forest"):
        estimator = make_estimator(kind, alpha=alpha)
        estimator.fit(x_train, y_train)
        predicted = estimator.predict(x_test)
        scores[kind] = regression_scores(y_test, predicted)
        predictions[kind] = predicted
        fitted[kind] = estimator
    winner = min(scores, key=lambda name: (scores[name]["rmse"], name))
    learned = min(("ridge", "random_forest"), key=lambda name: (scores[name]["rmse"], name))
    return {
        "scores": scores,
        "winner": winner,
        "learned": learned,
        "ridge_alpha": alpha,
        "estimator": fitted[winner],
        "train": train_rows,
        "test": test_rows,
        "predictions": predictions,
    }


def _parse_date(year, month, day):
    parts = {"year": year, "month": month, "day": day}
    numbers = {}
    for name, value in parts.items():
        text = str(value).strip()
        if not re.fullmatch(r"\d{1,4}", text):
            raise ValueError(f"{name} must be numeric")
        numbers[name] = int(text)
    try:
        return date(numbers["year"], numbers["month"], numbers["day"])
    except ValueError as exc:
        raise ValueError(f"invalid date: {exc}") from exc


def train_country(invoices, country, test=False):
    chosen, slug = select_country(invoices, country)
    daily = to_daily(chosen)
    frame = build_supervised(daily)
    labeled = trainable(frame)
    compared = compare_models(labeled)
    winner = compared["winner"]

    os.makedirs(model_dir(), exist_ok=True)
    bundle = {
        "estimator": compared["estimator"],
        "kind": winner,
        "features": FEATURE_COLUMNS,
        "country": slug,
        "version": MODEL_VERSION,
        "holdout_start": compared["test"].index[0].date().isoformat(),
    }
    joblib.dump(bundle, model_path(slug, test=test))

    serving = predictable(frame).reset_index()
    serving["date"] = serving["date"].dt.strftime("%Y-%m-%d")
    serving.to_csv(feature_path(slug, test=test), index=False)

    holdout = pd.DataFrame(
        {
            "date": compared["test"].index.strftime("%Y-%m-%d"),
            "actual": compared["test"]["target"].to_numpy(dtype=float),
            "baseline": compared["predictions"]["baseline"],
            "learned": compared["predictions"][compared["learned"]],
            "model": compared["predictions"][winner],
        }
    )
    holdout.to_csv(holdout_path(slug, test=test), index=False)

    summary = {
        "country": slug,
        "model": winner,
        "scores": compared["scores"],
        "holdout_rows": int(len(compared["test"])),
        "holdout_start": bundle["holdout_start"],
        "holdout_end": compared["test"].index[-1].date().isoformat(),
        "labeled_rows": int(len(labeled)),
        "learned_model": compared["learned"],
        "ridge_alpha": round(compared["ridge_alpha"], 2),
        "version": MODEL_VERSION,
    }
    write_log(
        "train",
        {
            "country": slug,
            "model": winner,
            "rmse": summary["scores"][winner]["rmse"],
            "baseline_rmse": summary["scores"]["baseline"]["rmse"],
            "version": MODEL_VERSION,
        },
        test=test,
    )
    return summary, holdout


def train(directory=None, countries=("all", "united_kingdom"), test=False):
    invoices = load_invoices(directory or data_dir())
    summaries = []
    holdouts = {}
    learned_names = {}
    for country in countries:
        summary, holdout = train_country(invoices, country, test=test)
        summaries.append(summary)
        holdouts[summary["country"]] = holdout
        learned_names[summary["country"]] = summary["learned_model"]

    with open(comparison_path(test=test), "w", encoding="utf-8") as handle:
        json.dump({"models": summaries}, handle, indent=2)

    if not test:
        from plots import write_model_comparison

        write_model_comparison(
            holdouts,
            os.path.join(figure_dir(), "model_vs_baseline.png"),
            learned_names=learned_names,
        )
    return summaries


def predict(country, year, month, day, test=False):
    slug_guess = country
    try:
        from ingest import country_slug

        slug_guess = country_slug(country)
    except ValueError:
        raise
    path = model_path(slug_guess, test=test)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no trained model for '{slug_guess}'")
    bundle = joblib.load(path)
    table = pd.read_csv(feature_path(bundle["country"], test=test), parse_dates=["date"])
    target = _parse_date(year, month, day)
    stamp = pd.Timestamp(target)
    matched = table.loc[table["date"] == stamp]
    if matched.empty:
        start = table["date"].min().date().isoformat()
        end = table["date"].max().date().isoformat()
        raise ValueError(f"date {target.isoformat()} is outside {start} to {end}")

    features = matched[FEATURE_COLUMNS].to_numpy(dtype=float)
    estimate = float(bundle["estimator"].predict(features)[0])
    baseline = float(matched["previous_30"].iloc[0])
    result = {
        "country": bundle["country"],
        "date": target.isoformat(),
        "y_pred": round(estimate, 2),
        "baseline": round(baseline, 2),
        "model": bundle["kind"],
    }
    write_log(
        "predict",
        {
            "country": result["country"],
            "date": result["date"],
            "y_pred": result["y_pred"],
            "model": result["model"],
            "version": MODEL_VERSION,
        },
        test=test,
    )
    return result


def main():
    summaries = train(data_dir(), countries=("all", "united_kingdom"), test=False)
    for summary in summaries:
        chosen = summary["model"]
        print(
            summary["country"],
            chosen,
            "rmse",
            summary["scores"][chosen]["rmse"],
            "baseline",
            summary["scores"]["baseline"]["rmse"],
        )


if __name__ == "__main__":
    main()
