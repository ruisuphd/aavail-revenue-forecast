import os

import joblib
import pandas as pd

from features import FEATURE_COLUMNS
from ingest import country_slug
from logging_utils import write_log
from model import MODEL_VERSION, regression_scores
from paths import feature_path, model_path


def monitor(country="all", test=False):
    """Score the saved model on its time-ordered holdout and compare it with the baseline."""
    slug = country_slug(country)
    path = model_path(slug, test=test)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no trained model for '{slug}'")
    bundle = joblib.load(path)
    table = pd.read_csv(feature_path(slug, test=test), parse_dates=["date"])
    labeled = table.dropna(subset=["target"]).sort_values("date")
    start = pd.Timestamp(bundle["holdout_start"])
    holdout = labeled.loc[labeled["date"] >= start]
    if holdout.empty:
        raise ValueError(f"no holdout rows for '{slug}'")

    predicted = bundle["estimator"].predict(holdout[FEATURE_COLUMNS].to_numpy(dtype=float))
    model_scores = regression_scores(holdout["target"], predicted)
    baseline_scores = regression_scores(holdout["target"], holdout["previous_30"])
    report = {
        "country": slug,
        "model": bundle["kind"],
        "version": MODEL_VERSION,
        "holdout_start": start.date().isoformat(),
        "holdout_end": holdout["date"].max().date().isoformat(),
        "holdout_rows": int(len(holdout)),
        "holdout_rmse": model_scores["rmse"],
        "holdout_mae": model_scores["mae"],
        "baseline_rmse": baseline_scores["rmse"],
        "baseline_mae": baseline_scores["mae"],
        "rmse_improvement": round(baseline_scores["rmse"] - model_scores["rmse"], 2),
    }
    write_log("monitor", report, test=test)
    return report
