# AAVAIL revenue forecast

Forecast the next 30 days of revenue for one country, or for every country combined. The service is the deployment half of the AAVAIL invoice case study: ingest the daily transactions, compare a few forecasts, and serve the better one from a small Flask API.

The invoice files are the public `cs-train` set from [aavail/ai-workflow-capstone](https://github.com/aavail/ai-workflow-capstone). They are not stored in this repo. `scripts/fetch_data.py` downloads them.

## Run the tests

```bash
python run_tests.py
```

That one script runs the API, model, and logging suites. The tests write under temporary directories and use a `test-` filename prefix, so they do not read or write the production models in `models/prod-*` or the production logs.

## Run the API

Saved models for `all` and `united_kingdom` are already in `models/`. From the repo root:

```bash
python app.py
```

The server listens on port 8091.

```bash
curl "http://127.0.0.1:8091/predict?country=all&year=2019&month=6&day=1"
curl "http://127.0.0.1:8091/predict?country=united_kingdom&year=2019&month=6&day=1"
curl "http://127.0.0.1:8091/monitor?country=all"
curl "http://127.0.0.1:8091/logs?kind=predict"
```

`POST /train` rebuilds the models. It expects the invoice JSON in `data/cs-train` (or in `AAVAIL_DATA_DIR`).

## Rebuild from the invoices

```bash
python scripts/fetch_data.py
python model.py
python scripts/make_figures.py
```

`model.py` fits three forecasts on a time-ordered holdout, keeps the lowest-RMSE one, and writes `figures/model_vs_baseline.png`. `scripts/make_figures.py` writes the EDA charts.

## Docker

```bash
docker build -t aavail-revenue .
docker run --rm -p 8091:8091 aavail-revenue
docker run --rm aavail-revenue python run_tests.py
```

The image contains the code, the saved models, and the figures. Retraining inside the container needs the invoice directory mounted at `/app/data/cs-train`.

## What was compared

The target on day `t` is revenue from `t` through `t+29`. Features are earlier windows of revenue, views, and invoices. The last 25% of labeled days is the holdout (2019-02-09 through 2019-07-02, 144 days). Nothing in that window is used to fit the saved model or to choose the ridge penalty.

| Series | Baseline RMSE | Ridge RMSE | Random forest RMSE | Served model |
|---|---:|---:|---:|---|
| All countries | 27,609.20 | 27,777.41 | 61,576.94 | baseline |
| United Kingdom | 22,863.16 | 23,187.17 | 63,015.71 | baseline |

The baseline predicts the next 30 days as the previous 30 days of revenue. Ridge (penalty chosen by time-series cross-validation on the training window only) is close and does not improve on it. The random forest is worse. `figures/model_vs_baseline.png` plots the holdout actuals, the baseline, and the ridge forecast.

## Lab checklist

| Question | Where it is answered |
|---|---|
| Unit tests for the API | `unittests/test_api.py` calls `/predict` for `all` and for the United Kingdom, and checks rejected input |
| Unit tests for the model | `unittests/test_model.py` covers ingestion, the 30-day target, and the three-way comparison |
| Unit tests for the logging | `unittests/test_logging.py` |
| One script, and the tests pass | `python run_tests.py` |
| A way to monitor performance | `monitor.py` and `GET /monitor`. It rescores the holdout and reports RMSE against the baseline |
| Test reads and writes kept off the production models and logs | `test=True` uses `test-` files. Tests also set `AAVAIL_MODEL_DIR` and `AAVAIL_LOG_DIR` to a temporary directory. `test_reads_and_writes_stay_out_of_production` checks the production files are unchanged |
| Predictions for one country and for all countries | `GET /predict?country=united_kingdom` and `GET /predict?country=all` |
| Ingestion as a function or script | `ingest.load_invoices` and `scripts/fetch_data.py` |
| More than one model compared | baseline, ridge, and random forest, in `models/prod-comparison.json` |
| EDA used visualizations | `figures/eda_top_countries.png`, `figures/eda_daily_revenue.png`, `figures/eda_2018_monthly.png` |
| Container image | `Dockerfile` |
| A chart of the model against the baseline | `figures/model_vs_baseline.png` |
