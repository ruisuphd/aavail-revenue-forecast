import os

from flask import Flask, jsonify, request, send_from_directory

from logging_utils import read_log
from model import predict, train
from monitor import monitor
from paths import data_dir, figure_dir

PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AAVAIL revenue forecast</title>
  <style>
    body { font: 16px/1.5 "Iowan Old Style", Georgia, serif; color: #1f2933;
           max-width: 760px; margin: 2rem auto; padding: 0 1.2rem; }
    code { font-family: ui-monospace, monospace; font-size: 0.9rem;
           background: #f3f4f6; padding: 0.1rem 0.3rem; }
    img { width: 100%; height: auto; margin: 1rem 0 1.5rem; }
    h1 { font-weight: 600; letter-spacing: -0.02em; }
  </style>
</head>
<body>
  <h1>AAVAIL revenue forecast</h1>
  <p>Next-30-day revenue for one country, or for every country combined.
     The call takes a country and a date. Lagged revenue is already on the server.</p>
  <ul>
    <li><code>GET /predict?country=all&amp;year=2019&amp;month=6&amp;day=1</code></li>
    <li><code>GET /predict?country=united_kingdom&amp;year=2019&amp;month=6&amp;day=1</code></li>
    <li><code>GET /monitor?country=all</code></li>
    <li><code>GET /logs?kind=predict</code></li>
    <li><code>POST /train</code></li>
  </ul>
  <img src="/figures/model_vs_baseline.png" alt="Holdout forecast versus the previous-30-day baseline">
  <img src="/figures/eda_2018_monthly.png" alt="2018 monthly revenue, purchases, and views">
</body>
</html>
"""


def create_app():
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.get("/")
    def home():
        return PAGE

    @app.get("/figures/<path:name>")
    def figures(name):
        return send_from_directory(figure_dir(), name)

    @app.get("/predict")
    def predict_route():
        country = request.args.get("country")
        year = request.args.get("year")
        month = request.args.get("month")
        day = request.args.get("day")
        if not all([country, year, month, day]):
            return jsonify({"error": "country, year, month, and day are required"}), 400
        try:
            result = predict(
                country,
                year,
                month,
                day,
                test=request.args.get("test") == "true",
            )
        except (ValueError, FileNotFoundError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(result)

    @app.post("/train")
    def train_route():
        body = request.get_json(silent=True) or {}
        countries = body.get("countries") or ["all", "united_kingdom"]
        try:
            summaries = train(data_dir(), countries=countries, test=bool(body.get("test", False)))
        except (ValueError, FileNotFoundError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"models": summaries})

    @app.get("/logs")
    def logs_route():
        kind = request.args.get("kind", "predict")
        if kind not in {"train", "predict", "monitor"}:
            return jsonify({"error": "kind must be train, predict, or monitor"}), 400
        return jsonify({"kind": kind, "entries": read_log(kind, test=request.args.get("test") == "true")})

    @app.get("/monitor")
    def monitor_route():
        try:
            report = monitor(
                request.args.get("country", "all"),
                test=request.args.get("test") == "true",
            )
        except (ValueError, FileNotFoundError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(report)

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8091"))
    app.run(host="0.0.0.0", port=port)
