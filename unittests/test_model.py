import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import build_supervised, to_daily
from ingest import load_invoices
from model import compare_models, train
from unittests.support import write_fixture


class ModelTests(unittest.TestCase):
    def test_ingestion_normalizes_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            write_fixture(directory)
            invoices = load_invoices(directory)
        self.assertIn("price", invoices.columns)
        self.assertIn("stream_id", invoices.columns)
        self.assertNotIn("total_price", invoices.columns)
        self.assertTrue(invoices["invoice"].str.fullmatch(r"\d+").all())
        self.assertEqual(invoices["invoice_date"].min(), pd.Timestamp("2018-01-01"))
        span = (invoices["invoice_date"].max() - invoices["invoice_date"].min()).days
        self.assertEqual(span, 119)

    def test_target_is_the_next_30_days(self):
        index = pd.date_range("2018-01-01", periods=40, freq="D")
        daily = pd.DataFrame(
            {
                "revenue": np.arange(40, dtype=float),
                "purchases": 1,
                "total_views": 1,
                "unique_invoices": 1,
                "unique_streams": 1,
            },
            index=index,
        )
        frame = build_supervised(daily)
        self.assertEqual(frame.loc[index[0], "target"], sum(range(30)))
        self.assertEqual(frame.loc[index[7], "previous_7"], sum(range(7)))
        self.assertEqual(frame.loc[index[30], "previous_30"], sum(range(30)))

    def test_compare_models_scores_more_than_one_model(self):
        with tempfile.TemporaryDirectory() as directory:
            data = os.path.join(directory, "data")
            os.makedirs(data)
            write_fixture(data)
            invoices = load_invoices(data)
            daily = to_daily(invoices)
            labeled = build_supervised(daily).dropna(subset=["previous_7", "target"]).copy()
            labeled = labeled.fillna(0.0)
            compared = compare_models(labeled)
        self.assertGreaterEqual(len(compared["scores"]), 2)
        self.assertIn(compared["winner"], compared["scores"])
        for scores in compared["scores"].values():
            self.assertIn("rmse", scores)
            self.assertGreaterEqual(scores["rmse"], 0)

    def test_training_writes_only_test_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            data = os.path.join(directory, "data")
            os.makedirs(data)
            write_fixture(data)
            previous = {key: os.environ.get(key) for key in ("AAVAIL_MODEL_DIR", "AAVAIL_LOG_DIR")}
            os.environ["AAVAIL_MODEL_DIR"] = os.path.join(directory, "models")
            os.environ["AAVAIL_LOG_DIR"] = os.path.join(directory, "logs")
            try:
                summaries = train(data, countries=("all", "united_kingdom"), test=True)
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            names = os.listdir(os.path.join(directory, "models"))
            self.assertTrue(any(name.startswith("test-all") for name in names))
            self.assertTrue(any(name.startswith("test-united_kingdom") for name in names))
            self.assertFalse(any(name.startswith("prod-") for name in names))
            self.assertEqual({item["country"] for item in summaries}, {"all", "united_kingdom"})
            self.assertIn(summaries[0]["model"], summaries[0]["scores"])
            self.assertIn("ridge", summaries[0]["scores"])


if __name__ == "__main__":
    unittest.main()
