import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from model import train
from unittests.support import ROOT, snapshot, write_fixture

PREDICT_DATE = {"year": "2018", "month": "3", "day": "2"}


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        self._previous = {
            key: os.environ.get(key)
            for key in ("AAVAIL_MODEL_DIR", "AAVAIL_LOG_DIR", "AAVAIL_DATA_DIR", "AAVAIL_FIGURE_DIR")
        }
        os.environ["AAVAIL_MODEL_DIR"] = os.path.join(root, "models")
        os.environ["AAVAIL_LOG_DIR"] = os.path.join(root, "logs")
        os.environ["AAVAIL_DATA_DIR"] = os.path.join(root, "data")
        os.environ["AAVAIL_FIGURE_DIR"] = os.path.join(root, "figures")
        os.makedirs(os.environ["AAVAIL_DATA_DIR"])
        write_fixture(os.environ["AAVAIL_DATA_DIR"])
        self.production_models = snapshot(os.path.join(ROOT, "models"))
        self.production_logs = snapshot(os.path.join(ROOT, "logs"))
        train(os.environ["AAVAIL_DATA_DIR"], countries=("all", "united_kingdom"), test=True)
        self.client = create_app().test_client()

    def tearDown(self):
        for key, value in self._previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self.tmp.cleanup()

    def test_predict_all_and_one_country(self):
        combined = self.client.get("/predict", query_string={"country": "all", "test": "true", **PREDICT_DATE})
        uk = self.client.get(
            "/predict",
            query_string={"country": "United Kingdom", "test": "true", **PREDICT_DATE},
        )
        self.assertEqual(combined.status_code, 200)
        self.assertEqual(uk.status_code, 200)
        combined_body = combined.get_json()
        uk_body = uk.get_json()
        self.assertEqual(combined_body["country"], "all")
        self.assertEqual(uk_body["country"], "united_kingdom")
        self.assertEqual(combined_body["date"], "2018-03-02")
        self.assertIsInstance(combined_body["y_pred"], float)
        self.assertIsInstance(uk_body["y_pred"], float)
        self.assertNotEqual(combined_body["y_pred"], uk_body["y_pred"])

    def test_predict_rejects_bad_input(self):
        missing = self.client.get("/predict", query_string={"country": "all"})
        self.assertEqual(missing.status_code, 400)
        bad_date = self.client.get(
            "/predict",
            query_string={"country": "all", "year": "2018", "month": "xx", "day": "1", "test": "true"},
        )
        self.assertEqual(bad_date.status_code, 400)
        unknown = self.client.get(
            "/predict",
            query_string={"country": "singapore", "test": "true", **PREDICT_DATE},
        )
        self.assertEqual(unknown.status_code, 400)

    def test_logs_and_monitor(self):
        self.client.get("/predict", query_string={"country": "all", "test": "true", **PREDICT_DATE})
        logs = self.client.get("/logs", query_string={"kind": "predict", "test": "true"})
        self.assertEqual(logs.status_code, 200)
        entries = logs.get_json()["entries"]
        self.assertTrue(any(entry["country"] == "all" for entry in entries))
        self.assertTrue(all(entry["test"] for entry in entries))

        report = self.client.get("/monitor", query_string={"country": "all", "test": "true"})
        self.assertEqual(report.status_code, 200)
        body = report.get_json()
        self.assertIn("holdout_rmse", body)
        self.assertIn("baseline_rmse", body)
        self.assertGreater(body["holdout_rows"], 0)

    def test_reads_and_writes_stay_out_of_production(self):
        names = os.listdir(os.environ["AAVAIL_MODEL_DIR"])
        self.assertTrue(names)
        self.assertTrue(all(name.startswith("test-") for name in names))
        self.assertFalse(any(name.startswith("prod-") for name in names))
        self.assertEqual(snapshot(os.path.join(ROOT, "models")), self.production_models)
        self.assertEqual(snapshot(os.path.join(ROOT, "logs")), self.production_logs)


if __name__ == "__main__":
    unittest.main()
