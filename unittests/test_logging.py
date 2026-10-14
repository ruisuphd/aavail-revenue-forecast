import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logging_utils import read_log, write_log
from paths import log_path
from unittests.support import ROOT, snapshot


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self._previous = os.environ.get("AAVAIL_LOG_DIR")
        os.environ["AAVAIL_LOG_DIR"] = os.path.join(self.tmp.name, "logs")
        self.production_logs = snapshot(os.path.join(ROOT, "logs"))

    def tearDown(self):
        if self._previous is None:
            os.environ.pop("AAVAIL_LOG_DIR", None)
        else:
            os.environ["AAVAIL_LOG_DIR"] = self._previous
        self.tmp.cleanup()

    def test_train_and_predict_logs_round_trip(self):
        write_log("train", {"country": "all", "model": "ridge", "rmse": 1.5}, test=True)
        write_log("predict", {"country": "united_kingdom", "y_pred": 10.0}, test=True)
        train_rows = read_log("train", test=True)
        predict_rows = read_log("predict", test=True)
        self.assertEqual(len(train_rows), 1)
        self.assertEqual(train_rows[0]["country"], "all")
        self.assertEqual(train_rows[0]["kind"], "train")
        self.assertTrue(train_rows[0]["test"])
        self.assertEqual(predict_rows[0]["country"], "united_kingdom")
        self.assertIn("ts", predict_rows[0])

    def test_test_logs_are_separate_files(self):
        write_log("train", {"country": "all"}, test=True)
        test_file = log_path("train", test=True)
        production_file = log_path("train", test=False)
        self.assertTrue(os.path.basename(test_file).startswith("test-"))
        self.assertTrue(os.path.exists(test_file))
        self.assertFalse(os.path.exists(production_file))
        self.assertEqual(snapshot(os.path.join(ROOT, "logs")), self.production_logs)

    def test_missing_log_is_empty(self):
        self.assertEqual(read_log("monitor", test=True), [])


if __name__ == "__main__":
    unittest.main()
