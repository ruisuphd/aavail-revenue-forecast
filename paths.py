import os

ROOT = os.path.dirname(os.path.abspath(__file__))


def data_dir():
    return os.environ.get("AAVAIL_DATA_DIR", os.path.join(ROOT, "data", "cs-train"))


def model_dir():
    return os.environ.get("AAVAIL_MODEL_DIR", os.path.join(ROOT, "models"))


def log_dir():
    return os.environ.get("AAVAIL_LOG_DIR", os.path.join(ROOT, "logs"))


def figure_dir():
    return os.environ.get("AAVAIL_FIGURE_DIR", os.path.join(ROOT, "figures"))


def _prefix(test):
    return "test" if test else "prod"


def model_path(country, test=False):
    return os.path.join(model_dir(), f"{_prefix(test)}-{country}.joblib")


def feature_path(country, test=False):
    return os.path.join(model_dir(), f"{_prefix(test)}-{country}-features.csv")


def holdout_path(country, test=False):
    return os.path.join(model_dir(), f"{_prefix(test)}-{country}-holdout.csv")


def comparison_path(test=False):
    return os.path.join(model_dir(), f"{_prefix(test)}-comparison.json")


def log_path(kind, test=False):
    name = f"{'test-' if test else ''}{kind}.log"
    return os.path.join(log_dir(), name)
