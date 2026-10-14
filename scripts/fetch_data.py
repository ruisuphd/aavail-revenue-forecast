import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = "https://github.com/aavail/ai-workflow-capstone.git"


def main():
    dest = os.path.join(ROOT, "data", "cs-train")
    if os.path.isdir(dest) and any(name.endswith(".json") for name in os.listdir(dest)):
        print(f"invoices already present in {dest}")
        return 0
    checkout = os.path.join(ROOT, "data", "src")
    if os.path.exists(checkout):
        shutil.rmtree(checkout)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    subprocess.check_call(["git", "clone", "--depth", "1", SOURCE, checkout])
    shutil.copytree(os.path.join(checkout, "cs-train"), dest)
    shutil.rmtree(checkout)
    print(f"saved invoices to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
