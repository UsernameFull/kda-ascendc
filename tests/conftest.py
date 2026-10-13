import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

FLA_ROOT = os.environ.get("FLA_ROOT", str(ROOT.parent / "fla"))
if os.path.isdir(FLA_ROOT) and FLA_ROOT not in sys.path:
    sys.path.insert(0, FLA_ROOT)


def pytest_configure(config):
    config.addinivalue_line("markers", "npu: requires an Ascend NPU")
    config.addinivalue_line("markers", "unit: backend-independent tests")
    config.addinivalue_line("markers", "slow: long-running tests")
