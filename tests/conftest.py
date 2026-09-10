"""Shared pytest configuration for the KDA test suite."""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

FLA_ROOT = os.environ.get("FLA_ROOT", str(REPO_ROOT.parent / "fla"))
if os.path.isdir(FLA_ROOT) and FLA_ROOT not in sys.path:
    sys.path.insert(0, FLA_ROOT)

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "npu: tests that require NPU device")
    config.addinivalue_line("markers", "cuda: tests that require CUDA device")
    config.addinivalue_line("markers", "unit: backend-independent tests")
    config.addinivalue_line("markers", "slow: long-running tests")
