from __future__ import annotations

import os

# Tests use the platform plugin before a QApplication is created, so set this
# during module import rather than in a fixture.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    return QApplication.instance() or QApplication([])
