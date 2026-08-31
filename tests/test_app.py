from __future__ import annotations

import pymupdf as fitz
from PyQt6.QtCore import QEventLoop, QTimer
from PyQt6.QtGui import QPalette

from pdf_differences_viewer.app import MainWindow, _apply_style


def test_main_window_uses_native_graphics_view(qapp) -> None:
    window = MainWindow()

    assert window.windowTitle() == "PDF Differences Viewer"
    assert window.viewer.view.scene() is window.viewer.scene
    assert window.viewer.scene.sceneRect().isValid()


def test_light_style_overrides_dark_system_control_colors(qapp) -> None:
    _apply_style(qapp)
    palette = qapp.palette()

    assert palette.color(QPalette.ColorRole.Button).name() == "#f8fafc"
    assert palette.color(QPalette.ColorRole.ButtonText).name() == "#17212e"
    assert palette.color(QPalette.ColorRole.WindowText).name() == "#17212e"
    assert palette.color(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText).name() == "#8a98a8"
    assert "QComboBox::drop-down" in qapp.styleSheet()


def _write_pdf(path, circle: bool) -> None:
    document = fitz.open()
    page = document.new_page(width=260, height=180)
    page.draw_rect(fitz.Rect(40, 40, 130, 100), color=(0, 0, 0), width=1.5)
    if circle:
        page.draw_circle((185, 125), 15, color=(0, 0, 0), width=1.5)
    document.save(path)
    document.close()


def test_main_window_receives_a_worker_result(qapp, tmp_path) -> None:
    old_pdf = tmp_path / "old.pdf"
    new_pdf = tmp_path / "new.pdf"
    _write_pdf(old_pdf, circle=False)
    _write_pdf(new_pdf, circle=True)
    window = MainWindow()
    window.old_path.setText(str(old_pdf))
    window.new_path.setText(str(new_pdf))
    window.show()
    qapp.processEvents()

    window._start_comparison()
    assert window.worker is not None
    event_loop = QEventLoop()
    window.worker.finished.connect(event_loop.quit)
    QTimer.singleShot(10_000, event_loop.quit)
    event_loop.exec()
    qapp.processEvents()

    assert not window.worker.isRunning()
    assert window.viewer.result is not None
    assert window.viewer.result.added_pixels > 0
