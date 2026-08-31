"""The native PyQt6 desktop application.

This module contains widgets only.  Image rendering and difference detection
live in :mod:`pdf_differences_viewer.engine`; ``QGraphicsScene`` and
``QGraphicsView`` presentation live in :mod:`pdf_differences_viewer.graphics`.
No part of the application uses Qt WebEngine or a local HTTP server.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QColor, QFont, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from engine import ComparisonResult, DifferenceRegion, compare_pdf_pages
from graphics import ComparisonGraphicsWidget


class ComparisonWorker(QThread):
    """Run PDF rasterisation and comparison away from the GUI thread."""

    progress = pyqtSignal(int, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        old_pdf: str,
        new_pdf: str,
        page_index: int,
        dpi: float,
        tolerance_px: int,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.old_pdf = old_pdf
        self.new_pdf = new_pdf
        self.page_index = page_index
        self.dpi = dpi
        self.tolerance_px = tolerance_px

    def run(self) -> None:
        def report(stage: str, fraction: float) -> None:
            self.progress.emit(round(max(0.0, min(1.0, fraction)) * 100), stage.capitalize())

        try:
            result = compare_pdf_pages(
                self.old_pdf,
                self.new_pdf,
                self.page_index,
                dpi=self.dpi,
                tolerance_px=self.tolerance_px,
                progress=report,
            )
        except Exception as error:  # The original exception is shown to the user.
            self.failed.emit(str(error))
        else:
            self.completed.emit(result)


class ChangeList(QFrame):
    """Sidebar that represents added and removed regions as native Qt rows."""

    region_activated = pyqtSignal(str)
    region_double_activated = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("changePanel")
        self._selecting_from_scene = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 10, 14)
        layout.setSpacing(10)

        heading = QLabel("Detected changes")
        heading.setObjectName("panelHeading")
        layout.addWidget(heading)

        self.summary = QLabel("Compare two PDFs to see changed regions here.")
        self.summary.setObjectName("changeSummary")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.tree = QTreeWidget()
        self.tree.setObjectName("changeTree")
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(False)
        self.tree.setIndentation(0)
        self.tree.currentItemChanged.connect(self._on_current_item_changed)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.tree, stretch=1)

        self.alignment = QLabel()
        self.alignment.setObjectName("alignmentNote")
        self.alignment.setWordWrap(True)
        layout.addWidget(self.alignment)

    @staticmethod
    def _region_label(region: DifferenceRegion) -> str:
        x, y, width, height = region.bbox
        verb = "Added" if region.kind == "added" else "Removed"
        return f"{verb} · {region.area:,} px\n{x}, {y} · {width} × {height}"

    def set_result(self, result: ComparisonResult) -> None:
        self.tree.clear()
        regions: list[tuple[str, DifferenceRegion]] = []
        regions.extend((f"added:{index}", region) for index, region in enumerate(result.added_regions))
        regions.extend((f"removed:{index}", region) for index, region in enumerate(result.removed_regions))
        regions.sort(key=lambda entry: entry[1].area, reverse=True)

        if not regions:
            self.summary.setText("No changed ink pixels were detected at the current tolerance.")
        else:
            self.summary.setText(
                f"{len(regions)} regions · {result.added_pixels:,} added pixels · "
                f"{result.removed_pixels:,} removed pixels"
            )

        for ident, region in regions:
            item = QTreeWidgetItem([self._region_label(region)])
            item.setData(0, Qt.ItemDataRole.UserRole, ident)
            item.setToolTip(0, "Click to locate this region; double-click to zoom to it.")
            color = QColor("#be2a2a") if region.kind == "added" else QColor("#2463b5")
            item.setForeground(0, color)
            self.tree.addTopLevelItem(item)

        alignment = result.alignment
        if alignment.success and alignment.method != "resize":
            score = f" (score {alignment.score:.3f})" if alignment.score is not None else ""
            self.alignment.setText(f"Alignment: {alignment.method}{score}")
        elif alignment.message:
            self.alignment.setText(f"Alignment: {alignment.message}")
        else:
            self.alignment.setText("Alignment: page dimensions normalised")

    def select_region(self, ident: str) -> None:
        for row in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(row)
            if item.data(0, Qt.ItemDataRole.UserRole) == ident:
                self._selecting_from_scene = True
                self.tree.setCurrentItem(item)
                self.tree.scrollToItem(item)
                self._selecting_from_scene = False
                return

    def _on_current_item_changed(self, current: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None) -> None:
        if current is None or self._selecting_from_scene:
            return
        ident = current.data(0, Qt.ItemDataRole.UserRole)
        if ident:
            self.region_activated.emit(str(ident))

    def _on_item_double_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        ident = item.data(0, Qt.ItemDataRole.UserRole)
        if ident:
            self.region_double_activated.emit(str(ident))


class MainWindow(QMainWindow):
    """Main native desktop window for a two-PDF comparison session."""

    def __init__(self) -> None:
        super().__init__()
        self.worker: ComparisonWorker | None = None
        self._close_after_worker = False
        self.setWindowTitle("PDF Differences Viewer")
        self.resize(1440, 900)
        self.setMinimumSize(1050, 680)
        self._build_ui()
        self._show_placeholder()

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(12, 12, 12, 8)
        layout.setSpacing(10)

        layout.addWidget(self._make_file_controls())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.changes = ChangeList()
        self.changes.setMinimumWidth(245)
        self.changes.setMaximumWidth(380)
        splitter.addWidget(self.changes)
        self.viewer = ComparisonGraphicsWidget()
        splitter.addWidget(self.viewer)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([290, 1050])
        layout.addWidget(splitter, stretch=1)

        layout.addWidget(self._make_review_controls())

        status = QStatusBar()
        self.setStatusBar(status)
        self.status_label = QLabel("Ready")
        status.addWidget(self.status_label, stretch=1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedWidth(220)
        self.progress.setTextVisible(True)
        status.addPermanentWidget(self.progress)

        self.changes.region_activated.connect(self.viewer.focus_region)
        self.changes.region_double_activated.connect(self.viewer.fit_region)
        self.viewer.region_selected.connect(self.changes.select_region)
        self.viewer.region_double_clicked.connect(self._focus_and_fit_region)

    def _make_file_controls(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("fileControls")
        grid = QGridLayout(frame)
        grid.setContentsMargins(14, 12, 14, 12)
        grid.setHorizontalSpacing(9)
        grid.setVerticalSpacing(8)

        old_label = QLabel("Old PDF")
        old_label.setObjectName("fieldLabel")
        self.old_path = QLineEdit()
        self.old_path.setReadOnly(True)
        self.old_path.setPlaceholderText("Select the earlier drawing revision …")
        self.old_browse = QPushButton("Browse")
        self.old_browse.clicked.connect(lambda: self._choose_pdf(self.old_path, "Select old PDF"))

        new_label = QLabel("New PDF")
        new_label.setObjectName("fieldLabel")
        self.new_path = QLineEdit()
        self.new_path.setReadOnly(True)
        self.new_path.setPlaceholderText("Select the later drawing revision …")
        self.new_browse = QPushButton("Browse")
        self.new_browse.clicked.connect(lambda: self._choose_pdf(self.new_path, "Select new PDF"))

        grid.addWidget(old_label, 0, 0)
        grid.addWidget(self.old_path, 0, 1)
        grid.addWidget(self.old_browse, 0, 2)
        grid.addWidget(new_label, 1, 0)
        grid.addWidget(self.new_path, 1, 1)
        grid.addWidget(self.new_browse, 1, 2)

        options = QHBoxLayout()
        options.setSpacing(8)
        options.addWidget(QLabel("Page"))
        self.page_spin = QSpinBox()
        self.page_spin.setRange(1, 999)
        self.page_spin.setValue(1)
        self.page_spin.setToolTip("The page number to compare in both PDFs.")
        options.addWidget(self.page_spin)
        options.addWidget(QLabel("Render quality"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItem("Balanced (144 DPI)", 144.0)
        self.quality_combo.addItem("High (216 DPI)", 216.0)
        self.quality_combo.addItem("Detail (288 DPI)", 288.0)
        self.quality_combo.setToolTip("Higher DPI improves small details but uses more memory.")
        options.addWidget(self.quality_combo)
        options.addWidget(QLabel("Tolerance"))
        self.tolerance_spin = QSpinBox()
        self.tolerance_spin.setRange(0, 20)
        self.tolerance_spin.setValue(2)
        self.tolerance_spin.setSuffix(" px")
        self.tolerance_spin.setToolTip("Allow this many pixels of rasterisation/alignment variation before marking ink as changed.")
        options.addWidget(self.tolerance_spin)
        options.addStretch(1)
        self.compare_button = QPushButton("Compare PDFs")
        self.compare_button.setObjectName("compareButton")
        self.compare_button.clicked.connect(self._start_comparison)
        options.addWidget(self.compare_button)
        grid.addLayout(options, 2, 0, 1, 3)
        return frame

    def _make_review_controls(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("reviewControls")
        row = QHBoxLayout(frame)
        row.setContentsMargins(14, 8, 14, 8)
        row.setSpacing(8)

        old_button = QPushButton("Old")
        old_button.clicked.connect(lambda: self.blend_slider.setValue(0))
        difference_button = QPushButton("Differences")
        difference_button.clicked.connect(lambda: self.blend_slider.setValue(50))
        new_button = QPushButton("New")
        new_button.clicked.connect(lambda: self.blend_slider.setValue(100))
        row.addWidget(old_button)
        row.addWidget(difference_button)
        row.addWidget(new_button)

        self.blend_label = QLabel("Old  ←  50%  →  New")
        self.blend_label.setMinimumWidth(135)
        row.addWidget(self.blend_label)
        self.blend_slider = QSlider(Qt.Orientation.Horizontal)
        self.blend_slider.setRange(0, 100)
        self.blend_slider.setValue(50)
        self.blend_slider.setToolTip("Blend from the old page to the new page. Difference layers peak in the middle.")
        self.blend_slider.valueChanged.connect(self._set_blend)
        row.addWidget(self.blend_slider, stretch=1)

        self.added_toggle = QCheckBox("Additions")
        self.added_toggle.setChecked(True)
        self.added_toggle.toggled.connect(self.viewer.toggle_added)
        self.removed_toggle = QCheckBox("Removals")
        self.removed_toggle.setChecked(True)
        self.removed_toggle.toggled.connect(self.viewer.toggle_removed)
        self.annotation_toggle = QCheckBox("Regions")
        self.annotation_toggle.setChecked(True)
        self.annotation_toggle.toggled.connect(self.viewer.toggle_annotations)
        row.addWidget(self.added_toggle)
        row.addWidget(self.removed_toggle)
        row.addWidget(self.annotation_toggle)

        fit_button = QPushButton("Fit")
        fit_button.setToolTip("Fit the page into the view (F)")
        fit_button.clicked.connect(self.viewer.fit_to_content)
        one_to_one_button = QPushButton("1:1")
        one_to_one_button.setToolTip("Show image pixels at 1:1 scale (0)")
        one_to_one_button.clicked.connect(self.viewer.reset_view)
        row.addWidget(fit_button)
        row.addWidget(one_to_one_button)
        return frame

    def _show_placeholder(self) -> None:
        scene = self.viewer.scene
        scene.clear()
        scene.setSceneRect(0, 0, 820, 520)
        message = scene.addText("Select an old and a new PDF to start a native comparison")
        font = QFont()
        font.setPointSize(15)
        font.setWeight(QFont.Weight.DemiBold)
        message.setFont(font)
        message.setDefaultTextColor(QColor("#61738a"))
        message.setPos(120, 225)
        self.viewer.fit_to_content()

    def _choose_pdf(self, field: QLineEdit, title: str) -> None:
        selected, _filter = QFileDialog.getOpenFileName(self, title, "", "PDF files (*.pdf);;All files (*)")
        if selected:
            field.setText(selected)

    def _start_comparison(self) -> None:
        old_pdf = Path(self.old_path.text().strip())
        new_pdf = Path(self.new_path.text().strip())
        if not old_pdf.is_file() or not new_pdf.is_file():
            QMessageBox.warning(self, "Select PDFs", "Choose an existing old PDF and new PDF before starting the comparison.")
            return
        self._set_busy(True)
        self.progress.setValue(0)
        self.status_label.setText("Starting comparison …")
        self.worker = ComparisonWorker(
            str(old_pdf),
            str(new_pdf),
            self.page_spin.value() - 1,
            float(self.quality_combo.currentData()),
            self.tolerance_spin.value(),
            self,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.completed.connect(self._on_completed)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    def _set_busy(self, busy: bool) -> None:
        for widget in (
            self.old_browse,
            self.new_browse,
            self.compare_button,
            self.page_spin,
            self.quality_combo,
            self.tolerance_spin,
        ):
            widget.setEnabled(not busy)

    def _on_progress(self, value: int, stage: str) -> None:
        self.progress.setValue(value)
        self.status_label.setText(stage)

    def _on_completed(self, result: ComparisonResult) -> None:
        self.viewer.set_result(result)
        self.viewer.set_blend(self.blend_slider.value())
        self.changes.set_result(result)
        self.progress.setValue(100)
        differences = "no changed pixels" if not result.has_differences else f"{result.changed_pixels:,} changed pixels"
        self.status_label.setText(f"Comparison complete — {differences}")

    def _on_failed(self, error: str) -> None:
        self.progress.setValue(0)
        self.status_label.setText("Comparison failed")
        QMessageBox.critical(self, "Comparison failed", error)

    def _on_worker_finished(self) -> None:
        self._set_busy(False)
        if self._close_after_worker:
            self._close_after_worker = False
            self.close()

    def _set_blend(self, value: int) -> None:
        self.viewer.set_blend(value)
        self.blend_label.setText(f"Old  ←  {value}%  →  New")

    def _focus_and_fit_region(self, ident: str) -> None:
        self.changes.select_region(ident)
        self.viewer.fit_region(ident)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Never destroy the QThread while its OpenCV/PDF work is active."""
        if self.worker is not None and self.worker.isRunning():
            answer = QMessageBox.question(
                self,
                "Comparison in progress",
                "The PDF comparison is still running. Close this window automatically when it finishes?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer == QMessageBox.StandardButton.Yes:
                self._close_after_worker = True
                self.status_label.setText("Comparison still running — this window will close when it finishes.")
            event.ignore()
            return
        event.accept()


def main(argv: list[str] | None = None) -> int:
    application = QApplication(argv if argv is not None else sys.argv)
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
