"""Native Qt graphics view for comparing two rendered PDF pages.

The module intentionally contains no QtWebEngine dependency.  Images supplied by
``engine.ComparisonResult`` are painted as pixmaps in a QGraphicsScene, making
zooming and reviewing inexpensive even for large rendered pages.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PyQt6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QPointF,
    QPropertyAnimation,
    QRectF,
    QTimer,
    Qt,
    pyqtProperty,
    pyqtSignal,
)
from PyQt6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView, QWidget

from pdf_differences_viewer.colors import DifferenceColors


class ChangeBoxPulseSettings:
    """Tunable animation settings for the pulsing change rectangles.

    Increase ``PERIOD_MS`` for a slower pulse or decrease it for a faster one.
    Qt's ``QPropertyAnimation`` handles interpolation and repaint timing.
    """

    PERIOD_MS = 2_000
    MIN_FILL_ALPHA = 18
    MAX_FILL_ALPHA = 82
    MIN_OUTLINE_ALPHA = 145
    MAX_OUTLINE_ALPHA = 255
    MIN_PEN_WIDTH = 1.5
    MAX_PEN_WIDTH = 3.25


def bgra_to_pixmap(array: np.ndarray) -> QPixmap:
    """Convert a BGRA (or RGB/BGR) numpy image to a detached Qt pixmap safely."""
    image = np.asarray(array)
    if image.ndim != 3 or image.shape[2] not in (3, 4) or image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("image must be a non-empty HxWx3 or HxWx4 array")
    # Qt's byte-oriented QImage formats require uint8; make a private,
    # contiguous buffer so a temporary numpy view can never be referenced.
    image = np.ascontiguousarray(image.astype(np.uint8, copy=False))
    rgba = np.ascontiguousarray(image[:, :, [2, 1, 0, 3]] if image.shape[2] == 4 else image[:, :, [2, 1, 0]])
    fmt = QImage.Format.Format_RGBA8888 if image.shape[2] == 4 else QImage.Format.Format_RGB888
    qimage = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], fmt).copy()
    return QPixmap.fromImage(qimage)


class DifferenceGraphicsView(QGraphicsView):
    """Graphics view with Ctrl-wheel zoom, scroll-wheel navigation, and fit keys."""

    item_clicked = pyqtSignal(str)
    item_double_clicked = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._press_scene = QPointF()
        self._press_view = QPointF()
        self._dragging = False

    def wheelEvent(self, event: Any) -> None:
        if not event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            super().wheelEvent(event)
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        next_scale = self.transform().m11() * factor
        if not 0.02 <= next_scale <= 100:
            event.accept()
            return
        old_pos = self.mapToScene(event.position().toPoint())
        self.scale(factor, factor)
        new_pos = self.mapToScene(event.position().toPoint())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
        event.accept()

    def mousePressEvent(self, event: Any) -> None:
        self._press_scene = self.mapToScene(event.position().toPoint())
        self._press_view = event.position()
        self._dragging = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:
        if (event.position() - self._press_view).manhattanLength() > 4:
            self._dragging = True
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        if not self._dragging and event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.position().toPoint())
            ident = item.data(0) if item is not None else None
            if ident is not None:
                self.item_clicked.emit(str(ident))
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: Any) -> None:
        item = self.itemAt(event.position().toPoint())
        ident = item.data(0) if item is not None else None
        if ident is not None:
            self.item_double_clicked.emit(str(ident))
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: Any) -> None:
        if event.key() == Qt.Key.Key_F:
            self.fitInView(self.scene().itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
            event.accept()
        elif event.key() == Qt.Key.Key_0:
            self.resetTransform()
            self.centerOn(self.scene().sceneRect().center())
            event.accept()
        else:
            super().keyPressEvent(event)


class ComparisonGraphicsWidget(QWidget):
    """Scene/controller widget that presents a :class:`ComparisonResult`."""

    region_selected = pyqtSignal(str)
    region_double_clicked = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        from PyQt6.QtWidgets import QVBoxLayout
        self.view = DifferenceGraphicsView(self)
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view)
        self._result = None
        self._items: dict[str, QGraphicsRectItem] = {}
        self._box_colors: dict[str, QColor] = {}
        self._layers: dict[str, QGraphicsPixmapItem] = {}
        self._blend = 50
        self._toggles = {"added": True, "removed": True, "annotations": True, "moved": True}
        self._fit_pending = False
        self._pulse_strength = 0.0
        self._pulse_animation = QPropertyAnimation(self, b"pulse_strength", self)
        self._pulse_animation.setDuration(max(1, ChangeBoxPulseSettings.PERIOD_MS))
        self._pulse_animation.setStartValue(0.0)
        self._pulse_animation.setKeyValueAt(0.5, 1.0)
        self._pulse_animation.setEndValue(0.0)
        self._pulse_animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._pulse_animation.setLoopCount(-1)
        self.view.item_clicked.connect(self._on_click)
        self.view.item_double_clicked.connect(self._on_double_click)

    @property
    def result(self):
        return self._result

    def set_result(self, result: Any | None) -> None:
        self._result = result
        self._pulse_animation.stop()
        self._pulse_animation.setCurrentTime(0)
        self.scene.clear(); self._items.clear(); self._box_colors.clear(); self._layers.clear()
        if result is None:
            return
        for z_value, (name, image) in enumerate((("old", result.old_bgra), ("new", result.new_bgra), ("added", result.added_layer), ("removed", result.removed_layer))):
            item = QGraphicsPixmapItem(bgra_to_pixmap(image)); item.setData(0, name); item.setZValue(z_value)
            self.scene.addItem(item); self._layers[name] = item
        self.scene.setSceneRect(QRectF(0, 0, result.width, result.height))
        for kind, regions in (("added", result.added_regions), ("removed", result.removed_regions)):
            for index, region in enumerate(regions):
                x, y, w, h = region.bbox; ident = f"{kind}:{index}"
                rect = QGraphicsRectItem(QRectF(x, y, w, h)); rect.setData(0, ident); rect.setZValue(10)
                color = QColor(*DifferenceColors.ADDITION_RGB) if kind == "added" else QColor(*DifferenceColors.REMOVAL_RGB)
                rect.setAcceptHoverEvents(True)
                rect.setCursor(Qt.CursorShape.PointingHandCursor)
                rect.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable, True)
                self.scene.addItem(rect)
                self._items[ident] = rect
                self._box_colors[ident] = color
        self._apply_state(); self._request_initial_fit()

    def set_blend(self, value: int) -> None:
        self._blend = max(0, min(100, int(value))); self._apply_state()

    def blend(self) -> int:
        return self._blend

    def _apply_state(self) -> None:
        t = self._blend / 100.0; smooth = t * t * (3 - 2 * t)
        # Both pages are opaque. Keeping the old page opaque and fading the new
        # page over it produces an actual per-pixel cross-fade without making
        # shared black linework darker in the middle of the blend.
        if "old" in self._layers: self._layers["old"].setOpacity(1.0)
        if "new" in self._layers: self._layers["new"].setOpacity(smooth)
        diff_opacity = math.sin(math.pi * t) * 0.9
        for kind in ("added", "removed"):
            if kind in self._layers: self._layers[kind].setOpacity(diff_opacity if self._toggles[kind] else 0.0)
        for ident, item in self._items.items():
            kind = ident.split(":", 1)[0]
            item.setVisible(self._toggles["annotations"] and self._toggles[kind])
        self._refresh_pulse_animation()

    def _set_toggle(self, name: str, enabled: bool) -> None:
        self._toggles[name] = bool(enabled); self._apply_state()

    def toggle_added(self, enabled: bool) -> None: self._set_toggle("added", enabled)
    def toggle_removed(self, enabled: bool) -> None: self._set_toggle("removed", enabled)
    def toggle_annotations(self, enabled: bool) -> None: self._set_toggle("annotations", enabled)
    def toggle_moved(self, enabled: bool) -> None: self._set_toggle("moved", enabled)

    def _on_click(self, ident: str) -> None:
        if ident in self._items: self.region_selected.emit(ident)
    def _on_double_click(self, ident: str) -> None:
        if ident in self._items: self.region_double_clicked.emit(ident)

    def fit_to_content(self) -> None:
        if self.scene.sceneRect().isValid() and not self.scene.sceneRect().isEmpty():
            if not self.isVisible() or self.view.viewport().width() <= 1 or self.view.viewport().height() <= 1:
                self._request_initial_fit()
                return
            self.view.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
    def reset_view(self) -> None:
        """Return to a 1:1 scene scale, keeping the page centered."""
        self.view.resetTransform()
        if self.scene.sceneRect().isValid():
            self.view.centerOn(self.scene.sceneRect().center())
    def fit_region(self, ident: str) -> None:
        if ident in self._items:
            self._select_region_item(ident)
            self.view.fitInView(self._items[ident], Qt.AspectRatioMode.KeepAspectRatio)
    def focus_region(self, ident: str) -> None:
        if ident in self._items:
            self._select_region_item(ident)
            self.view.centerOn(self._items[ident])

    def _select_region_item(self, ident: str) -> None:
        self.scene.clearSelection()
        self._items[ident].setSelected(True)

    def _get_pulse_strength(self) -> float:
        return self._pulse_strength

    def _set_pulse_strength(self, strength: float) -> None:
        """Apply the value supplied by ``QPropertyAnimation`` to every box."""
        self._pulse_strength = max(0.0, min(1.0, float(strength)))
        fill_alpha = round(
            ChangeBoxPulseSettings.MIN_FILL_ALPHA
            + self._pulse_strength * (ChangeBoxPulseSettings.MAX_FILL_ALPHA - ChangeBoxPulseSettings.MIN_FILL_ALPHA)
        )
        outline_alpha = round(
            ChangeBoxPulseSettings.MIN_OUTLINE_ALPHA
            + self._pulse_strength * (ChangeBoxPulseSettings.MAX_OUTLINE_ALPHA - ChangeBoxPulseSettings.MIN_OUTLINE_ALPHA)
        )
        pen_width = (
            ChangeBoxPulseSettings.MIN_PEN_WIDTH
            + self._pulse_strength * (ChangeBoxPulseSettings.MAX_PEN_WIDTH - ChangeBoxPulseSettings.MIN_PEN_WIDTH)
        )
        for ident, item in self._items.items():
            color = self._box_colors[ident]
            pen = QPen(QColor(color.red(), color.green(), color.blue(), outline_alpha), pen_width)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), fill_alpha)))

    pulse_strength = pyqtProperty(float, _get_pulse_strength, _set_pulse_strength)

    def _refresh_pulse_animation(self) -> None:
        """Animate only while at least one change box is visible."""
        if any(item.isVisible() for item in self._items.values()):
            if self._pulse_animation.state() != QAbstractAnimation.State.Running:
                self._pulse_animation.setCurrentTime(0)
                self._pulse_animation.start()
        else:
            self._pulse_animation.stop()

    def _request_initial_fit(self) -> None:
        """Fit once after the widget receives a usable viewport size.

        Results can arrive before a top-level window is shown.  Calling
        ``fitInView`` while the viewport is still 0×0 leaves a tiny drawing
        when the window subsequently appears, so defer that one initial fit.
        """
        self._fit_pending = True
        QTimer.singleShot(0, self._fit_if_pending)

    def _fit_if_pending(self) -> None:
        if not self._fit_pending:
            return
        if self.view.viewport().width() <= 1 or self.view.viewport().height() <= 1:
            return
        self._fit_pending = False
        self.fit_to_content()

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        self._fit_if_pending()

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        self._fit_if_pending()


# Convenient shorter aliases for applications.
GraphicsView = DifferenceGraphicsView
ComparisonView = ComparisonGraphicsWidget
