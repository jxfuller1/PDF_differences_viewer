from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import QAbstractAnimation, QPoint, QPointF, Qt
from PyQt6.QtGui import QImage, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QGraphicsView

import pdf_differences_viewer.graphics as graphics_module
from pdf_differences_viewer.colors import DifferenceColors
from pdf_differences_viewer.engine import compare_page_images
from pdf_differences_viewer.graphics import (
    ChangeBoxPulseSettings,
    ComparisonGraphicsWidget,
    bgra_to_pixmap,
)


def _result_with_region():
    old = np.full((120, 160, 3), 255, dtype=np.uint8)
    new = old.copy()
    cv2.circle(new, (80, 60), 14, (0, 0, 0), thickness=-1)
    return compare_page_images(old, new, tolerance_px=0, minimum_region_area=4)


def _reference_pixmap(array: np.ndarray) -> QPixmap:
    """Reproduce the original explicit BGR(A)-to-RGB(A) conversion."""
    order = [2, 1, 0, 3] if array.shape[2] == 4 else [2, 1, 0]
    pixels = np.ascontiguousarray(array[:, :, order])
    image_format = (
        QImage.Format.Format_RGBA8888
        if array.shape[2] == 4
        else QImage.Format.Format_RGB888
    )
    image = QImage(
        pixels.data,
        pixels.shape[1],
        pixels.shape[0],
        pixels.strides[0],
        image_format,
    ).copy()
    return QPixmap.fromImage(image)


def _rgba_pixels(pixmap: QPixmap) -> bytes:
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    return bytes(image.constBits().asstring(image.sizeInBytes()))


def test_bgra_to_pixmap_matches_original_colors_and_owns_its_pixels(qapp) -> None:
    source = np.array(
        [
            [[3, 29, 241, 255], [213, 81, 17, 180]],
            [[99, 42, 7, 96], [255, 255, 255, 0]],
        ],
        dtype=np.uint8,
    )
    expected = _reference_pixmap(source)
    actual = bgra_to_pixmap(source)

    # The returned pixmap must not retain a borrowed view of the result array.
    source.fill(0)

    assert _rgba_pixels(actual) == _rgba_pixels(expected)


def test_bgra_to_pixmap_retains_three_channel_and_noncontiguous_support(qapp) -> None:
    storage = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    source = storage[::2, 1::2]
    assert not source.flags.c_contiguous
    expected = _reference_pixmap(source)
    actual = bgra_to_pixmap(source)

    storage.fill(0)

    assert _rgba_pixels(actual) == _rgba_pixels(expected)


def test_bgra_to_pixmap_big_endian_fallback_matches_original(qapp, monkeypatch) -> None:
    source = np.arange(5 * 7 * 4, dtype=np.uint8).reshape(5, 7, 4)
    expected = _reference_pixmap(source)
    monkeypatch.setattr(graphics_module, "byteorder", "big")

    actual = bgra_to_pixmap(source)

    assert _rgba_pixels(actual) == _rgba_pixels(expected)


def test_scene_uses_native_layers_and_blend_opacity(qapp) -> None:
    viewer = ComparisonGraphicsWidget()
    viewer.resize(640, 480)
    viewer.set_result(_result_with_region())
    viewer.set_blend(50)

    assert set(viewer._layers) == {"old", "new", "added", "removed"}
    assert viewer._layers["old"].opacity() == 1.0
    assert viewer._layers["new"].opacity() == 0.5
    assert viewer._layers["added"].opacity() == 0.9
    assert viewer._layers["removed"].opacity() == 0.9


def test_layer_and_annotation_toggles_are_independent(qapp) -> None:
    viewer = ComparisonGraphicsWidget()
    viewer.set_result(_result_with_region())
    viewer.set_blend(50)
    assert viewer._items

    viewer.toggle_added(False)
    assert viewer._layers["added"].opacity() == 0.0
    assert not any(item.isVisible() for ident, item in viewer._items.items() if ident.startswith("added:"))

    viewer.toggle_added(True)
    viewer.toggle_annotations(False)
    assert viewer._layers["added"].opacity() == 0.9
    assert not any(item.isVisible() for item in viewer._items.values())

    viewer.toggle_annotations(True)
    ident = next(iter(viewer._items))
    viewer.focus_region(ident)
    viewer.fit_region(ident)
    assert viewer._items[ident].isSelected()


def test_change_boxes_pulse_in_semantic_color_and_use_hand_cursor(qapp) -> None:
    viewer = ComparisonGraphicsWidget()
    viewer.resize(640, 480)
    viewer.set_result(_result_with_region())
    viewer.show()
    qapp.processEvents()

    ident, box = next(iter(viewer._items.items()))
    assert ident.startswith("added:")
    assert viewer._pulse_animation.state() == QAbstractAnimation.State.Running
    assert viewer._pulse_animation.propertyName() == b"pulse_strength"
    assert box.acceptHoverEvents()
    assert box.cursor().shape() == Qt.CursorShape.PointingHandCursor
    assert viewer.view.itemAt(viewer.view.mapFromScene(box.rect().center())) is box

    # The first frame starts subtle. The fill remains translucent so the page
    # beneath the annotation stays readable.
    assert box.pen().color().getRgb()[:3] == DifferenceColors.ADDITION_RGB
    assert ChangeBoxPulseSettings.MIN_OUTLINE_ALPHA <= box.pen().color().alpha() <= ChangeBoxPulseSettings.MAX_OUTLINE_ALPHA
    assert ChangeBoxPulseSettings.MIN_FILL_ALPHA <= box.brush().color().alpha() <= ChangeBoxPulseSettings.MAX_FILL_ALPHA
    assert box.brush().color().alpha() < 255

    # Move the QPropertyAnimation precisely to its peak to prove it changes
    # both the outline and translucent interior.
    viewer._pulse_animation.setCurrentTime(ChangeBoxPulseSettings.PERIOD_MS // 2)
    assert box.pen().color().alpha() == ChangeBoxPulseSettings.MAX_OUTLINE_ALPHA
    assert box.brush().color().alpha() == ChangeBoxPulseSettings.MAX_FILL_ALPHA

    viewer.toggle_annotations(False)
    assert viewer._pulse_animation.state() == QAbstractAnimation.State.Stopped
    viewer.toggle_annotations(True)
    assert viewer._pulse_animation.state() == QAbstractAnimation.State.Running


def test_zoom_requires_control_and_view_never_uses_pan_hand_drag(qapp) -> None:
    viewer = ComparisonGraphicsWidget()
    viewer.resize(640, 480)
    viewer.set_result(_result_with_region())
    viewer.show()
    qapp.processEvents()

    view = viewer.view
    assert view.dragMode() == QGraphicsView.DragMode.NoDrag
    assert view.cursor().shape() == Qt.CursorShape.ArrowCursor
    original_scale = view.transform().m11()
    position = QPointF(100, 100)

    no_modifier = QWheelEvent(
        position,
        position,
        QPoint(),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    view.wheelEvent(no_modifier)
    assert view.transform().m11() == original_scale

    ctrl_held = QWheelEvent(
        position,
        position,
        QPoint(),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.ControlModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    view.wheelEvent(ctrl_held)
    assert view.transform().m11() > original_scale
