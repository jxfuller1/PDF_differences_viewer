from __future__ import annotations

import time

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QGraphicsView

from pdf_differences_viewer.colors import DifferenceColors
from pdf_differences_viewer.engine import compare_page_images
from pdf_differences_viewer.graphics import (
    ChangeBoxPulseSettings,
    ComparisonGraphicsWidget,
)


def _result_with_region():
    old = np.full((120, 160, 3), 255, dtype=np.uint8)
    new = old.copy()
    cv2.circle(new, (80, 60), 14, (0, 0, 0), thickness=-1)
    return compare_page_images(old, new, tolerance_px=0, minimum_region_area=4)


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
    assert viewer._pulse_timer.isActive()
    assert box.acceptHoverEvents()
    assert box.cursor().shape() == Qt.CursorShape.PointingHandCursor
    assert viewer.view.itemAt(viewer.view.mapFromScene(box.rect().center())) is box

    # The first frame starts subtle. The fill remains translucent so the page
    # beneath the annotation stays readable.
    assert box.pen().color().getRgb()[:3] == DifferenceColors.ADDITION_RGB
    assert ChangeBoxPulseSettings.MIN_OUTLINE_ALPHA <= box.pen().color().alpha() <= ChangeBoxPulseSettings.MAX_OUTLINE_ALPHA
    assert ChangeBoxPulseSettings.MIN_FILL_ALPHA <= box.brush().color().alpha() <= ChangeBoxPulseSettings.MAX_FILL_ALPHA
    assert box.brush().color().alpha() < 255

    # Move precisely to the pulse peak to prove the tunable animation changes
    # both the outline and translucent interior.
    viewer._pulse_started_at = time.monotonic() - ChangeBoxPulseSettings.PERIOD_MS / 2_000
    viewer._update_box_pulse()
    assert box.pen().color().alpha() == ChangeBoxPulseSettings.MAX_OUTLINE_ALPHA
    assert box.brush().color().alpha() == ChangeBoxPulseSettings.MAX_FILL_ALPHA


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
