from __future__ import annotations

import cv2
import numpy as np

from pdf_differences_viewer.engine import compare_page_images
from pdf_differences_viewer.graphics import ComparisonGraphicsWidget


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
