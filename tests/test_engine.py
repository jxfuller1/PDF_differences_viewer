from __future__ import annotations

import cv2
import pymupdf as fitz
import numpy as np

from pdf_differences_viewer.colors import DifferenceColors
from pdf_differences_viewer.engine import compare_page_images, compare_pdf_pages, render_pdf_page


def _blank(height: int = 180, width: int = 240) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def test_blank_pages_are_safe_and_have_no_differences() -> None:
    result = compare_page_images(_blank(), _blank(), tolerance_px=1)

    assert result.width == 240
    assert result.height == 180
    assert result.added_pixels == 0
    assert result.removed_pixels == 0
    assert not result.has_differences
    assert result.added_layer.shape == (180, 240, 4)
    assert result.removed_layer.shape == (180, 240, 4)


def test_new_and_removed_ink_become_colored_layers() -> None:
    old = _blank()
    new = _blank()
    cv2.rectangle(old, (30, 30), (60, 60), (0, 0, 0), thickness=-1)
    cv2.circle(new, (180, 120), 16, (0, 0, 0), thickness=-1)

    result = compare_page_images(old, new, tolerance_px=0, minimum_region_area=4)

    assert result.added_pixels > 0
    assert result.removed_pixels > 0
    assert result.added_regions
    assert result.removed_regions
    # OpenCV BGRA: additions are bright blue and removals are bright red.
    assert np.any(np.all(result.added_layer[:, :, :3] == DifferenceColors.ADDITION_BGR, axis=2))
    assert np.any(np.all(result.removed_layer[:, :, :3] == DifferenceColors.REMOVAL_BGR, axis=2))


def test_old_page_is_resized_to_new_page_dimensions() -> None:
    old = _blank(120, 160)
    new = _blank(200, 300)
    cv2.line(old, (20, 20), (130, 90), (0, 0, 0), thickness=3)
    cv2.line(new, (38, 34), (244, 150), (0, 0, 0), thickness=5)

    result = compare_page_images(old, new, tolerance_px=5)

    assert result.old_bgra.shape[:2] == new.shape[:2]
    assert result.new_bgra.shape[:2] == new.shape[:2]
    assert result.alignment.target_size == (300, 200)


def test_page_translation_is_aligned_before_difference_detection() -> None:
    old = _blank(400, 500)
    cv2.rectangle(old, (120, 100), (350, 300), (0, 0, 0), thickness=3)
    cv2.line(old, (150, 250), (320, 150), (0, 0, 0), thickness=2)
    new = _blank(400, 500)
    new[7:, 11:] = old[:-7, :-11]

    result = compare_page_images(old, new, tolerance_px=1)

    assert result.alignment.method in {"ecc-euclidean", "phase-correlation"}
    assert result.alignment.moved
    assert result.changed_pixels < 100


def test_nearby_changed_marks_are_grouped_for_review() -> None:
    old = _blank()
    new = _blank()
    cv2.circle(new, (90, 90), 4, (0, 0, 0), thickness=-1)
    cv2.circle(new, (108, 90), 4, (0, 0, 0), thickness=-1)

    result = compare_page_images(old, new, tolerance_px=0, region_merge_distance=20)

    assert len(result.added_regions) == 1
    assert result.added_regions[0].area > 0


def _write_pdf(path, *, add_circle: bool) -> None:
    document = fitz.open()
    page = document.new_page(width=240, height=180)
    page.draw_rect(fitz.Rect(30, 35, 120, 100), color=(0, 0, 0), width=1.5)
    if add_circle:
        page.draw_circle((175, 125), 16, color=(0, 0, 0), width=1.5)
    document.save(path)
    document.close()


def test_pdf_pages_render_and_compare(tmp_path) -> None:
    old_pdf = tmp_path / "old.pdf"
    new_pdf = tmp_path / "new.pdf"
    _write_pdf(old_pdf, add_circle=False)
    _write_pdf(new_pdf, add_circle=True)

    rendered = render_pdf_page(old_pdf, dpi=96)
    result = compare_pdf_pages(old_pdf, new_pdf, dpi=96, tolerance_px=1)

    assert rendered.width > 0 and rendered.height > 0
    assert rendered.bgra.shape[2] == 4
    assert result.added_pixels > 0
