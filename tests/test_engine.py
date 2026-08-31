from __future__ import annotations

import cv2
import pymupdf as fitz
import numpy as np

from pdf_differences_viewer.colors import DifferenceColors
from pdf_differences_viewer.engine import (
    _bgr_to_bgra,
    _bgr_to_gray,
    _can_use_qt_affine_warp,
    _dilate_binary_mask,
    _gray_to_bgr,
    _ink_mask,
    _resize_bgr,
    _rgb_to_bgr,
    _warp_bgr_affine,
    _warp_bgr_affine_numpy,
    compare_page_images,
    compare_pdf_pages,
    render_pdf_page,
)


def _blank(height: int = 180, width: int = 240) -> np.ndarray:
    return np.full((height, width, 3), 255, dtype=np.uint8)


def test_numpy_channel_conversions_preserve_expected_color_layout() -> None:
    rgb = np.array([[[1, 2, 3], [10, 20, 30]]], dtype=np.uint8)
    bgr = _rgb_to_bgr(rgb)

    np.testing.assert_array_equal(bgr, [[[3, 2, 1], [30, 20, 10]]])
    assert bgr.flags.c_contiguous
    np.testing.assert_array_equal(
        _gray_to_bgr(np.array([[0, 128]], dtype=np.uint8)),
        [[[0, 0, 0], [128, 128, 128]]],
    )
    np.testing.assert_array_equal(
        _bgr_to_gray(np.array([[[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)),
        [[0, 29, 150, 76]],
    )
    bgra = _bgr_to_bgra(bgr)
    np.testing.assert_array_equal(bgra[:, :, :3], bgr)
    assert np.all(bgra[:, :, 3] == 255)


def test_pillow_resize_and_numpy_affine_warp_preserve_bgr_geometry() -> None:
    source = _blank(5, 5)
    source[2, 3] = (0, 0, 0)

    resized = _resize_bgr(source, 10, 8)
    assert resized.shape == (8, 10, 3)
    assert resized.flags.c_contiguous
    identity = _resize_bgr(source, 5, 5)
    np.testing.assert_array_equal(identity, source)
    assert identity is not source

    # The affine matrix is destination-to-source, matching the former
    # cv2.WARP_INVERSE_MAP call: source x=3 appears at destination x=2.
    matrix = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
    warped = _warp_bgr_affine(source, matrix, 5, 5)
    np.testing.assert_array_equal(warped[2, 2], (0, 0, 0))
    assert np.count_nonzero(np.all(warped == 0, axis=2)) == 1


def test_fast_qt_affine_path_preserves_the_numpy_ink_mask() -> None:
    source = _blank(64, 64)
    source[12:45, 20] = (0, 0, 0)
    source[28, 24:48] = (244, 244, 244)
    matrix = np.array([[1, 0, 0.003], [0, 1, -0.002]], dtype=np.float32)

    assert _can_use_qt_affine_warp(matrix, ink_threshold=245)
    accelerated = _warp_bgr_affine(source, matrix, 64, 64, ink_threshold=245)
    precise = _warp_bgr_affine_numpy(source, matrix, 64, 64)

    np.testing.assert_array_equal(_ink_mask(accelerated, 245), _ink_mask(precise, 245))


def test_fast_qt_affine_path_accepts_read_only_source_images() -> None:
    source = _blank(64, 64)
    source[12:45, 20] = (0, 0, 0)
    source.setflags(write=False)
    matrix = np.array([[1, 0, 0.003], [0, 1, -0.002]], dtype=np.float32)

    accelerated = _warp_bgr_affine(source, matrix, 64, 64, ink_threshold=245)
    precise = _warp_bgr_affine_numpy(source, matrix, 64, 64)

    np.testing.assert_array_equal(_ink_mask(accelerated, 245), _ink_mask(precise, 245))


def test_rotated_affine_transform_uses_exact_numpy_fallback() -> None:
    source = _blank(64, 64)
    source[12:45, 20] = (0, 0, 0)
    radians = np.deg2rad(0.5)
    matrix = np.array(
        [[np.cos(radians), np.sin(radians), 0], [-np.sin(radians), np.cos(radians), 0]],
        dtype=np.float32,
    )

    assert not _can_use_qt_affine_warp(matrix, ink_threshold=245)
    np.testing.assert_array_equal(
        _warp_bgr_affine(source, matrix, 64, 64, ink_threshold=245),
        _warp_bgr_affine_numpy(source, matrix, 64, 64),
    )


def test_qt_affine_path_rejects_scale_and_shear() -> None:
    assert not _can_use_qt_affine_warp(
        np.array([[2, 0, 0], [0, 0.5, 0]], dtype=np.float32),
        ink_threshold=245,
    )
    assert not _can_use_qt_affine_warp(
        np.array([[1, 0.5, 0], [0, 1, 0]], dtype=np.float32),
        ink_threshold=245,
    )


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


def test_binary_mask_dilation_matches_opencv_rectangular_kernels() -> None:
    rng = np.random.default_rng(7)
    mask = np.where(rng.random((23, 31)) > 0.88, 255, 0).astype(np.uint8)
    mask[0, 0] = mask[-1, -1] = 255

    for kernel_size in (1, 2, 3, 4, 20):
        expected = cv2.dilate(
            mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size)),
            iterations=1,
        )
        np.testing.assert_array_equal(_dilate_binary_mask(mask, kernel_size), expected)


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
