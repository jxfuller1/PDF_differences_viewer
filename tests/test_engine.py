from __future__ import annotations

import cv2
import pymupdf as fitz
import numpy as np

import pdf_differences_viewer.engine as engine
from pdf_differences_viewer.colors import DifferenceColors
from pdf_differences_viewer.engine import (
    DifferenceRegion,
    _align_old_to_new,
    _bgr_to_bgra,
    _bgr_to_gray,
    _can_use_qt_affine_warp,
    _connected_components_8,
    _dilate_binary_mask,
    _difference_mask,
    _gray_to_bgr,
    _ink_mask,
    _regions,
    _resize_bgr,
    _rgb_to_bgr,
    _should_fast_reject_ecc,
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


def test_fast_reject_requires_low_phase_confidence_and_low_ink_overlap(monkeypatch) -> None:
    old = _blank(96, 120)
    new = _blank(96, 120)
    old[12:36, 12:36] = 0
    new[58:82, 82:106] = 0

    monkeypatch.setattr(engine.cv2, "phaseCorrelate", lambda *_args: ((0.0, 0.0), 0.0))
    monkeypatch.setattr(engine.cv2, "findTransformECC", lambda *_args: (0.1, np.eye(2, 3, dtype=np.float32)))
    assert _should_fast_reject_ecc(_bgr_to_gray(old), _bgr_to_gray(new), 245)

    # Low phase confidence alone is not enough: matching ink remains eligible
    # for the unchanged full-resolution ECC path.
    assert not _should_fast_reject_ecc(_bgr_to_gray(old), _bgr_to_gray(old), 245)


def test_fast_reject_defers_when_coarse_ecc_finds_a_plausible_alignment(monkeypatch) -> None:
    old = _blank(96, 120)
    new = _blank(96, 120)
    old[12:36, 12:36] = 0
    new[58:82, 82:106] = 0
    monkeypatch.setattr(engine.cv2, "phaseCorrelate", lambda *_args: ((0.0, 0.0), 0.0))
    monkeypatch.setattr(engine.cv2, "findTransformECC", lambda *_args: (0.95, np.eye(2, 3, dtype=np.float32)))

    assert not _should_fast_reject_ecc(_bgr_to_gray(old), _bgr_to_gray(new), 245)


def test_fast_reject_defers_when_coarse_ecc_is_inconclusive(monkeypatch) -> None:
    old = _blank(96, 120)
    new = _blank(96, 120)
    old[12:36, 12:36] = 0
    new[58:82, 82:106] = 0
    monkeypatch.setattr(engine.cv2, "phaseCorrelate", lambda *_args: ((0.0, 0.0), 0.0))

    def failed_ecc(*_args):
        raise cv2.error("coarse alignment was inconclusive")

    monkeypatch.setattr(engine.cv2, "findTransformECC", failed_ecc)
    assert not _should_fast_reject_ecc(_bgr_to_gray(old), _bgr_to_gray(new), 245)


def test_fast_reject_skips_ecc_and_preserves_resize_only_regions(monkeypatch) -> None:
    old = _blank(480, 600)
    new = _blank(480, 600)
    old[60:180, 60:180] = 0
    new[290:410, 410:530] = 0
    monkeypatch.setattr(engine.cv2, "phaseCorrelate", lambda *_args: ((0.0, 0.0), 0.0))
    ecc_shapes: list[tuple[int, int]] = []

    def low_score_coarse_ecc(template, *_args):
        ecc_shapes.append(template.shape)
        return 0.1, np.eye(2, 3, dtype=np.float32)

    monkeypatch.setattr(engine.cv2, "findTransformECC", low_score_coarse_ecc)
    aligned, metadata = _align_old_to_new(old, new)
    np.testing.assert_array_equal(aligned, _resize_bgr(old, new.shape[1], new.shape[0]))
    assert metadata.method == "resize"
    assert metadata.message.startswith("fast-reject:")
    assert ecc_shapes == [(384, 480)]

    result = compare_page_images(old, new, tolerance_px=0)
    old_ink, new_ink = _ink_mask(old, 245), _ink_mask(new, 245)
    expected_added = _difference_mask(new_ink, old_ink, 0)
    expected_removed = _difference_mask(old_ink, new_ink, 0)
    np.testing.assert_array_equal(result.added_mask, expected_added)
    np.testing.assert_array_equal(result.removed_mask, expected_removed)
    assert result.added_regions == _regions(expected_added, "added", 4, 20)
    assert result.removed_regions == _regions(expected_removed, "removed", 4, 20)
    assert ecc_shapes == [(384, 480), (384, 480)]


def test_fast_reject_defers_to_full_ecc_when_phase_is_plausible(monkeypatch) -> None:
    old = _blank(96, 120)
    old[12:36, 12:36] = 0
    calls = 0
    monkeypatch.setattr(engine.cv2, "phaseCorrelate", lambda *_args: ((0.0, 0.0), 1.0))

    def successful_ecc(*_args):
        nonlocal calls
        calls += 1
        return 0.95, np.eye(2, 3, dtype=np.float32)

    monkeypatch.setattr(engine.cv2, "findTransformECC", successful_ecc)
    _aligned, metadata = _align_old_to_new(old, old)
    assert calls == 1
    assert metadata.method == "ecc-euclidean"


def test_nearby_changed_marks_are_grouped_for_review() -> None:
    old = _blank()
    new = _blank()
    cv2.circle(new, (90, 90), 4, (0, 0, 0), thickness=-1)
    cv2.circle(new, (108, 90), 4, (0, 0, 0), thickness=-1)

    result = compare_page_images(old, new, tolerance_px=0, region_merge_distance=20)

    assert len(result.added_regions) == 1
    assert result.added_regions[0].area > 0


def _legacy_regions(mask: np.ndarray, kind: str, minimum_area: int, merge_distance: int) -> list[DifferenceRegion]:
    """The pre-vectorization implementation retained as a test oracle."""
    grouped = _dilate_binary_mask(mask, merge_distance) if merge_distance else mask
    count, labels = cv2.connectedComponents(grouped, connectivity=8)
    regions: list[DifferenceRegion] = []
    for label in range(1, count):
        in_group = (labels == label) & (mask > 0)
        area = int(np.count_nonzero(in_group))
        if area < minimum_area:
            continue
        ys, xs = np.nonzero(in_group)
        regions.append(
            DifferenceRegion(
                (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)),
                area,
                kind,
            )
        )
    return regions


def _canonical_component_labels(labels: np.ndarray) -> np.ndarray:
    """Normalize label numbers by first row-major occurrence for comparison."""
    foreground_labels = labels[labels > 0]
    if not foreground_labels.size:
        return labels
    unique_labels, first_indices = np.unique(foreground_labels, return_index=True)
    ordered_labels = unique_labels[np.argsort(first_indices)]
    mapping = np.zeros(int(unique_labels.max()) + 1, dtype=np.int32)
    mapping[ordered_labels] = np.arange(1, ordered_labels.size + 1, dtype=np.int32)
    return mapping[labels]


def _region_order(region: DifferenceRegion) -> tuple[int, int, int, int, int]:
    x, y, width, height = region.bbox
    return y, x, height, width, region.area


def test_numpy_connected_components_matches_opencv_8_connectivity() -> None:
    masks = [
        np.zeros((5, 7), dtype=np.uint8),
        np.array(
            [
                [255, 0, 0, 0, 255],
                [0, 255, 0, 255, 0],
                [0, 0, 255, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        ),
        np.full((9, 11), 255, dtype=np.uint8),
    ]
    rng = np.random.default_rng(23)
    masks.extend(
        np.where(rng.random((height, width)) < probability, 17, 0).astype(np.uint8)
        for height, width, probability in ((3, 4, 0.4), (19, 23, 0.1), (29, 31, 0.7))
    )

    for mask in masks:
        expected_count, expected_labels = cv2.connectedComponents(mask, connectivity=8)
        actual_count, actual_labels = _connected_components_8(mask)
        assert actual_count == expected_count
        np.testing.assert_array_equal(
            _canonical_component_labels(actual_labels),
            _canonical_component_labels(expected_labels),
        )


def test_regions_vectorized_aggregation_preserves_legacy_results() -> None:
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[2:4, 3:6] = 255       # area 6, first in component-label order
    mask[3:5, 9:11] = 255      # area 4, merges with the first at distance 5
    mask[12:15, 20:22] = 255   # area 6, remains separate
    mask[17, 1] = 255          # filtered when the minimum area is 4

    for kind in ("added", "removed"):
        for minimum_area in (1, 4, 7):
            for merge_distance in (0, 2, 5):
                expected = _legacy_regions(mask, kind, minimum_area, merge_distance)
                assert _regions(mask, kind, minimum_area, merge_distance) == sorted(
                    expected,
                    key=_region_order,
                )


def test_regions_without_opencv_components_match_randomized_legacy_results() -> None:
    rng = np.random.default_rng(29)
    for height, width, probability in ((19, 23, 0.06), (31, 37, 0.2), (43, 47, 0.6)):
        mask = np.where(rng.random((height, width)) < probability, 255, 0).astype(np.uint8)
        for minimum_area in (1, 4, 7):
            for merge_distance in (0, 2, 5):
                expected = _legacy_regions(mask, "added", minimum_area, merge_distance)
                actual = _regions(mask, "added", minimum_area, merge_distance)
                # OpenCV's default Spaghetti implementation does not promise
                # row-major label IDs; component membership and region geometry
                # are the observable behavior being preserved here.
                assert actual == sorted(expected, key=_region_order)


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


def test_sparse_binary_mask_dilation_matches_opencv_at_page_edges() -> None:
    mask = np.zeros((23, 31), dtype=np.uint8)
    mask[0, 0] = mask[-1, -1] = 255

    for kernel_size in (2, 3, 20):
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


