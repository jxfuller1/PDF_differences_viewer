from __future__ import annotations

import cv2
import pymupdf as fitz
import numpy as np
import pytest

import pdf_differences_viewer.engine as engine
from pdf_differences_viewer.colors import DifferenceColors
from pdf_differences_viewer.engine import (
    DifferenceRegion,
    EccConvergenceError,
    EccSettings,
    _EccIterationWorkspace,
    _align_old_to_new,
    _bgr_to_bgra,
    _bgr_to_gray,
    _can_use_qt_affine_warp,
    _connected_components_8,
    _distance_transform_l2_mask3,
    _dilate_binary_mask,
    _difference_mask,
    _find_transform_ecc_euclidean,
    _gaussian_blur_5x5,
    _gray_to_bgr,
    _ink_mask,
    _phase_correlate,
    _regions,
    _resize_bgr,
    _rgb_to_bgr,
    _should_fast_reject_ecc,
    _warp_bgr_affine,
    _warp_bgr_affine_numpy,
    _warp_binary_translation_nearest,
    _warp_ecc_channels,
    _warp_ecc_channels_numpy,
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


def _ink_mask_oracle(image: np.ndarray, threshold: int) -> np.ndarray:
    return np.where(_bgr_to_gray(image) < threshold, 255, 0).astype(np.uint8)


@pytest.mark.parametrize("threshold", [0, 1, 245, 255, 256])
def test_ink_mask_matches_oracle_for_monochrome_pages(threshold: int) -> None:
    image = np.full((19, 23, 3), 255, dtype=np.uint8)
    image[2:8, 3:11] = 0
    image[12, 17] = 245

    actual = _ink_mask(image, threshold)

    np.testing.assert_array_equal(actual, _ink_mask_oracle(image, threshold))
    assert actual.dtype == np.uint8
    assert actual.flags.c_contiguous


def test_ink_mask_matches_oracle_for_sparse_off_grid_color() -> None:
    image = np.full((37, 41, 3), 255, dtype=np.uint8)
    # Keep colored pixels off the accelerated sampler's grid.
    image[1, 2] = (0, 64, 255)
    image[14, 27] = (240, 0, 240)
    image[36, 40] = (10, 20, 30)

    np.testing.assert_array_equal(_ink_mask(image, 245), _ink_mask_oracle(image, 245))


@pytest.mark.parametrize("threshold", [1, 29, 76, 128, 245, 255])
def test_ink_mask_matches_oracle_for_dense_color_and_ignores_bgra_alpha(
    threshold: int,
) -> None:
    rng = np.random.default_rng(8128)
    bgr = rng.integers(0, 256, size=(127, 131, 3), dtype=np.uint8)
    alpha = rng.integers(0, 256, size=(127, 131, 1), dtype=np.uint8)
    bgra = np.concatenate((bgr, alpha), axis=2)

    actual = _ink_mask(bgra, threshold)

    np.testing.assert_array_equal(actual, _ink_mask_oracle(bgra, threshold))
    assert actual.flags.c_contiguous


def test_ink_mask_accepts_non_contiguous_read_only_input() -> None:
    base = np.full((28, 34, 3), 255, dtype=np.uint8)
    base[::2, 1::3] = (0, 0, 0)
    image = base[1:, 2:, :]
    image.setflags(write=False)

    actual = _ink_mask(image, 245)

    np.testing.assert_array_equal(actual, _ink_mask_oracle(image, 245))
    assert actual.flags.c_contiguous


def test_ink_mask_non_uint8_uses_exact_gray_fallback() -> None:
    image = np.array([[[0.0, 0.0, 0.0], [255.0, 255.0, 255.0]]], dtype=np.float32)

    actual = _ink_mask(image, 245)

    np.testing.assert_array_equal(actual, _ink_mask_oracle(image, 245))
    assert actual.dtype == np.uint8
    assert actual.flags.c_contiguous


def test_ink_mask_fractional_threshold_preserves_previous_semantics() -> None:
    values = np.array([0, 29, 30, 244, 245, 255], dtype=np.uint8)
    image = np.repeat(values[np.newaxis, :, np.newaxis], 3, axis=2)

    for threshold in (0.5, 29.5, 245.5):
        np.testing.assert_array_equal(
            _ink_mask(image, threshold),
            _ink_mask_oracle(image, threshold),
        )


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


def test_numpy_binary_translation_matches_opencv_nearest_inverse_warp() -> None:
    rng = np.random.default_rng(47)
    source = np.where(rng.random((17, 23)) < 0.22, 1, 0).astype(np.uint8)
    shifts = (
        (-4.75, 3.125),
        (-1.5006, -0.5006),
        (-1.5, -0.5),
        (-1.4996, -0.4996),
        (0.0, 0.0),
        (0.4996, 1.4996),
        (0.5, 1.5),
        (0.5006, 1.5006),
        (8.25, -6.75),
    )

    for target_width, target_height in ((23, 17), (29, 21), (11, 9)):
        for shift in shifts:
            matrix = np.float32([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]])
            expected = cv2.warpAffine(
                source,
                matrix,
                (target_width, target_height),
                flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            actual = _warp_binary_translation_nearest(
                source,
                shift,
                target_width,
                target_height,
            )
            np.testing.assert_array_equal(actual, expected)


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


def test_numpy_distance_transform_matches_opencv_l2_mask3() -> None:
    masks = [
        np.zeros((5, 7), dtype=np.uint8),
        np.full((5, 7), 255, dtype=np.uint8),
        np.array(
            [
                [0, 255, 255, 255, 255],
                [255, 255, 255, 255, 255],
                [255, 255, 255, 255, 255],
                [255, 255, 255, 255, 255],
            ],
            dtype=np.uint8,
        ),
    ]
    rng = np.random.default_rng(19)
    masks.extend(
        np.where(rng.random((height, width)) < probability, 255, 0).astype(np.uint8)
        for height, width, probability in ((3, 4, 0.2), (19, 23, 0.5), (29, 31, 0.9))
    )

    for mask in masks:
        expected = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        actual = _distance_transform_l2_mask3(mask)
        # OpenCV's IPP and portable paths use different internal arithmetic,
        # so their raw float values can differ by a few ULPs. The tolerance
        # UI uses whole pixels; this keeps the compatible 3x3 weights close
        # while the next test proves all selectable decisions are identical.
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)


def test_numpy_distance_transform_preserves_tolerance_masks() -> None:
    rng = np.random.default_rng(31)
    source = np.where(rng.random((29, 31)) < 0.22, 255, 0).astype(np.uint8)
    reference = np.where(rng.random((29, 31)) < 0.18, 255, 0).astype(np.uint8)
    for tolerance in range(21):
        distance = cv2.distanceTransform(cv2.bitwise_not(reference), cv2.DIST_L2, 3)
        expected = np.where((source > 0) & (distance > tolerance), 255, 0).astype(np.uint8)
        np.testing.assert_array_equal(_difference_mask(source, reference, tolerance), expected)


def test_tolerance_mask_acceleration_matches_opencv_for_all_ui_tolerances() -> None:
    rng = np.random.default_rng(131)
    source = (rng.random((47, 61)) < 0.15).astype(np.uint8) * 7
    reference = (rng.random((47, 61)) < 0.12).astype(np.uint8) * 3
    binary_reference = np.where(reference > 0, 0, 255).astype(np.uint8)
    for tolerance in range(21):
        expected = np.where(
            (source > 0)
            & (cv2.distanceTransform(binary_reference, cv2.DIST_L2, 3) > tolerance),
            255,
            0,
        ).astype(np.uint8)
        np.testing.assert_array_equal(_difference_mask(source, reference, tolerance), expected)


def test_tolerance_mask_preserves_positive_source_ink_predicate() -> None:
    source = np.array([[-1.0, np.nan, 0.0, 1.0]], dtype=np.float32)
    reference = np.zeros_like(source)
    np.testing.assert_array_equal(
        _difference_mask(source, reference, 0),
        np.array([[0, 0, 0, 255]], dtype=np.uint8),
    )


def test_tolerance_mask_dispatch_exercises_direct_span_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.zeros((80, 100), dtype=np.uint8)
    reference = np.zeros_like(source)
    source[10, 10] = 255
    source[40, 50] = 255
    reference[10, 11] = 255

    direct = engine._difference_mask_local(source, reference, 2)
    assert direct is not None
    direct_expected = np.where(
        (source > 0)
        & (cv2.distanceTransform(cv2.bitwise_not(reference), cv2.DIST_L2, 3) > 2),
        255,
        0,
    ).astype(np.uint8)
    np.testing.assert_array_equal(direct, direct_expected)

    rng = np.random.default_rng(197)
    source = np.where(rng.random((37, 43)) < 0.18, 255, 0).astype(np.uint8)
    reference = np.where(rng.random((37, 43)) < 0.14, 255, 0).astype(np.uint8)
    expected = np.where(
        (source > 0)
        & (cv2.distanceTransform(cv2.bitwise_not(reference), cv2.DIST_L2, 3) > 10),
        255,
        0,
    ).astype(np.uint8)

    monkeypatch.setattr(engine.DifferenceMaskSettings, "DIRECT_MAX_WORK_FRACTION", 0.0)
    monkeypatch.setattr(engine.DifferenceMaskSettings, "DIRECT_SMALL_OFFSET_MAX_WORK_FRACTION", 0.0)
    monkeypatch.setattr(engine.DifferenceMaskSettings, "SPAN_MAX_WORK_FRACTION", 100.0)
    span = engine._difference_mask_local(source, reference, 10)
    assert span is not None
    np.testing.assert_array_equal(span, expected)

    monkeypatch.setattr(engine.DifferenceMaskSettings, "SPAN_MAX_WORK_FRACTION", 0.0)
    assert engine._difference_mask_local(source, reference, 10) is None
    np.testing.assert_array_equal(_difference_mask(source, reference, 10), expected)


def test_numpy_phase_correlate_matches_opencv_shift_and_response() -> None:
    rng = np.random.default_rng(83)
    noisy = rng.normal(size=(99, 121)).astype(np.float32)
    structured = np.full((400, 500), 255, dtype=np.float32)
    structured[65:180, 90:330] = 0
    structured[230:315, 170:205] = 72
    translated = np.full_like(structured, 255)
    translated[7:, 11:] = structured[:-7, :-11]
    cases = [
        (noisy, np.roll(noisy, shift=(7, -11), axis=(0, 1))),
        (structured, translated),
        (np.zeros((99, 121), dtype=np.float32), np.zeros((99, 121), dtype=np.float32)),
    ]

    for first, second in cases:
        expected_shift, expected_response = cv2.phaseCorrelate(first, second)
        actual_shift, actual_response = _phase_correlate(first, second)
        np.testing.assert_allclose(actual_shift, expected_shift, rtol=0, atol=5e-3)
        np.testing.assert_allclose(actual_response, expected_response, rtol=0, atol=2e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_parallel_phase_spectra_preserve_sequential_results(
    monkeypatch,
    dtype,
) -> None:
    rng = np.random.default_rng(809)
    first = rng.normal(size=(73, 91)).astype(dtype)
    second = np.roll(first, shift=(5, -9), axis=(0, 1))

    monkeypatch.setattr(
        engine.PhaseCorrelationSettings,
        "PARALLEL_FORWARD_MIN_PIXELS",
        first.size * 10,
    )
    sequential = _phase_correlate(first, second)
    monkeypatch.setattr(engine, "cpu_count", lambda: 2)
    monkeypatch.setattr(
        engine.PhaseCorrelationSettings,
        "PARALLEL_FORWARD_MIN_PIXELS",
        0,
    )
    parallel = _phase_correlate(first, second)

    assert parallel == sequential
    expected_shift, expected_response = cv2.phaseCorrelate(first, second)
    np.testing.assert_allclose(parallel[0], expected_shift, rtol=0, atol=5e-3)
    np.testing.assert_allclose(parallel[1], expected_response, rtol=0, atol=2e-5)


def test_phase_spectra_fall_back_when_threads_are_unavailable(monkeypatch) -> None:
    rng = np.random.default_rng(821)
    first = rng.normal(size=(67, 89)).astype(np.float32)
    second = np.roll(first, shift=(-4, 7), axis=(0, 1))
    monkeypatch.setattr(
        engine.PhaseCorrelationSettings,
        "PARALLEL_FORWARD_MIN_PIXELS",
        first.size * 10,
    )
    expected = _phase_correlate(first, second)

    class UnavailableExecutor:
        def __init__(self, **_kwargs) -> None:
            raise RuntimeError("worker threads unavailable")

    monkeypatch.setattr(engine, "ThreadPoolExecutor", UnavailableExecutor)
    monkeypatch.setattr(engine, "cpu_count", lambda: 2)
    monkeypatch.setattr(
        engine.PhaseCorrelationSettings,
        "PARALLEL_FORWARD_MIN_PIXELS",
        0,
    )

    assert _phase_correlate(first, second) == expected


def test_numpy_ecc_preprocessing_matches_opencv_gaussian() -> None:
    rng = np.random.default_rng(101)
    image = rng.random((53, 71), dtype=np.float32)

    expected = cv2.GaussianBlur(image, (5, 5), 0)
    actual = _gaussian_blur_5x5(image)

    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-7)


def test_ecc_iteration_workspace_preserves_direct_array_math() -> None:
    rng = np.random.default_rng(307)
    height, width = 31, 43
    template = rng.random((height, width), dtype=np.float32)
    warped = rng.normal(size=(height, width, 3)).astype(np.float32)
    valid = rng.random((height, width)) > 0.17
    valid_y, valid_x = np.nonzero(valid)

    template_values = template[valid_y, valid_x]
    input_values = warped[valid_y, valid_x, 0]
    expected_template = template_values - np.mean(
        template_values,
        dtype=np.float64,
    )
    expected_input = input_values - np.mean(input_values, dtype=np.float64)
    expected_template_norm = float(
        np.sum(expected_template * expected_template, dtype=np.float64)
    )
    expected_input_norm = float(
        np.sum(expected_input * expected_input, dtype=np.float64)
    )
    expected_correlation = float(
        np.sum(expected_template * expected_input, dtype=np.float64)
    )

    workspace = _EccIterationWorkspace(template.shape)
    (
        actual_template,
        actual_input,
        actual_template_norm,
        actual_input_norm,
        actual_correlation,
    ) = workspace.prepare_values(template, warped, valid)

    np.testing.assert_array_equal(actual_template, expected_template)
    np.testing.assert_array_equal(actual_input, expected_input)
    assert actual_template_norm == expected_template_norm
    assert actual_input_norm == expected_input_norm
    assert actual_correlation == expected_correlation

    angle = np.float32(0.037)
    sine = np.float32(np.sin(angle))
    cosine = np.float32(np.cos(angle))
    gradient_x = warped[valid_y, valid_x, 1]
    gradient_y = warped[valid_y, valid_x, 2]
    expected_jacobian = np.empty((valid_x.size, 3), dtype=np.float32)
    x_coordinates = valid_x.astype(np.float32, copy=False)
    y_coordinates = valid_y.astype(np.float32, copy=False)
    expected_jacobian[:, 0] = gradient_x * (
        -sine * x_coordinates - cosine * y_coordinates
    ) + gradient_y * (
        cosine * x_coordinates - sine * y_coordinates
    )
    expected_jacobian[:, 1] = gradient_x
    expected_jacobian[:, 2] = gradient_y

    jacobian, projection_jacobian = workspace.build_jacobians(
        warped,
        sine,
        cosine,
    )

    np.testing.assert_array_equal(jacobian.T, expected_jacobian)
    np.testing.assert_array_equal(
        projection_jacobian,
        expected_jacobian.T.astype(np.float64),
    )


def test_pillow_ecc_warp_preserves_numpy_validity_and_border_sampling() -> None:
    rng = np.random.default_rng(307)
    intensity = rng.random((768, 1024), dtype=np.float32)
    intensity = _gaussian_blur_5x5(intensity)
    gradient_x, gradient_y = engine._central_gradients(intensity)
    source = np.stack((intensity, gradient_x, gradient_y), axis=2)
    padded = np.pad(source, ((1, 1), (1, 1), (0, 0)), mode="constant")
    angle = np.deg2rad(0.7)
    matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), -0.35],
            [np.sin(angle), np.cos(angle), 0.4],
        ],
        dtype=np.float32,
    )
    destination_x = np.arange(source.shape[1], dtype=np.float32)[np.newaxis, :]
    destination_y = np.arange(source.shape[0], dtype=np.float32)[:, np.newaxis]

    expected_warped, expected_valid = _warp_ecc_channels_numpy(
        padded,
        source.shape[:2],
        matrix,
        destination_x,
        destination_y,
    )
    actual_warped, actual_valid = _warp_ecc_channels(
        padded,
        source.shape[:2],
        matrix,
        destination_x,
        destination_y,
    )

    np.testing.assert_array_equal(actual_valid, expected_valid)
    np.testing.assert_allclose(
        actual_warped[actual_valid],
        expected_warped[expected_valid],
        rtol=0,
        atol=2e-4,
    )


def test_pillow_ecc_warp_falls_back_to_numpy(monkeypatch) -> None:
    source = np.arange(7 * 9 * 3, dtype=np.float32).reshape(7, 9, 3)
    padded = np.pad(source, ((1, 1), (1, 1), (0, 0)), mode="constant")
    matrix = np.array([[1, 0, 0.25], [0, 1, -0.4]], dtype=np.float32)
    destination_x = np.arange(source.shape[1], dtype=np.float32)[np.newaxis, :]
    destination_y = np.arange(source.shape[0], dtype=np.float32)[:, np.newaxis]
    expected = _warp_ecc_channels_numpy(
        padded,
        source.shape[:2],
        matrix,
        destination_x,
        destination_y,
    )

    transform_calls = 0

    def unavailable(*_args, **_kwargs):
        nonlocal transform_calls
        transform_calls += 1
        raise OSError("compiled affine sampler unavailable")

    monkeypatch.setattr(engine.Image.Image, "transform", unavailable)
    with engine._EccChannelWarper(
        padded,
        source.shape[:2],
        destination_x,
        destination_y,
    ) as warper:
        actual = warper.warp(matrix)
        calls_after_failure = transform_calls
        second = warper.warp(matrix)

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(second[0], expected[0])
    np.testing.assert_array_equal(second[1], expected[1])
    assert calls_after_failure > 0
    assert transform_calls == calls_after_failure


def test_numpy_ecc_matches_opencv_euclidean_alignment() -> None:
    height, width = 240, 320
    source = np.ones((height, width), dtype=np.float32)
    source[35:145, 50:230] = 0.12
    source[168:205, 180:280] = 0.45
    angle = np.deg2rad(0.8)
    expected_matrix = np.array(
        [
            [np.cos(angle), -np.sin(angle), 7.25],
            [np.sin(angle), np.cos(angle), -5.5],
        ],
        dtype=np.float32,
    )
    template = cv2.warpAffine(
        source,
        expected_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    initial = np.array([[1, 0, 7.25], [0, 1, -5.5]], dtype=np.float32)
    expected_score, expected = cv2.findTransformECC(
        template,
        source,
        initial.copy(),
        cv2.MOTION_EUCLIDEAN,
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-6),
    )

    actual_score, actual = _find_transform_ecc_euclidean(
        template,
        source,
        initial,
        60,
        1e-6,
    )

    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-4)
    np.testing.assert_allclose(actual_score, expected_score, rtol=0, atol=2e-6)


def test_numpy_ecc_reports_uniform_images_as_inconclusive() -> None:
    uniform = np.ones((64, 80), dtype=np.float32)

    with pytest.raises(EccConvergenceError, match="uniform"):
        _find_transform_ecc_euclidean(
            uniform,
            uniform,
            np.eye(2, 3, dtype=np.float32),
            20,
            1e-4,
        )


def test_numpy_ecc_pyramid_keeps_full_resolution_matrix_coordinates(monkeypatch) -> None:
    monkeypatch.setattr(EccSettings, "MAX_WORKING_SHORT_SIDE_PX", 120)
    monkeypatch.setattr(EccSettings, "REFINEMENT_MIN_SOURCE_SHORT_SIDE_PX", 1)
    monkeypatch.setattr(EccSettings, "REFINEMENT_SHORT_SIDE_PX", 180)
    height, width = 240, 320
    source = np.ones((height, width), dtype=np.float32)
    source[35:145, 50:230] = 0.12
    source[168:205, 180:280] = 0.45
    angle = np.deg2rad(0.8)
    expected = np.array(
        [
            [np.cos(angle), -np.sin(angle), 7.25],
            [np.sin(angle), np.cos(angle), -5.5],
        ],
        dtype=np.float32,
    )
    template = cv2.warpAffine(
        source,
        expected,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    score, actual = _find_transform_ecc_euclidean(
        template,
        source,
        np.array([[1, 0, 7.25], [0, 1, -5.5]], dtype=np.float32),
        60,
        1e-6,
    )

    assert score > 0.95
    np.testing.assert_allclose(actual[:, :2], expected[:, :2], rtol=0, atol=2e-3)
    np.testing.assert_allclose(actual[:, 2], expected[:, 2], rtol=0, atol=0.11)


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

    monkeypatch.setattr(engine, "_phase_correlate", lambda *_args: ((0.0, 0.0), 0.0))
    monkeypatch.setattr(engine, "_find_transform_ecc_euclidean", lambda *_args: (0.1, np.eye(2, 3, dtype=np.float32)))
    assert _should_fast_reject_ecc(_bgr_to_gray(old), _bgr_to_gray(new), 245)

    # Low phase confidence alone is not enough: matching ink remains eligible
    # for the unchanged full-resolution ECC path.
    assert not _should_fast_reject_ecc(_bgr_to_gray(old), _bgr_to_gray(old), 245)


def test_fast_reject_defers_when_coarse_ecc_finds_a_plausible_alignment(monkeypatch) -> None:
    old = _blank(96, 120)
    new = _blank(96, 120)
    old[12:36, 12:36] = 0
    new[58:82, 82:106] = 0
    monkeypatch.setattr(engine, "_phase_correlate", lambda *_args: ((0.0, 0.0), 0.0))
    monkeypatch.setattr(engine, "_find_transform_ecc_euclidean", lambda *_args: (0.95, np.eye(2, 3, dtype=np.float32)))

    assert not _should_fast_reject_ecc(_bgr_to_gray(old), _bgr_to_gray(new), 245)


def test_fast_reject_defers_when_coarse_ecc_is_inconclusive(monkeypatch) -> None:
    old = _blank(96, 120)
    new = _blank(96, 120)
    old[12:36, 12:36] = 0
    new[58:82, 82:106] = 0
    monkeypatch.setattr(engine, "_phase_correlate", lambda *_args: ((0.0, 0.0), 0.0))

    def failed_ecc(*_args):
        raise EccConvergenceError("coarse alignment was inconclusive")

    monkeypatch.setattr(engine, "_find_transform_ecc_euclidean", failed_ecc)
    assert not _should_fast_reject_ecc(_bgr_to_gray(old), _bgr_to_gray(new), 245)


def test_fast_reject_skips_ecc_and_preserves_resize_only_regions(monkeypatch) -> None:
    old = _blank(480, 600)
    new = _blank(480, 600)
    old[60:180, 60:180] = 0
    new[290:410, 410:530] = 0
    monkeypatch.setattr(engine, "_phase_correlate", lambda *_args: ((0.0, 0.0), 0.0))
    ecc_shapes: list[tuple[int, int]] = []

    def low_score_coarse_ecc(template, *_args):
        ecc_shapes.append(template.shape)
        return 0.1, np.eye(2, 3, dtype=np.float32)

    monkeypatch.setattr(engine, "_find_transform_ecc_euclidean", low_score_coarse_ecc)
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
    monkeypatch.setattr(engine, "_phase_correlate", lambda *_args: ((0.0, 0.0), 1.0))

    def successful_ecc(*_args):
        nonlocal calls
        calls += 1
        return 0.95, np.eye(2, 3, dtype=np.float32)

    monkeypatch.setattr(engine, "_find_transform_ecc_euclidean", successful_ecc)
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


def test_regions_edge_shapes_and_offsets_match_legacy_oracle() -> None:
    """Exercise small/degenerate masks and positive (not just 255) pixels."""
    masks = [
        np.zeros((1, 1), dtype=np.uint8),  # empty foreground
        np.array([[0, 255, 0, 255, 0]], dtype=np.uint8),  # single row
        np.array([[17], [0], [23], [0], [17]], dtype=np.uint8),  # single column
        np.array(
            [[255, 0, 0, 0, 255], [0, 255, 0, 255, 0], [0, 0, 255, 0, 0]],
            dtype=np.uint8,
        ),  # diagonal 8-connected pixels
        np.array(
            [[31, 0, 0, 0, 31], [0, 0, 0, 0, 0], [31, 0, 0, 0, 31]],
            dtype=np.uint8,
        ),  # border and non-binary positive values
    ]
    for base in masks:
        for offset_x, offset_y in ((0, 0), (3, 2)):
            mask = np.zeros((base.shape[0] + offset_y + 2, base.shape[1] + offset_x + 2), dtype=np.uint8)
            if base.size:
                mask[offset_y : offset_y + base.shape[0], offset_x : offset_x + base.shape[1]] = base
            for minimum_area in (1, 2, 5):
                for merge_distance in (0, 2, 20):
                    for kind in ("added", "removed"):
                        expected = sorted(
                            _legacy_regions(mask, kind, minimum_area, merge_distance),
                            key=_region_order,
                        )
                        assert _regions(mask, kind, minimum_area, merge_distance) == expected


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
