"""PDF rendering and pixel comparison for native image-based review.

The returned BGRA arrays are deliberately suitable for direct conversion to a
``QImage.Format_ARGB32`` by a PyQt application.  The comparison method uses
the distance between *ink* pixels, rather than a brittle exact RGB comparison:
small rasterisation shifts therefore do not become differences.
"""

from __future__ import annotations

import ctypes
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from math import asin, atan2, cos, degrees, hypot, sin
from os import cpu_count
from pathlib import Path
from typing import Callable, Optional, Sequence

import pymupdf as fitz
import numpy as np
from PIL import Image
from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QTransform

from colors import DifferenceColors


ProgressCallback = Callable[[str, float], None]
BBox = tuple[int, int, int, int]  # x, y, width, height
DEFAULT_INK_THRESHOLD = 245


class AffineWarpSettings:
    """Tuning values for the accelerated, Qt-backed affine path.

    Qt's raster interpolation is very fast but not byte-identical to the
    NumPy sampler. Pixels near the ink threshold are therefore resampled by
    the NumPy implementation before the difference masks are created.
    """

    QT_MAX_ROTATION_DEGREES = 0.0001
    GRAY_GUARD_BAND = 32
    MAX_GUARDED_PIXEL_FRACTION = 0.25


class InkMaskSettings:
    """Exact BT.601 mask constants and color-path dispatch limits."""

    BLUE_WEIGHT = 114
    GREEN_WEIGHT = 587
    RED_WEIGHT = 299
    BGR_LUMA_WEIGHTS = np.array(
        (BLUE_WEIGHT, GREEN_WEIGHT, RED_WEIGHT),
        dtype=np.uint32,
    )
    BGR_LUMA_WEIGHTS.setflags(write=False)
    LUMA_SCALE = 1000
    LUMA_ROUNDING = 500
    CHUNK_ROWS = 128
    COLOR_SAMPLE_STRIDE = 32
    SPARSE_COLOR_MAX_FRACTION = 0.03


class AlignmentSettings:
    """Conservative settings for deciding whether ECC is worth attempting."""

    FAST_REJECT_SHORT_SIDE_PX = 384
    FAST_REJECT_MIN_PHASE_SCORE = 0.20
    FAST_REJECT_MIN_INK_IOU = 0.10
    FAST_REJECT_MAX_ECC_SCORE = 0.20
    FAST_REJECT_ECC_MAX_ITERATIONS = 20
    FAST_REJECT_ECC_EPSILON = 1e-4
    FAST_REJECT_MAX_SHIFT_FRACTION = 0.35


class EccSettings:
    """OpenCV-compatible Euclidean ECC constants and memory limits."""

    GAUSSIAN_KERNEL = np.array((1, 4, 6, 4, 1), dtype=np.float32) / 16.0
    MAX_WORKING_SHORT_SIDE_PX = 768
    MIN_VALID_PIXELS = 9
    REFINEMENT_MIN_SOURCE_SHORT_SIDE_PX = 2048
    REFINEMENT_SHORT_SIDE_PX = 1280
    REFINEMENT_ITERATIONS = 2
    REFINEMENT_MIN_ROTATION_RADIANS = 1e-4
    REFINEMENT_MIN_PHASE_SCORE = 0.50
    REFINEMENT_MAX_PHASE_SHIFT_PX = 2.0


class EccConvergenceError(RuntimeError):
    """Raised when Euclidean ECC cannot produce a trustworthy update."""


class PhaseCorrelationSettings:
    """OpenCV-compatible phase-correlation algorithm constants."""

    OPTIMAL_DFT_FACTORS = (2, 3, 5)
    CENTROID_RADIUS = 2
    PARALLEL_FORWARD_MIN_PIXELS = 300_000


class DistanceTransformSettings:
    """Weights used by OpenCV's approximate 3x3 L2 distance transform."""

    AXIAL_COST = np.float32(0.955)
    DIAGONAL_COST = np.float32(1.3693)


class DifferenceMaskSettings:
    """Dispatch limits for the exact, small-radius tolerance mask path."""

    # Larger tolerance balls cost more to query than the full chamfer transform
    # on realistic drawing pages, so they retain the existing implementation.
    MAX_LOCAL_TOLERANCE_PX = 10
    DIRECT_MAX_WORK_FRACTION = 0.10
    DIRECT_SMALL_OFFSET_COUNT = 32
    DIRECT_SMALL_OFFSET_MAX_WORK_FRACTION = 2.0
    SPAN_MAX_WORK_FRACTION = 0.75


class MaskDilationSettings:
    """Tuning values for grouped sparse change masks."""

    QT_MAX_FOREGROUND_FRACTION = 0.005
    QT_MAX_RUN_FRACTION = 0.005


@dataclass(frozen=True)
class DifferenceRegion:
    """A connected changed area in image coordinates."""

    bbox: BBox
    area: int
    kind: str  # "added" or "removed"


@dataclass(frozen=True)
class AlignmentMetadata:
    """How the old page was made to correspond to the new page."""

    method: str
    success: bool
    score: float | None = None
    affine_matrix: np.ndarray | None = field(default=None, repr=False, compare=False)
    original_old_size: tuple[int, int] = (0, 0)  # width, height
    target_size: tuple[int, int] = (0, 0)  # width, height
    moved: bool = False
    message: str = ""


@dataclass
class RenderedPage:
    """A rendered page, retained separately when callers want its dimensions."""

    bgra: np.ndarray
    page_index: int
    width: int
    height: int
    dpi: float


@dataclass
class ComparisonResult:
    """Complete data for a UI to present a native image-overlay comparison."""

    old_bgra: np.ndarray
    new_bgra: np.ndarray
    added_layer: np.ndarray
    removed_layer: np.ndarray
    added_mask: np.ndarray
    removed_mask: np.ndarray
    added_regions: list[DifferenceRegion]
    removed_regions: list[DifferenceRegion]
    alignment: AlignmentMetadata
    width: int
    height: int
    added_pixels: int
    removed_pixels: int
    changed_pixels: int

    @property
    def has_differences(self) -> bool:
        return self.changed_pixels > 0


def _progress(callback: ProgressCallback | None, stage: str, fraction: float) -> None:
    if callback is not None:
        callback(stage, max(0.0, min(1.0, float(fraction))))


def render_pdf_page(
    pdf_path: str | Path,
    page_index: int = 0,
    *,
    dpi: float = 144.0,
    progress: ProgressCallback | None = None,
) -> RenderedPage:
    """Render one zero-indexed PDF page into a contiguous BGRA image.

    ``ValueError`` is raised for an invalid page/DPI; file and PDF errors are
    intentionally allowed through with PyMuPDF's useful original exceptions.
    """
    if dpi <= 0:
        raise ValueError("dpi must be greater than zero")
    _progress(progress, "opening PDF", 0.0)
    with fitz.open(str(pdf_path)) as document:
        if not 0 <= page_index < document.page_count:
            raise ValueError(f"page_index {page_index} is outside 0..{document.page_count - 1}")
        _progress(progress, "rendering page", 0.25)
        pixmap = document.load_page(page_index).get_pixmap(
            matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0),
            alpha=False,
        )
        # ``samples`` first copies the complete MuPDF raster into a Python
        # ``bytes`` object.  The memoryview avoids that intermediate copy and
        # remains valid while ``pixmap`` is alive.  Copy straight into the
        # final BGRA allocation so rendering does not also build temporary BGR
        # and alpha arrays for these large high-DPI pages.
        sample_rows = np.frombuffer(pixmap.samples_mv, dtype=np.uint8).reshape(
            pixmap.height,
            pixmap.stride,
        )
        pixels = sample_rows[:, : pixmap.width * pixmap.n].reshape(
            pixmap.height,
            pixmap.width,
            pixmap.n,
        )
        bgra = np.empty((pixmap.height, pixmap.width, 4), dtype=np.uint8)
        if pixmap.n == 1:
            bgra[:, :, :3] = pixels
        else:
            bgra[:, :, :3] = pixels[:, :, 2::-1]
        bgra[:, :, 3] = 255
    _progress(progress, "page rendered", 1.0)
    return RenderedPage(
        bgra=np.ascontiguousarray(bgra),
        page_index=page_index,
        width=bgra.shape[1],
        height=bgra.shape[0],
        dpi=dpi,
    )


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty numpy array")
    if image.ndim == 2:
        return _gray_to_bgr(image.astype(np.uint8))
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("image must have 1, 3, or 4 channels")
    image = image.astype(np.uint8, copy=False)
    return np.ascontiguousarray(image[:, :, :3])


def _rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    """Return a contiguous BGR copy of an RGB image without OpenCV."""
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    """Expand one gray channel into contiguous BGR channels."""
    return np.repeat(gray[:, :, np.newaxis], 3, axis=2)


def _bgr_to_bgra(bgr: np.ndarray) -> np.ndarray:
    """Append an opaque alpha channel to a BGR image."""
    alpha = np.full((*bgr.shape[:2], 1), 255, dtype=np.uint8)
    return np.concatenate((bgr, alpha), axis=2)


def _bgr_to_gray(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to luma with the standard BT.601 integer coefficients.

    PDF drawings are commonly monochrome RGB rasters. Because the integer
    coefficients sum to the luma scale, equal B/G/R channels map exactly to
    that shared channel. A cheap sample avoids full equality scans for ordinary
    color images; the complete checks keep this fast path exact.
    """
    channels = bgr[:, :, :3]
    if channels.dtype == np.uint8:
        stride = InkMaskSettings.COLOR_SAMPLE_STRIDE
        sample = channels[::stride, ::stride]
        if (
            np.array_equal(sample[:, :, 0], sample[:, :, 1])
            and np.array_equal(sample[:, :, 0], sample[:, :, 2])
        ):
            blue = channels[:, :, 0]
            if np.array_equal(blue, channels[:, :, 1]) and np.array_equal(
                blue,
                channels[:, :, 2],
            ):
                return np.ascontiguousarray(blue)
    if channels.dtype != np.uint8:
        channels = channels.astype(np.uint32, copy=False)
    weighted = np.einsum(
        "ijk,k->ij",
        channels,
        InkMaskSettings.BGR_LUMA_WEIGHTS,
        optimize=False,
    )
    np.add(weighted, InkMaskSettings.LUMA_ROUNDING, out=weighted)
    np.floor_divide(weighted, InkMaskSettings.LUMA_SCALE, out=weighted)
    return weighted.astype(np.uint8)


def _resize_bgr(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize a BGR image with Pillow, preserving the requested exact size.

    Pillow labels three-channel arrays as RGB, but its resize filters operate
    on every channel independently.  Keeping the channels in BGR order avoids
    two full-image BGR/RGB reversal copies without changing any pixel values.
    """
    if image.shape[:2] == (target_height, target_width):
        return image.copy()
    source_height, source_width = image.shape[:2]
    resample = (
        Image.Resampling.BOX
        if source_width * source_height > target_width * target_height
        else Image.Resampling.BICUBIC
    )
    bgr = Image.fromarray(np.ascontiguousarray(image))
    resized_bgr = bgr.resize((target_width, target_height), resample=resample)
    return np.asarray(resized_bgr)


def _warp_bgr_affine(
    image: np.ndarray,
    matrix: np.ndarray,
    target_width: int,
    target_height: int,
    *,
    ink_threshold: int = DEFAULT_INK_THRESHOLD,
) -> np.ndarray:
    """Apply an inverse-map affine transform, favoring a fast Qt raster path.

    Pure translations and vanishingly small rotations are common in scanned
    drawings. Qt handles those operations in native code. Its few pixels
    around the ink threshold are repaired with the exact NumPy sampler so the
    reported changes remain stable. Larger rotations use the fully precise
    NumPy implementation below.
    """
    transform = np.asarray(matrix, dtype=np.float32).reshape(2, 3)
    if _can_use_qt_affine_warp(transform, ink_threshold):
        try:
            accelerated = _warp_bgr_affine_qt(
                image,
                transform,
                target_width,
                target_height,
                ink_threshold,
            )
        except (RuntimeError, TypeError, ValueError):
            accelerated = None
        if accelerated is not None:
            return accelerated
    return _warp_bgr_affine_numpy(image, transform, target_width, target_height)


def _can_use_qt_affine_warp(matrix: np.ndarray, ink_threshold: int) -> bool:
    """Return whether Qt's fast path can preserve the thresholded result."""
    # Custom thresholds retain the all-NumPy path, which is exact for every
    # accepted transform rather than just the near-translation fast case.
    if ink_threshold != DEFAULT_INK_THRESHOLD or not np.isfinite(matrix).all():
        return False
    a, b, _ = matrix[0]
    c, d, _ = matrix[1]
    determinant = float(a * d - b * c)
    rotation = abs(degrees(atan2(float(c), float(a))))
    return (
        abs(determinant - 1.0) <= 1e-3
        and rotation <= AffineWarpSettings.QT_MAX_ROTATION_DEGREES
        and abs(hypot(float(a), float(c)) - 1.0) <= 1e-3
        and abs(hypot(float(b), float(d)) - 1.0) <= 1e-3
        and abs(float(a * b + c * d)) <= 1e-3
    )


def _warp_bgr_affine_qt(
    image: np.ndarray,
    matrix: np.ndarray,
    target_width: int,
    target_height: int,
    ink_threshold: int,
) -> np.ndarray | None:
    """Use Qt's native raster engine, then repair threshold-edge pixels.

    ``None`` asks the caller to use the exact all-NumPy path instead. This
    happens for image content with too many threshold-adjacent pixels, where
    the repair would no longer be an acceleration.
    """
    source = np.ascontiguousarray(image)
    if not source.flags.writeable:
        source = source.copy()
    source_height, source_width = source.shape[:2]
    a, b, translate_x = (float(value) for value in matrix[0])
    c, d, translate_y = (float(value) for value in matrix[1])
    determinant = a * d - b * c
    if abs(determinant) < 1e-8:
        return None

    source_image = QImage(
        source.data,
        source_width,
        source_height,
        source.strides[0],
        QImage.Format.Format_BGR888,
    )
    output_image = QImage(target_width, target_height, QImage.Format.Format_BGR888)
    output_image.fill(Qt.GlobalColor.white)
    forward = QTransform(
        d / determinant,
        -c / determinant,
        0.0,
        -b / determinant,
        a / determinant,
        0.0,
        (b * translate_y - d * translate_x) / determinant,
        (c * translate_x - a * translate_y) / determinant,
        1.0,
    )
    painter = QPainter(output_image)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    painter.setTransform(forward)
    painter.drawImage(0, 0, source_image)
    painter.end()

    bits = output_image.bits()
    bits.setsize(output_image.sizeInBytes())
    accelerated = np.frombuffer(bits, dtype=np.uint8).reshape(
        target_height,
        output_image.bytesPerLine(),
    )
    accelerated = accelerated[:, : target_width * 3].reshape(target_height, target_width, 3).copy()
    return _repair_affine_threshold_pixels(
        accelerated,
        source,
        matrix,
        ink_threshold,
    )


def _repair_affine_threshold_pixels(
    accelerated: np.ndarray,
    source_image: np.ndarray,
    matrix: np.ndarray,
    ink_threshold: int,
) -> np.ndarray | None:
    """Resample only threshold-adjacent pixels with the exact NumPy formula."""
    gray = _bgr_to_gray(accelerated)
    lower_bound = max(0, ink_threshold - AffineWarpSettings.GRAY_GUARD_BAND)
    # The near-translation restriction and broad repair band cover pixels that
    # could affect the default threshold. Excluding pure white avoids revisiting
    # the full paper background.
    upper_bound = min(254, ink_threshold + AffineWarpSettings.GRAY_GUARD_BAND)
    ys, xs = np.nonzero((gray >= lower_bound) & (gray <= upper_bound))
    max_guarded_pixels = (
        accelerated.shape[0]
        * accelerated.shape[1]
        * AffineWarpSettings.MAX_GUARDED_PIXEL_FRACTION
    )
    if xs.size > max_guarded_pixels:
        return None
    if not xs.size:
        return accelerated

    source_height, source_width = source_image.shape[:2]
    source = np.pad(source_image, ((1, 1), (1, 1), (0, 0)), constant_values=255)
    destination_x = xs.astype(np.float32)
    destination_y = ys.astype(np.float32)
    source_x = matrix[0, 0] * destination_x + matrix[0, 1] * destination_y + matrix[0, 2]
    source_y = matrix[1, 0] * destination_x + matrix[1, 1] * destination_y + matrix[1, 2]
    left_x = np.floor(source_x).astype(np.int32)
    top_y = np.floor(source_y).astype(np.int32)
    fraction_x = source_x - left_x
    fraction_y = source_y - top_y

    left_index = np.clip(left_x, -1, source_width) + 1
    right_index = np.clip(left_x + 1, -1, source_width) + 1
    top_index = np.clip(top_y, -1, source_height) + 1
    bottom_index = np.clip(top_y + 1, -1, source_height) + 1
    top_left = source[top_index, left_index].astype(np.float32)
    top_right = source[top_index, right_index].astype(np.float32)
    bottom_left = source[bottom_index, left_index].astype(np.float32)
    bottom_right = source[bottom_index, right_index].astype(np.float32)
    top = top_left * (1.0 - fraction_x[:, np.newaxis]) + top_right * fraction_x[:, np.newaxis]
    bottom = bottom_left * (1.0 - fraction_x[:, np.newaxis]) + bottom_right * fraction_x[:, np.newaxis]
    interpolated = top * (1.0 - fraction_y[:, np.newaxis]) + bottom * fraction_y[:, np.newaxis]
    accelerated[ys, xs] = np.clip(np.rint(interpolated), 0, 255).astype(np.uint8)
    return accelerated


def _warp_bgr_affine_numpy(
    image: np.ndarray,
    matrix: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Apply an inverse-map affine matrix with NumPy bilinear sampling.

    ``findTransformECC`` and phase correlation provide matrices that map each
    destination pixel back to the source image.  Processing modest row chunks
    keeps this OpenCV-free resampler practical for high-DPI drawings while
    preserving a white border outside the source page.
    """
    source_height, source_width = image.shape[:2]
    transform = np.asarray(matrix, dtype=np.float32).reshape(2, 3)
    output = np.empty((target_height, target_width, 3), dtype=np.uint8)
    source = np.pad(image, ((1, 1), (1, 1), (0, 0)), constant_values=255)
    destination_x = np.arange(target_width, dtype=np.float32)[np.newaxis, :]

    for start_y in range(0, target_height, 256):
        end_y = min(start_y + 256, target_height)
        destination_y = np.arange(start_y, end_y, dtype=np.float32)[:, np.newaxis]
        source_x = transform[0, 0] * destination_x + transform[0, 1] * destination_y + transform[0, 2]
        source_y = transform[1, 0] * destination_x + transform[1, 1] * destination_y + transform[1, 2]
        left_x = np.floor(source_x).astype(np.int32)
        top_y = np.floor(source_y).astype(np.int32)
        fraction_x = source_x - left_x
        fraction_y = source_y - top_y

        left_index = np.clip(left_x, -1, source_width) + 1
        right_index = np.clip(left_x + 1, -1, source_width) + 1
        top_index = np.clip(top_y, -1, source_height) + 1
        bottom_index = np.clip(top_y + 1, -1, source_height) + 1
        top_left = source[top_index, left_index].astype(np.float32)
        top_right = source[top_index, right_index].astype(np.float32)
        bottom_left = source[bottom_index, left_index].astype(np.float32)
        bottom_right = source[bottom_index, right_index].astype(np.float32)

        top = top_left * (1.0 - fraction_x)[..., np.newaxis] + top_right * fraction_x[..., np.newaxis]
        bottom = bottom_left * (1.0 - fraction_x)[..., np.newaxis] + bottom_right * fraction_x[..., np.newaxis]
        interpolated = top * (1.0 - fraction_y)[..., np.newaxis] + bottom * fraction_y[..., np.newaxis]
        output[start_y:end_y] = np.clip(np.rint(interpolated), 0, 255).astype(np.uint8)

    return output


def _threshold_gray_channel(gray: np.ndarray, ink_threshold: int) -> np.ndarray:
    """Return a zero/255 mask while writing the comparison directly to uint8."""
    result = np.empty(gray.shape, dtype=np.uint8)
    np.less(gray, ink_threshold, out=result)
    np.multiply(result, 255, out=result)
    return result


def _weighted_ink_mask(bgr: np.ndarray, ink_threshold: int) -> np.ndarray:
    """Threshold exact BT.601 luma in cache-sized chunks without a gray image."""
    height, width = bgr.shape[:2]
    result = np.empty((height, width), dtype=np.uint8)
    working_rows = min(InkMaskSettings.CHUNK_ROWS, height)
    weighted = np.empty((working_rows, width), dtype=np.uint32)
    scratch = np.empty_like(weighted)
    threshold_sum = np.uint32(
        ink_threshold * InkMaskSettings.LUMA_SCALE - InkMaskSettings.LUMA_ROUNDING
    )

    for start in range(0, height, InkMaskSettings.CHUNK_ROWS):
        stop = min(start + InkMaskSettings.CHUNK_ROWS, height)
        row_count = stop - start
        score = weighted[:row_count]
        temporary = scratch[:row_count]
        source = bgr[start:stop]
        np.multiply(
            source[:, :, 2],
            InkMaskSettings.RED_WEIGHT,
            dtype=np.uint32,
            out=score,
        )
        np.multiply(
            source[:, :, 1],
            InkMaskSettings.GREEN_WEIGHT,
            dtype=np.uint32,
            out=temporary,
        )
        np.add(score, temporary, out=score)
        np.multiply(
            source[:, :, 0],
            InkMaskSettings.BLUE_WEIGHT,
            dtype=np.uint32,
            out=temporary,
        )
        np.add(score, temporary, out=score)
        output = result[start:stop]
        np.less(score, threshold_sum, out=output)
        np.multiply(output, 255, out=output)
    return result


def _sparse_color_ink_mask(bgr: np.ndarray, ink_threshold: int) -> np.ndarray | None:
    """Use one gray channel for a mostly monochrome page and repair color pixels."""
    blue, green, red = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    colored = np.empty(bgr.shape[:2], dtype=bool)
    result = np.empty(bgr.shape[:2], dtype=np.uint8)
    np.not_equal(blue, green, out=colored)
    np.not_equal(blue, red, out=result)
    np.logical_or(colored, result, out=colored)
    colored_count = int(np.count_nonzero(colored))
    if colored_count > colored.size * InkMaskSettings.SPARSE_COLOR_MAX_FRACTION:
        return None

    np.less(blue, ink_threshold, out=result)
    np.multiply(result, 255, out=result)
    if not colored_count:
        return result

    colored_indices = np.flatnonzero(colored)
    pixels = bgr.reshape(-1, bgr.shape[2])[colored_indices]
    weighted = pixels[:, 2].astype(np.uint32)
    np.multiply(weighted, InkMaskSettings.RED_WEIGHT, out=weighted)
    weighted += pixels[:, 1].astype(np.uint32) * InkMaskSettings.GREEN_WEIGHT
    weighted += pixels[:, 0].astype(np.uint32) * InkMaskSettings.BLUE_WEIGHT
    colored_result = np.empty(colored_count, dtype=np.uint8)
    np.less(
        weighted,
        ink_threshold * InkMaskSettings.LUMA_SCALE - InkMaskSettings.LUMA_ROUNDING,
        out=colored_result,
    )
    np.multiply(colored_result, 255, out=colored_result)
    result.ravel()[colored_indices] = colored_result
    return result


def _ink_mask(bgr: np.ndarray, ink_threshold: int) -> np.ndarray:
    """Mark non-paper pixels.  White/near-white pages remain safely empty."""
    if bgr.dtype != np.uint8 or not isinstance(ink_threshold, (int, np.integer)):
        return _threshold_gray_channel(_bgr_to_gray(bgr), ink_threshold)
    if ink_threshold <= 0:
        return np.zeros(bgr.shape[:2], dtype=np.uint8)
    if ink_threshold > 255:
        return np.full(bgr.shape[:2], 255, dtype=np.uint8)

    stride = InkMaskSettings.COLOR_SAMPLE_STRIDE
    sample = bgr[::stride, ::stride]
    sample_colored = (sample[:, :, 0] != sample[:, :, 1]) | (
        sample[:, :, 0] != sample[:, :, 2]
    )
    sampled_color_count = int(np.count_nonzero(sample_colored))
    if sampled_color_count == 0 and np.array_equal(
        bgr[:, :, 0], bgr[:, :, 1]
    ) and np.array_equal(bgr[:, :, 0], bgr[:, :, 2]):
        return _threshold_gray_channel(bgr[:, :, 0], ink_threshold)

    if (
        sampled_color_count
        <= sample_colored.size * InkMaskSettings.SPARSE_COLOR_MAX_FRACTION
    ):
        sparse_result = _sparse_color_ink_mask(bgr, ink_threshold)
        if sparse_result is not None:
            return sparse_result
    return _weighted_ink_mask(bgr, ink_threshold)


def _coarse_gray(image: np.ndarray) -> np.ndarray:
    """Reduce a grayscale page for a cheap alignment-confidence check."""
    height, width = image.shape
    scale = min(1.0, AlignmentSettings.FAST_REJECT_SHORT_SIDE_PX / min(height, width))
    if scale == 1.0:
        return image
    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))
    return np.asarray(
        Image.fromarray(image).resize((target_width, target_height), resample=Image.Resampling.BOX)
    )


def _optimal_dft_size(size: int) -> int:
    """Return OpenCV's next 2/3/5-factorable DFT length."""
    if size < 1:
        raise ValueError("DFT size must be positive")
    candidate = size
    while True:
        remainder = candidate
        for factor in PhaseCorrelationSettings.OPTIMAL_DFT_FACTORS:
            while remainder % factor == 0:
                remainder //= factor
        if remainder == 1:
            return candidate
        candidate += 1


def _phase_forward_spectra(
    first: np.ndarray,
    second: np.ndarray,
    padded_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Transform two read-only phase inputs, concurrently when worthwhile."""

    def transform(image: np.ndarray) -> np.ndarray:
        return np.fft.rfft2(image, s=padded_shape, axes=(-2, -1))

    padded_pixels = padded_shape[0] * padded_shape[1]
    if (
        padded_pixels < PhaseCorrelationSettings.PARALLEL_FORWARD_MIN_PIXELS
        or (cpu_count() or 1) < 2
    ):
        return transform(first), transform(second)

    executor: ThreadPoolExecutor | None = None
    try:
        executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="pdf-phase-fft",
        )
        first_future = executor.submit(transform, first)
        second_future = executor.submit(transform, second)
    except (OSError, RuntimeError):
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        return transform(first), transform(second)
    try:
        return first_future.result(), second_future.result()
    finally:
        executor.shutdown()


def _phase_correlate(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[tuple[float, float], float]:
    """Return OpenCV-compatible translation and confidence using NumPy FFTs.

    OpenCV zero-pads to efficient DFT dimensions, normalizes the cross-power
    spectrum, and refines the peak with a clamped 5x5 weighted centroid.  The
    inverse NumPy FFT is already divided by the image area, so its centroid sum
    is the same response that OpenCV obtains after its final area division.
    """
    first = np.asarray(first)
    second = np.asarray(second)
    if first.ndim != 2 or first.shape != second.shape or not first.size:
        raise ValueError("phase-correlation inputs must be non-empty, same-size 2D arrays")
    if first.dtype != second.dtype or first.dtype not in (
        np.dtype(np.float32),
        np.dtype(np.float64),
    ):
        raise ValueError("phase-correlation inputs must have the same float32 or float64 dtype")

    padded_height = _optimal_dft_size(first.shape[0])
    padded_width = _optimal_dft_size(first.shape[1])
    padded_shape = (padded_height, padded_width)
    first_spectrum, second_spectrum = _phase_forward_spectra(
        first,
        second,
        padded_shape,
    )
    np.conjugate(second_spectrum, out=second_spectrum)
    np.multiply(first_spectrum, second_spectrum, out=first_spectrum)

    # The second spectrum is dead after forming the cross-power product. Its
    # complex storage holds exactly two real arrays, so reuse those contiguous
    # halves for magnitude and normalization instead of allocating both.
    real_dtype = first_spectrum.real.dtype
    workspace = second_spectrum.view(real_dtype).reshape(-1)
    spectrum_size = first_spectrum.size
    magnitude = workspace[:spectrum_size].reshape(first_spectrum.shape)
    normalizer = workspace[spectrum_size:].reshape(first_spectrum.shape)
    np.abs(first_spectrum, out=magnitude)
    floating_info = np.finfo(first.dtype)
    # P * |P| / (|P|**2 + epsilon), rearranged to avoid overflow.
    np.maximum(magnitude, floating_info.tiny, out=normalizer)
    np.divide(floating_info.eps, normalizer, out=normalizer)
    np.add(normalizer, magnitude, out=normalizer)
    np.divide(first_spectrum, normalizer, out=first_spectrum)
    correlation = np.fft.irfft2(first_spectrum, s=padded_shape, axes=(-2, -1))

    # Locate the unshifted peak, then map only its 5x5 neighborhood into the
    # shifted coordinate system. This avoids copying the full correlation map.
    unshifted_y, unshifted_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    peak_y = (int(unshifted_y) + padded_height // 2) % padded_height
    peak_x = (int(unshifted_x) + padded_width // 2) % padded_width
    radius = PhaseCorrelationSettings.CENTROID_RADIUS
    shifted_ys = np.arange(max(0, peak_y - radius), min(padded_height, peak_y + radius + 1))
    shifted_xs = np.arange(max(0, peak_x - radius), min(padded_width, peak_x + radius + 1))
    source_ys = (shifted_ys - padded_height // 2) % padded_height
    source_xs = (shifted_xs - padded_width // 2) % padded_width
    centroid_patch = correlation[np.ix_(source_ys, source_xs)].astype(np.float64, copy=False)
    response = float(np.sum(centroid_patch, dtype=np.float64))
    denominator = response + np.finfo(np.float64).eps
    centroid_x = float(
        np.sum(centroid_patch * shifted_xs[None, :], dtype=np.float64) / denominator
    )
    centroid_y = float(
        np.sum(centroid_patch * shifted_ys[:, None], dtype=np.float64) / denominator
    )
    return (
        (padded_width / 2.0 - centroid_x, padded_height / 2.0 - centroid_y),
        response,
    )


def _warp_binary_translation_nearest(
    source: np.ndarray,
    shift: tuple[float, float],
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """Apply OpenCV-compatible inverse translation with nearest sampling.

    This is the only affine operation needed by the coarse ink-overlap gate.
    OpenCV first stores its translation matrix as float32, maps every target
    coordinate back into the source, rounds to the nearest-even integer, and
    fills coordinates outside the source with zero.
    """
    source = np.asarray(source)
    if source.ndim != 2 or not source.size:
        raise ValueError("source must be a non-empty 2D array")
    if target_width < 1 or target_height < 1:
        raise ValueError("target dimensions must be positive")
    shift_x, shift_y = np.asarray(shift, dtype=np.float32)
    if not np.isfinite(shift_x) or not np.isfinite(shift_y):
        raise ValueError("translation must be finite")

    source_xs = np.rint(
        np.arange(target_width, dtype=np.float64) + np.float64(shift_x)
    ).astype(np.intp)
    source_ys = np.rint(
        np.arange(target_height, dtype=np.float64) + np.float64(shift_y)
    ).astype(np.intp)
    valid_x = (source_xs >= 0) & (source_xs < source.shape[1])
    valid_y = (source_ys >= 0) & (source_ys < source.shape[0])
    output = np.zeros((target_height, target_width), dtype=source.dtype)
    if np.any(valid_x) and np.any(valid_y):
        destination_xs = np.flatnonzero(valid_x)
        destination_ys = np.flatnonzero(valid_y)
        selected_source_xs = source_xs[valid_x]
        selected_source_ys = source_ys[valid_y]
        x_is_contiguous = selected_source_xs.size == 1 or np.all(
            np.diff(selected_source_xs) == 1
        )
        y_is_contiguous = selected_source_ys.size == 1 or np.all(
            np.diff(selected_source_ys) == 1
        )
        if x_is_contiguous and y_is_contiguous:
            output[
                destination_ys[0] : destination_ys[-1] + 1,
                destination_xs[0] : destination_xs[-1] + 1,
            ] = source[
                selected_source_ys[0] : selected_source_ys[-1] + 1,
                selected_source_xs[0] : selected_source_xs[-1] + 1,
            ]
        else:
            output[np.ix_(destination_ys, destination_xs)] = source[
                np.ix_(selected_source_ys, selected_source_xs)
            ]
    return output


def _coarse_ink_iou(
    old_gray: np.ndarray,
    new_gray: np.ndarray,
    shift: tuple[float, float],
    ink_threshold: int,
) -> float:
    """Return raw ink IoU after phase's destination-to-source translation."""
    old_ink = (old_gray < ink_threshold).astype(np.uint8)
    new_ink = new_gray < ink_threshold
    old_aligned = _warp_binary_translation_nearest(
        old_ink,
        shift,
        new_gray.shape[1],
        new_gray.shape[0],
    )
    union = np.count_nonzero((old_aligned > 0) | new_ink)
    if not union:
        return 1.0
    intersection = np.count_nonzero((old_aligned > 0) & new_ink)
    return float(intersection / union)


def _gaussian_blur_5x5(image: np.ndarray) -> np.ndarray:
    """Apply OpenCV's default 5x5, sigma-zero Gaussian to a float image."""
    source = np.asarray(image, dtype=np.float32)
    kernel = EccSettings.GAUSSIAN_KERNEL
    horizontal_source = np.pad(source, ((0, 0), (2, 2)), mode="reflect")
    horizontal = sum(
        weight * horizontal_source[:, offset : offset + source.shape[1]]
        for offset, weight in enumerate(kernel)
    )
    vertical_source = np.pad(horizontal, ((2, 2), (0, 0)), mode="reflect")
    return np.asarray(
        sum(
            weight * vertical_source[offset : offset + source.shape[0]]
            for offset, weight in enumerate(kernel)
        ),
        dtype=np.float32,
    )


def _central_gradients(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the source gradients used by OpenCV's ECC implementation."""
    source = np.asarray(image, dtype=np.float32)
    gradient_x = np.zeros_like(source)
    gradient_y = np.zeros_like(source)
    if source.shape[1] > 2:
        gradient_x[:, 1:-1] = (source[:, 2:] - source[:, :-2]) * np.float32(0.5)
    if source.shape[0] > 2:
        gradient_y[1:-1] = (source[2:] - source[:-2]) * np.float32(0.5)
    return gradient_x, gradient_y


def _warp_ecc_channels_numpy(
    padded_channels: np.ndarray,
    source_shape: tuple[int, int],
    matrix: np.ndarray,
    destination_x: np.ndarray,
    destination_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-sample ECC channels into contiguous channel-first storage."""
    source_height, source_width = source_shape
    source_x = (
        matrix[0, 0] * destination_x
        + matrix[0, 1] * destination_y
        + matrix[0, 2]
    )
    source_y = (
        matrix[1, 0] * destination_x
        + matrix[1, 1] * destination_y
        + matrix[1, 2]
    )
    nearest_x = np.rint(source_x)
    nearest_y = np.rint(source_y)
    valid = (
        (nearest_x >= 0)
        & (nearest_x < source_width)
        & (nearest_y >= 0)
        & (nearest_y < source_height)
    )

    left_x = np.floor(source_x).astype(np.int32)
    top_y = np.floor(source_y).astype(np.int32)
    fraction_x = source_x - left_x
    fraction_y = source_y - top_y
    left_index = np.clip(left_x, -1, source_width) + 1
    right_index = np.clip(left_x + 1, -1, source_width) + 1
    top_index = np.clip(top_y, -1, source_height) + 1
    bottom_index = np.clip(top_y + 1, -1, source_height) + 1

    top = padded_channels[top_index, left_index].copy()
    top *= (1.0 - fraction_x)[..., np.newaxis]
    top += padded_channels[top_index, right_index] * fraction_x[..., np.newaxis]
    bottom = padded_channels[bottom_index, left_index].copy()
    bottom *= (1.0 - fraction_x)[..., np.newaxis]
    bottom += padded_channels[bottom_index, right_index] * fraction_x[..., np.newaxis]
    top *= (1.0 - fraction_y)[..., np.newaxis]
    top += bottom * fraction_y[..., np.newaxis]
    return np.ascontiguousarray(np.moveaxis(top, 2, 0)), valid


def _ecc_pillow_affine(matrix: np.ndarray) -> tuple[float, ...]:
    """Return Pillow inverse-map coefficients for pixel-centered ECC coordinates."""
    a, b, translate_x = (float(value) for value in matrix[0])
    c, d, translate_y = (float(value) for value in matrix[1])
    # Pillow evaluates its affine map at the center of each destination pixel,
    # then its bilinear sampler subtracts half a pixel. This offset keeps the
    # effective source coordinate equal to M @ [x, y, 1], as used by NumPy.
    return (
        a,
        b,
        translate_x + 0.5 - 0.5 * (a + b),
        c,
        d,
        translate_y + 0.5 - 0.5 * (c + d),
    )


def _prepare_ecc_channel_images(
    padded_channels: np.ndarray,
) -> tuple[Image.Image, ...]:
    """Copy the three ECC channels into reusable Pillow float images."""
    channels = padded_channels[1:-1, 1:-1]
    return tuple(
        Image.fromarray(np.ascontiguousarray(channels[:, :, index]))
        for index in range(channels.shape[2])
    )


class _ArrowSchema(ctypes.Structure):
    """ABI-stable Arrow C Data Interface schema header."""


_ArrowSchema._fields_ = [
    ("format", ctypes.c_char_p),
    ("name", ctypes.c_char_p),
    ("metadata", ctypes.c_char_p),
    ("flags", ctypes.c_int64),
    ("n_children", ctypes.c_int64),
    ("children", ctypes.c_void_p),
    ("dictionary", ctypes.c_void_p),
    ("release", ctypes.c_void_p),
    ("private_data", ctypes.c_void_p),
]


class _ArrowArray(ctypes.Structure):
    """ABI-stable Arrow C Data Interface array header."""


_ArrowArray._fields_ = [
    ("length", ctypes.c_int64),
    ("null_count", ctypes.c_int64),
    ("offset", ctypes.c_int64),
    ("n_buffers", ctypes.c_int64),
    ("n_children", ctypes.c_int64),
    ("buffers", ctypes.POINTER(ctypes.c_void_p)),
    ("children", ctypes.c_void_p),
    ("dictionary", ctypes.c_void_p),
    ("release", ctypes.c_void_p),
    ("private_data", ctypes.c_void_p),
]

try:
    _py_capsule_get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
except AttributeError:  # Non-CPython runtimes retain the array-interface path.
    _py_capsule_get_pointer = None
else:
    _py_capsule_get_pointer.argtypes = (ctypes.py_object, ctypes.c_char_p)
    _py_capsule_get_pointer.restype = ctypes.c_void_p


def _copy_pillow_float_image(
    image: Image.Image,
    destination: np.ndarray,
) -> bool:
    """Copy a mode-F image through Arrow when Pillow exposes zero-copy data.

    Pillow 11.2.1 and newer can export its contiguous pixel allocation through
    the ABI-stable Arrow C Data Interface.  Reading that buffer directly avoids
    ``Image.__array_interface__`` serializing the complete image to ``bytes``
    before NumPy can copy it.  The capsules stay alive through ``np.copyto``;
    older Pillow builds and unsupported allocations retain the exact existing
    array-interface path.
    """
    export_arrow = getattr(image, "__arrow_c_array__", None)
    if (
        image.mode == "F"
        and callable(export_arrow)
        and _py_capsule_get_pointer is not None
    ):
        try:
            schema_capsule, array_capsule = export_arrow()
            schema_pointer = _py_capsule_get_pointer(
                schema_capsule,
                b"arrow_schema",
            )
            array_pointer = _py_capsule_get_pointer(
                array_capsule,
                b"arrow_array",
            )
            if not schema_pointer or not array_pointer:
                raise ValueError("Arrow export did not return valid capsules")
            arrow_schema = ctypes.cast(
                schema_pointer,
                ctypes.POINTER(_ArrowSchema),
            ).contents
            arrow_array = ctypes.cast(
                array_pointer,
                ctypes.POINTER(_ArrowArray),
            ).contents
            if (
                arrow_schema.format != b"f"
                or arrow_schema.n_children != 0
                or arrow_array.length != destination.size
                or arrow_array.null_count != 0
                or arrow_array.offset < 0
                or arrow_array.n_buffers < 2
                or arrow_array.n_children != 0
                or not arrow_array.buffers
                or not arrow_array.buffers[1]
            ):
                raise ValueError("unsupported Pillow Arrow array layout")
            data_address = int(arrow_array.buffers[1]) + (
                int(arrow_array.offset) * ctypes.sizeof(ctypes.c_float)
            )
            values = (ctypes.c_float * destination.size).from_address(
                data_address,
            )
            source = np.ctypeslib.as_array(values).reshape(destination.shape)
            np.copyto(destination, source)
            # Keep both capsules referenced until after the shared pixels have
            # been copied. Their destructors release Pillow's exported view.
            _ = schema_capsule, array_capsule
            return True
        except (
            AttributeError,
            BufferError,
            ctypes.ArgumentError,
            OSError,
            OverflowError,
            RuntimeError,
            SystemError,
            TypeError,
            ValueError,
        ):
            pass
    np.copyto(destination, np.asarray(image))
    return False


def _warp_ecc_channels_pillow(
    padded_channels: np.ndarray,
    source_shape: tuple[int, int],
    matrix: np.ndarray,
    destination_x: np.ndarray,
    destination_y: np.ndarray,
    source_images: tuple[Image.Image, ...],
    executor: ThreadPoolExecutor | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp ECC channels in Pillow's compiled sampler, repairing its borders."""
    source_height, source_width = source_shape
    destination_height = destination_y.shape[0]
    destination_width = destination_x.shape[1]
    affine = _ecc_pillow_affine(matrix)
    warped = np.empty(
        (len(source_images), destination_height, destination_width),
        dtype=np.float32,
    )

    def transform_channel(item: tuple[int, Image.Image]) -> None:
        index, image = item
        transformed = image.transform(
            (destination_width, destination_height),
            Image.Transform.AFFINE,
            affine,
            resample=Image.Resampling.BILINEAR,
            fillcolor=0.0,
        )
        # Every task owns one disjoint channel plane. Copying here keeps the
        # Pillow-to-NumPy conversion parallel and avoids a later stack copy.
        _copy_pillow_float_image(transformed, warped[index])

    if executor is None:
        for item in enumerate(source_images):
            transform_channel(item)
        futures = ()
    else:
        # Coordinate validity is independent of the sampled channel values.
        # Start the compiled Pillow work first so NumPy can build the mask on
        # this thread while the three channel transforms run in parallel.
        futures = tuple(
            executor.submit(transform_channel, item)
            for item in enumerate(source_images)
        )

    # Preserve the original nearest-coordinate validity rule exactly. Pillow's
    # nearest filter differs at a few half-pixel ties.
    source_x = (
        matrix[0, 0] * destination_x
        + matrix[0, 1] * destination_y
        + matrix[0, 2]
    )
    source_y = (
        matrix[1, 0] * destination_x
        + matrix[1, 1] * destination_y
        + matrix[1, 2]
    )
    nearest_x = np.rint(source_x)
    nearest_y = np.rint(source_y)
    valid = (
        (nearest_x >= 0)
        & (nearest_x < source_width)
        & (nearest_y >= 0)
        & (nearest_y < source_height)
    )

    # Pillow extends edge pixels during bilinear sampling, while ECC uses a
    # zero-valued constant border. Only the narrow valid perimeter can observe
    # that difference, so resample those pixels with the original arithmetic.
    edge = valid & (
        (source_x < 0)
        | (source_x > source_width - 1)
        | (source_y < 0)
        | (source_y > source_height - 1)
    )
    edge_y, edge_x = np.nonzero(edge)
    if edge_x.size:
        sample_x = source_x[edge_y, edge_x]
        sample_y = source_y[edge_y, edge_x]
        left_x = np.floor(sample_x).astype(np.int32)
        top_y = np.floor(sample_y).astype(np.int32)
        fraction_x = sample_x - left_x
        fraction_y = sample_y - top_y
        left_index = np.clip(left_x, -1, source_width) + 1
        right_index = np.clip(left_x + 1, -1, source_width) + 1
        top_index = np.clip(top_y, -1, source_height) + 1
        bottom_index = np.clip(top_y + 1, -1, source_height) + 1

        top = padded_channels[top_index, left_index].copy()
        top *= (1.0 - fraction_x)[:, np.newaxis]
        top += padded_channels[top_index, right_index] * fraction_x[:, np.newaxis]
        bottom = padded_channels[bottom_index, left_index].copy()
        bottom *= (1.0 - fraction_x)[:, np.newaxis]
        bottom += padded_channels[bottom_index, right_index] * fraction_x[:, np.newaxis]
        top *= (1.0 - fraction_y)[:, np.newaxis]
        top += bottom * fraction_y[:, np.newaxis]
        repaired_edge = top.T
    else:
        repaired_edge = None

    if futures:
        for future in futures:
            future.result()
    if repaired_edge is not None:
        warped[:, edge_y, edge_x] = repaired_edge
    return warped, valid


class _EccChannelWarper:
    """Reuse Pillow inputs and permanently fall back after an accelerator error."""

    def __init__(
        self,
        padded_channels: np.ndarray,
        source_shape: tuple[int, int],
        destination_x: np.ndarray,
        destination_y: np.ndarray,
    ) -> None:
        self._padded_channels = padded_channels
        self._source_shape = source_shape
        self._destination_x = destination_x
        self._destination_y = destination_y
        self._source_images: tuple[Image.Image, ...] | None = None
        self._executor: ThreadPoolExecutor | None = None
        try:
            self._source_images = _prepare_ecc_channel_images(padded_channels)
            # Pillow's compiled transforms release the GIL. The independent
            # intensity/x-gradient/y-gradient warps can therefore run together.
            self._executor = ThreadPoolExecutor(
                max_workers=len(self._source_images),
                thread_name_prefix="pdf-ecc-warp",
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            self._source_images = None
            self._executor = None

    def __enter__(self) -> _EccChannelWarper:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def _disable_accelerator(self) -> None:
        self.close()
        self._source_images = None

    def warp(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._source_images is not None:
            try:
                return _warp_ecc_channels_pillow(
                    self._padded_channels,
                    self._source_shape,
                    matrix,
                    self._destination_x,
                    self._destination_y,
                    self._source_images,
                    self._executor,
                )
            except (OSError, RuntimeError, ValueError):
                self._disable_accelerator()
        return _warp_ecc_channels_numpy(
            self._padded_channels,
            self._source_shape,
            matrix,
            self._destination_x,
            self._destination_y,
        )


def _warp_ecc_channels(
    padded_channels: np.ndarray,
    source_shape: tuple[int, int],
    matrix: np.ndarray,
    destination_x: np.ndarray,
    destination_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-sample ECC channels through Pillow, with an exact NumPy fallback."""
    with _EccChannelWarper(
        padded_channels,
        source_shape,
        destination_x,
        destination_y,
    ) as warper:
        return warper.warp(matrix)


class _EccIterationWorkspace:
    """Reuse the large arrays needed by every Euclidean ECC iteration."""

    def __init__(self, shape: tuple[int, int]) -> None:
        height, width = shape
        maximum_pixels = height * width
        self._width = width
        self._sample = np.empty(maximum_pixels, dtype=np.float32)
        self._template_zero_mean = np.empty(maximum_pixels, dtype=np.float64)
        self._input_zero_mean = np.empty(maximum_pixels, dtype=np.float64)
        self._x_coordinates = np.empty(maximum_pixels, dtype=np.float32)
        self._y_coordinates = np.empty(maximum_pixels, dtype=np.float32)
        self._jacobian = np.empty(3 * maximum_pixels, dtype=np.float32)
        self._projection_jacobian = np.empty(
            3 * maximum_pixels,
            dtype=np.float64,
        )
        self._valid_indices: np.ndarray | None = None
        self._valid_count = 0

    def prepare_values(
        self,
        template: np.ndarray,
        warped: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        """Gather valid pixels and calculate their zero-mean statistics."""

        valid_indices = np.flatnonzero(valid)
        count = int(valid_indices.size)
        if count < EccSettings.MIN_VALID_PIXELS:
            raise EccConvergenceError("ECC has too little overlapping image area")
        self._valid_indices = valid_indices
        self._valid_count = count

        sample = self._sample[:count]
        template_zero_mean = self._template_zero_mean[:count]
        input_zero_mean = self._input_zero_mean[:count]
        # ``flatnonzero`` guarantees that every index is nonnegative and in
        # bounds. ``clip`` is therefore equivalent to the default ``raise``
        # behavior here, while allowing NumPy to write directly into ``out``.
        np.take(template, valid_indices, out=sample, mode="clip")
        np.subtract(
            sample,
            np.mean(sample, dtype=np.float64),
            out=template_zero_mean,
        )
        jacobian = self._jacobian[: 3 * count].reshape(3, count)
        # A row-wise gather stays on NumPy's fast contiguous 1-D indexing path.
        # Only one temporary channel is live at a time, and the destination is
        # still the reusable parameter-major Jacobian buffer.
        warped_channels = warped.reshape(3, -1)
        for source_channel, destination_channel in zip(warped_channels, jacobian):
            np.copyto(destination_channel, source_channel[valid_indices])
        input_values = jacobian[0]
        np.subtract(
            input_values,
            np.mean(input_values, dtype=np.float64),
            out=input_zero_mean,
        )

        # Before it receives the promoted Jacobian, one row is a reusable
        # float64 reduction buffer for the three image statistics.
        projection_jacobian = self._projection_jacobian[: 3 * count].reshape(
            3,
            count,
        )
        arithmetic = projection_jacobian[0]
        np.multiply(template_zero_mean, template_zero_mean, out=arithmetic)
        template_norm_squared = float(np.sum(arithmetic, dtype=np.float64))
        np.multiply(input_zero_mean, input_zero_mean, out=arithmetic)
        input_norm_squared = float(np.sum(arithmetic, dtype=np.float64))
        np.multiply(template_zero_mean, input_zero_mean, out=arithmetic)
        correlation = float(np.sum(arithmetic, dtype=np.float64))
        return (
            template_zero_mean,
            input_zero_mean,
            template_norm_squared,
            input_norm_squared,
            correlation,
        )

    def build_jacobians(
        self,
        sine: np.float32,
        cosine: np.float32,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build contiguous parameter rows for the current valid pixels."""

        valid_indices = self._valid_indices
        if valid_indices is None:
            raise RuntimeError("ECC values must be prepared before its Jacobian")
        count = self._valid_count
        x_coordinates = self._x_coordinates[:count]
        y_coordinates = self._y_coordinates[:count]
        # The sampling buffer is dead after ``prepare_values`` and can now be
        # reused for the Euclidean coordinate products.
        coordinate_scratch = self._sample[:count]
        np.divmod(
            valid_indices,
            self._width,
            out=(y_coordinates, x_coordinates),
            casting="unsafe",
        )

        jacobian = self._jacobian[: 3 * count].reshape(3, count)
        jacobian_theta = jacobian[0]
        jacobian_gradient_x = jacobian[1]
        jacobian_gradient_y = jacobian[2]

        # These in-place operations preserve the original float32 evaluation
        # order while avoiding several full-size coordinate temporaries.
        np.multiply(cosine, x_coordinates, out=jacobian_theta)
        np.multiply(sine, y_coordinates, out=coordinate_scratch)
        np.subtract(jacobian_theta, coordinate_scratch, out=jacobian_theta)
        np.multiply(-sine, x_coordinates, out=x_coordinates)
        np.multiply(cosine, y_coordinates, out=coordinate_scratch)
        np.subtract(x_coordinates, coordinate_scratch, out=x_coordinates)
        np.multiply(jacobian_gradient_x, x_coordinates, out=x_coordinates)
        np.multiply(
            jacobian_gradient_y,
            jacobian_theta,
            out=coordinate_scratch,
        )
        np.add(x_coordinates, coordinate_scratch, out=jacobian_theta)

        # Zero-mean vectors are float64, so mixed-dtype matrix products would
        # promote this same Jacobian afresh for every projection. Promote it
        # once per iteration and reuse the contiguous parameter-major copy.
        projection_jacobian = self._projection_jacobian[: 3 * count].reshape(
            3,
            count,
        )
        np.copyto(projection_jacobian, jacobian, casting="unsafe")
        return jacobian, projection_jacobian


def _find_transform_ecc_euclidean_core(
    template: np.ndarray,
    input_image: np.ndarray,
    initial_matrix: np.ndarray,
    max_iterations: int,
    epsilon: float,
) -> tuple[float, np.ndarray]:
    """Optimize OpenCV's forward-additive Euclidean ECC equations."""
    template_blurred = _gaussian_blur_5x5(template)
    input_blurred = _gaussian_blur_5x5(input_image)
    gradient_x, gradient_y = _central_gradients(input_blurred)
    source_channels = np.stack((input_blurred, gradient_x, gradient_y), axis=2)
    padded_channels = np.pad(
        source_channels,
        ((1, 1), (1, 1), (0, 0)),
        mode="constant",
    )
    height, width = template_blurred.shape
    destination_x = np.arange(width, dtype=np.float32)[np.newaxis, :]
    destination_y = np.arange(height, dtype=np.float32)[:, np.newaxis]
    matrix = np.asarray(initial_matrix, dtype=np.float32).reshape(2, 3).copy()
    iteration_workspace = _EccIterationWorkspace(template_blurred.shape)
    rho = -1.0
    with _EccChannelWarper(
        padded_channels,
        input_blurred.shape,
        destination_x,
        destination_y,
    ) as warper:
        for _ in range(max_iterations):
            last_rho = rho
            warped, valid = warper.warp(matrix)
            (
                template_zero_mean,
                input_zero_mean,
                template_norm_squared,
                input_norm_squared,
                correlation,
            ) = iteration_workspace.prepare_values(template_blurred, warped, valid)
            if template_norm_squared <= 0.0 or input_norm_squared <= 0.0:
                raise EccConvergenceError("ECC cannot align a uniform image")
            rho = correlation / np.sqrt(template_norm_squared * input_norm_squared)
            if not np.isfinite(rho):
                raise EccConvergenceError("ECC correlation became non-finite")

            angle = asin(float(np.clip(matrix[1, 0], -1.0, 1.0)))
            sine = np.float32(sin(angle))
            cosine = np.float32(cos(angle))
            jacobian, projection_jacobian = iteration_workspace.build_jacobians(
                sine,
                cosine,
            )

            hessian = jacobian @ jacobian.T
            image_projection = projection_jacobian @ input_zero_mean
            template_projection = projection_jacobian @ template_zero_mean
            try:
                image_projection_hessian = np.linalg.solve(hessian, image_projection)
            except np.linalg.LinAlgError as exc:
                raise EccConvergenceError("ECC Hessian is singular") from exc
            lambda_numerator = input_norm_squared - float(
                image_projection @ image_projection_hessian
            )
            lambda_denominator = correlation - float(
                template_projection @ image_projection_hessian
            )
            if not np.isfinite(lambda_denominator) or lambda_denominator <= 0.0:
                raise EccConvergenceError(
                    "ECC images are uncorrelated or non-overlapping"
                )
            illumination_scale = lambda_numerator / lambda_denominator
            # The template vector is dead after its projection, so it can hold
            # the error without allocating two more float64 arrays.
            np.multiply(
                illumination_scale,
                template_zero_mean,
                out=template_zero_mean,
            )
            np.subtract(
                template_zero_mean,
                input_zero_mean,
                out=template_zero_mean,
            )
            error_projection = projection_jacobian @ template_zero_mean
            try:
                delta = np.linalg.solve(hessian, error_projection)
            except np.linalg.LinAlgError as exc:
                raise EccConvergenceError("ECC Hessian is singular") from exc
            if not np.isfinite(delta).all():
                raise EccConvergenceError("ECC update became non-finite")

            angle += float(delta[0])
            matrix[0, 0] = cos(angle)
            matrix[0, 1] = -sin(angle)
            matrix[1, 0] = sin(angle)
            matrix[1, 1] = cos(angle)
            matrix[0, 2] += delta[1]
            matrix[1, 2] += delta[2]
            if abs(rho - last_rho) < epsilon:
                break

    return float(rho), matrix


def _resize_ecc_pair(
    template: np.ndarray,
    input_image: np.ndarray,
    maximum_short_side: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Resize an ECC pair together and return its uniform coordinate scale."""
    short_side = min(template.shape)
    scale = min(1.0, maximum_short_side / short_side)
    if scale == 1.0:
        return template, input_image, scale
    target_height = max(1, round(template.shape[0] * scale))
    target_width = max(1, round(template.shape[1] * scale))
    template_working = np.asarray(
        Image.fromarray(template).resize(
            (target_width, target_height),
            resample=Image.Resampling.BOX,
        ),
        dtype=np.float32,
    )
    input_working = np.asarray(
        Image.fromarray(input_image).resize(
            (target_width, target_height),
            resample=Image.Resampling.BOX,
        ),
        dtype=np.float32,
    )
    return template_working, input_working, scale


def _scaled_ecc_matrix(matrix: np.ndarray, scale: float) -> np.ndarray:
    """Move a destination-to-source transform into a scaled coordinate space."""
    scaled = np.asarray(matrix, dtype=np.float32).reshape(2, 3).copy()
    scaled[:, 2] *= np.float32(scale)
    return scaled


def _polish_ecc_translation(
    template_working: np.ndarray,
    input_working: np.ndarray,
    matrix: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Use phase correlation to remove a small post-ECC translation residual."""
    height, width = template_working.shape
    working_matrix = _scaled_ecc_matrix(matrix, scale)
    input_u8 = np.clip(np.rint(input_working * 255.0), 0, 255).astype(np.uint8)
    template_u8 = np.clip(np.rint(template_working * 255.0), 0, 255).astype(np.uint8)
    warped_input = _warp_bgr_affine_numpy(
        _gray_to_bgr(input_u8),
        working_matrix,
        width,
        height,
    )[:, :, 0]
    try:
        residual, response = _phase_correlate(
            template_u8.astype(np.float32),
            warped_input.astype(np.float32),
        )
    except (MemoryError, TypeError, ValueError):
        return matrix
    if (
        not np.isfinite(response)
        or response < EccSettings.REFINEMENT_MIN_PHASE_SCORE
        or not np.isfinite(residual).all()
        or np.hypot(*residual) > EccSettings.REFINEMENT_MAX_PHASE_SHIFT_PX
    ):
        return matrix
    polished = matrix.copy()
    correction = np.asarray(residual, dtype=np.float32) / np.float32(scale)
    polished[:, 2] += polished[:, :2] @ correction
    return polished


def _find_transform_ecc_euclidean(
    template: np.ndarray,
    input_image: np.ndarray,
    initial_matrix: np.ndarray,
    max_iterations: int,
    epsilon: float,
) -> tuple[float, np.ndarray]:
    """Find a destination-to-source Euclidean transform without OpenCV.

    This follows OpenCV's ECC preprocessing, Jacobian, illumination model,
    update, and convergence rules. Large pages are optimized at a bounded
    working resolution; the full-resolution phase estimate remains the
    initializer and translations are scaled back into page coordinates.
    """
    template_float = np.asarray(template, dtype=np.float32)
    input_float = np.asarray(input_image, dtype=np.float32)
    if (
        template_float.ndim != 2
        or input_float.ndim != 2
        or template_float.shape != input_float.shape
        or not template_float.size
    ):
        raise ValueError("ECC inputs must be non-empty, same-size 2D arrays")
    matrix = np.asarray(initial_matrix, dtype=np.float32)
    if matrix.shape != (2, 3):
        raise ValueError("Euclidean ECC requires a 2x3 initial matrix")
    if max_iterations < 1 or epsilon < 0:
        raise ValueError("ECC iteration count and epsilon must be non-negative")
    if not (
        np.isfinite(template_float).all()
        and np.isfinite(input_float).all()
        and np.isfinite(matrix).all()
    ):
        raise EccConvergenceError("ECC inputs must be finite")

    source_short_side = min(template_float.shape)
    template_working, input_working, scale = _resize_ecc_pair(
        template_float,
        input_float,
        EccSettings.MAX_WORKING_SHORT_SIDE_PX,
    )
    score, matrix = _find_transform_ecc_euclidean_core(
        template_working,
        input_working,
        _scaled_ecc_matrix(matrix, scale),
        max_iterations,
        epsilon,
    )
    if scale < 1.0:
        matrix[:, 2] /= np.float32(scale)

    rotation = abs(atan2(float(matrix[1, 0]), float(matrix[0, 0])))
    should_refine = (
        source_short_side >= EccSettings.REFINEMENT_MIN_SOURCE_SHORT_SIDE_PX
        and score >= 0.50
        and rotation >= EccSettings.REFINEMENT_MIN_ROTATION_RADIANS
    )
    if should_refine:
        refinement_template, refinement_input, refinement_scale = _resize_ecc_pair(
            template_float,
            input_float,
            EccSettings.REFINEMENT_SHORT_SIDE_PX,
        )
        try:
            refined_score, refined_matrix = _find_transform_ecc_euclidean_core(
                refinement_template,
                refinement_input,
                _scaled_ecc_matrix(matrix, refinement_scale),
                EccSettings.REFINEMENT_ITERATIONS,
                epsilon,
            )
        except (EccConvergenceError, MemoryError, np.linalg.LinAlgError):
            pass
        else:
            if refinement_scale < 1.0:
                refined_matrix[:, 2] /= np.float32(refinement_scale)
            score = refined_score
            matrix = _polish_ecc_translation(
                refinement_template,
                refinement_input,
                refined_matrix,
                refinement_scale,
            )
    return score, matrix


def _coarse_ecc_score(
    old_gray: np.ndarray,
    new_gray: np.ndarray,
    shift: tuple[float, float],
) -> float | None:
    """Return a cheap Euclidean ECC score, or ``None`` when it is inconclusive."""
    matrix = np.float32([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]])
    try:
        score, _ = _find_transform_ecc_euclidean(
            new_gray.astype(np.float32) / 255.0,
            old_gray.astype(np.float32) / 255.0,
            matrix,
            AlignmentSettings.FAST_REJECT_ECC_MAX_ITERATIONS,
            AlignmentSettings.FAST_REJECT_ECC_EPSILON,
        )
    except (EccConvergenceError, MemoryError, np.linalg.LinAlgError):
        return None
    return float(score) if np.isfinite(score) else None


def _should_fast_reject_ecc(old_gray: np.ndarray, new_gray: np.ndarray, ink_threshold: int) -> bool:
    """Return true only when three coarse signals strongly reject alignment.

    This gate never supplies a transform.  When it is inconclusive, the
    existing full-resolution phase/ECC sequence remains exactly unchanged.
    """
    coarse_old = _coarse_gray(old_gray)
    coarse_new = _coarse_gray(new_gray)
    try:
        shift, phase_score = _phase_correlate(
            coarse_new.astype(np.float32), coarse_old.astype(np.float32)
        )
    except (MemoryError, TypeError, ValueError):
        return False
    max_shift = max(coarse_new.shape) * AlignmentSettings.FAST_REJECT_MAX_SHIFT_FRACTION
    phase_is_plausible = (
        np.isfinite(phase_score)
        and phase_score >= AlignmentSettings.FAST_REJECT_MIN_PHASE_SCORE
        and np.hypot(*shift) <= max_shift
    )
    if phase_is_plausible:
        return False
    if _coarse_ink_iou(coarse_old, coarse_new, shift, ink_threshold) >= AlignmentSettings.FAST_REJECT_MIN_INK_IOU:
        return False
    coarse_score = _coarse_ecc_score(coarse_old, coarse_new, shift)
    return coarse_score is not None and coarse_score < AlignmentSettings.FAST_REJECT_MAX_ECC_SCORE


def _align_old_to_new(
    old_bgr: np.ndarray,
    new_bgr: np.ndarray,
    *,
    ink_threshold: int = DEFAULT_INK_THRESHOLD,
) -> tuple[np.ndarray, AlignmentMetadata]:
    target_h, target_w = new_bgr.shape[:2]
    old_h, old_w = old_bgr.shape[:2]
    resized = _resize_bgr(old_bgr, target_w, target_h)
    base = AlignmentMetadata("resize", True, original_old_size=(old_w, old_h), target_size=(target_w, target_h), moved=(old_w, old_h) != (target_w, target_h))
    old_gray = _bgr_to_gray(resized)
    new_gray = _bgr_to_gray(new_bgr)
    # ECC is deterministic and particularly effective for scanned/printed pages.
    if old_gray.std() < 1.0 or new_gray.std() < 1.0:
        return resized, AlignmentMetadata(**{**base.__dict__, "message": "blank or near-uniform page; resize only"})
    if _should_fast_reject_ecc(old_gray, new_gray, ink_threshold):
        return resized, AlignmentMetadata(
            **{
                **base.__dict__,
                "message": "fast-reject: coarse phase, ink overlap, and ECC indicate unrelated pages; resize only",
            }
        )
    # ECC only converges within a fairly small basin.  Phase correlation gives
    # a cheap translation estimate first, which makes ordinary scanner/page
    # offsets reliable instead of falling straight back to resize-only mode.
    try:
        shift, phase_score = _phase_correlate(
            new_gray.astype(np.float32), old_gray.astype(np.float32)
        )
        matrix = np.float32([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]])
    except (MemoryError, TypeError, ValueError):
        shift, phase_score = (0.0, 0.0), 0.0
        matrix = np.eye(2, 3, dtype=np.float32)
    max_shift = max(target_h, target_w) * 0.35
    phase_is_plausible = np.isfinite(phase_score) and phase_score >= 0.20 and np.hypot(*shift) <= max_shift
    try:
        score, matrix = _find_transform_ecc_euclidean(
            new_gray.astype(np.float32) / 255.0,
            old_gray.astype(np.float32) / 255.0,
            matrix,
            60,
            1e-6,
        )
        if not np.isfinite(score) or score < 0.50:
            return resized, AlignmentMetadata(**{**base.__dict__, "message": f"ECC score {score:.3f} was too low; resize only"})
        aligned = _warp_bgr_affine(
            resized,
            matrix,
            target_w,
            target_h,
            ink_threshold=ink_threshold,
        )
        moved = not np.allclose(matrix, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), atol=0.15)
        return aligned, AlignmentMetadata("ecc-euclidean", True, float(score), matrix, (old_w, old_h), (target_w, target_h), moved)
    except (EccConvergenceError, MemoryError, np.linalg.LinAlgError) as exc:
        if phase_is_plausible:
            aligned = _warp_bgr_affine(
                resized,
                matrix,
                target_w,
                target_h,
                ink_threshold=ink_threshold,
            )
            moved = not np.allclose(matrix, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), atol=0.15)
            return aligned, AlignmentMetadata(
                "phase-correlation",
                True,
                float(phase_score),
                matrix,
                (old_w, old_h),
                (target_w, target_h),
                moved,
                "ECC refinement unavailable; using phase-correlation alignment",
            )
        return resized, AlignmentMetadata(**{**base.__dict__, "message": f"ECC alignment unavailable: {str(exc).splitlines()[0]}"})


def _distance_transform_l2_mask3(source: np.ndarray) -> np.ndarray:
    """Match ``cv2.distanceTransform(source, cv2.DIST_L2, 3)`` with NumPy.

    The 3x3 L2 mode is an approximate two-pass chamfer transform rather than
    a mathematically exact Euclidean distance.  The passes retain OpenCV's
    axial and diagonal weights, zero-pixel sources, and infinite border.
    """
    nonzero = np.ascontiguousarray(source != 0)
    height, width = nonzero.shape
    max_distance = np.finfo(np.float32).max
    axial_cost = DistanceTransformSettings.AXIAL_COST
    diagonal_cost = DistanceTransformSettings.DIAGONAL_COST
    distances = np.where(nonzero, max_distance, 0).astype(np.float32)
    column_costs = np.arange(width, dtype=np.float32) * axial_cost

    # OpenCV's forward pass considers north-west, north, north-east, and west.
    for row_index in range(height):
        if row_index:
            above = distances[row_index - 1]
            candidates = above + axial_cost
            np.minimum(
                candidates[1:],
                above[:-1] + diagonal_cost,
                out=candidates[1:],
            )
            np.minimum(
                candidates[:-1],
                above[1:] + diagonal_cost,
                out=candidates[:-1],
            )
        else:
            candidates = np.full(width, max_distance, dtype=np.float32)
        candidates[~nonzero[row_index]] = 0
        candidates -= column_costs
        np.minimum.accumulate(candidates, out=candidates)
        candidates += column_costs
        distances[row_index] = candidates

    # OpenCV's backward pass considers south-east, south, south-west, and east.
    for row_index in range(height - 1, -1, -1):
        candidates = distances[row_index]
        if row_index + 1 < height:
            below = distances[row_index + 1]
            np.minimum(candidates, below + axial_cost, out=candidates)
            np.minimum(
                candidates[1:],
                below[:-1] + diagonal_cost,
                out=candidates[1:],
            )
            np.minimum(
                candidates[:-1],
                below[1:] + diagonal_cost,
                out=candidates[:-1],
            )
        right_to_left = candidates[::-1]
        right_to_left -= column_costs
        np.minimum.accumulate(right_to_left, out=right_to_left)
        right_to_left += column_costs

    return distances


@lru_cache(maxsize=DifferenceMaskSettings.MAX_LOCAL_TOLERANCE_PX + 1)
def _tolerance_neighborhood(
    tolerance_px: int,
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Return the exact local chamfer ball and its per-row spans.

    The cached transform is deliberately produced by the same oracle used by
    the fallback.  Thus this optimization only changes how the reference mask
    is queried; it does not introduce a second approximation of OpenCV's
    mask3 distance transform.
    """
    radius = int(np.ceil(tolerance_px / float(DistanceTransformSettings.AXIAL_COST))) + 2
    size = radius * 2 + 1
    center = radius
    probe = np.ones((size, size), dtype=np.uint8)
    probe[center, center] = 0
    distances = _distance_transform_l2_mask3(probe)
    offsets: list[tuple[int, int]] = []
    spans: list[tuple[int, int]] = []
    for dy in range(-radius, radius + 1):
        valid_dx = np.flatnonzero(distances[center + dy] <= tolerance_px)
        if valid_dx.size:
            dx = valid_dx - center
            offsets.extend((dy, int(value)) for value in dx)
            spans.append((dy, int(np.max(np.abs(dx)))))
    return tuple(offsets), tuple(spans)


def _difference_candidates(source_ink: np.ndarray, reference_ink: np.ndarray) -> np.ndarray:
    """Return source ink that does not overlap reference ink, without temporaries."""
    candidates = np.empty(source_ink.shape, dtype=bool)
    np.equal(reference_ink, 0, out=candidates)
    np.greater(source_ink, 0, out=candidates, where=candidates)
    return candidates


def _difference_mask_local(
    source_ink: np.ndarray,
    reference_ink: np.ndarray,
    tolerance_px: float,
) -> np.ndarray | None:
    """Compute small finite-radius tolerance masks without a full-page DT."""
    if not np.isfinite(tolerance_px) or tolerance_px < 0:
        return None
    rounded_tolerance = int(tolerance_px)
    if (
        tolerance_px != rounded_tolerance
        or rounded_tolerance > DifferenceMaskSettings.MAX_LOCAL_TOLERANCE_PX
    ):
        return None
    offsets, spans = _tolerance_neighborhood(rounded_tolerance)
    candidate_mask = _difference_candidates(source_ink, reference_ink)
    candidate_count = int(np.count_nonzero(candidate_mask))
    if not candidate_count:
        return np.zeros(source_ink.shape, dtype=np.uint8)
    page_size = source_ink.size
    offset_count = len(offsets)
    work = candidate_count * offset_count
    if work <= DifferenceMaskSettings.DIRECT_MAX_WORK_FRACTION * page_size or (
        offset_count <= DifferenceMaskSettings.DIRECT_SMALL_OFFSET_COUNT
        and work <= DifferenceMaskSettings.DIRECT_SMALL_OFFSET_MAX_WORK_FRACTION * page_size
    ):
        radius = int(np.ceil(rounded_tolerance / float(DistanceTransformSettings.AXIAL_COST))) + 2
        padded_reference = np.pad(reference_ink != 0, radius, mode="constant")
        ys, xs = np.nonzero(candidate_mask)
        keep = np.ones(candidate_count, dtype=bool)
        for dy, dx in offsets:
            keep &= ~padded_reference[ys + radius + dy, xs + radius + dx]
    elif candidate_count * len(spans) <= DifferenceMaskSettings.SPAN_MAX_WORK_FRACTION * page_size:
        radius = int(np.ceil(rounded_tolerance / float(DistanceTransformSettings.AXIAL_COST))) + 2
        height, width = reference_ink.shape
        prefix = np.zeros(
            (height + 2 * radius, width + 1),
            dtype=np.uint32,
        )
        np.cumsum(
            reference_ink != 0,
            axis=1,
            dtype=np.uint32,
            out=prefix[radius : radius + height, 1:],
        )
        ys, xs = np.nonzero(candidate_mask)
        keep = np.ones(candidate_count, dtype=bool)
        for dy, half_width in spans:
            rows = ys + radius + dy
            left = np.maximum(xs - half_width, 0)
            right = np.minimum(xs + half_width + 1, width)
            keep &= (prefix[rows, right] - prefix[rows, left]) == 0
    else:
        return None
    result = np.zeros(source_ink.shape, dtype=np.uint8)
    result[ys[keep], xs[keep]] = 255
    return result


def _difference_mask(source_ink: np.ndarray, reference_ink: np.ndarray, tolerance_px: float) -> np.ndarray:
    if tolerance_px == 0:
        result = np.zeros(source_ink.shape, dtype=np.uint8)
        result[_difference_candidates(source_ink, reference_ink)] = 255
        return result
    accelerated = _difference_mask_local(source_ink, reference_ink, tolerance_px)
    if accelerated is not None:
        return accelerated
    # The transform measures each source pixel's distance to reference ink.
    # Normalize the reference to the binary representation expected by the
    # distance transform; callers may use any nonzero value for ink.
    reference_background = reference_ink == 0
    distance = _distance_transform_l2_mask3(reference_background)
    candidates = _difference_candidates(source_ink, reference_ink)
    np.greater(distance, tolerance_px, out=candidates, where=candidates)
    result = np.zeros(source_ink.shape, dtype=np.uint8)
    result[candidates] = 255
    return result


def _dilate_binary_mask(
    mask: np.ndarray,
    kernel_size: int,
    *,
    foreground_runs: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Dilate a zero/255 mask with OpenCV's rectangular-kernel semantics.

    Sparse masks use Qt's native raster painter to fill the kernel rectangle at
    each ink pixel. Drawing-like masks reuse their horizontal foreground runs
    and paint one rectangle per merged run; dense masks use a summed-area table
    instead. All paths preserve OpenCV's default center anchor for even kernels:
    a 20-pixel kernel examines offsets -10..9 from each output pixel, so a
    source pixel expands 9 pixels up/left and 10 pixels down/right.
    Out-of-bounds pixels remain background.
    """
    if kernel_size == 1:
        return mask.copy()
    if foreground_runs is None:
        foreground_count = int(np.count_nonzero(mask))
    else:
        foreground_count = int(np.sum(foreground_runs[2] - foreground_runs[1] + 1))
    if foreground_count <= mask.size * MaskDilationSettings.QT_MAX_FOREGROUND_FRACTION:
        try:
            return _dilate_sparse_binary_mask_qt(mask, kernel_size)
        except (RuntimeError, TypeError, ValueError):
            pass
    if (
        foreground_runs is not None
        and foreground_runs[0].size <= mask.size * MaskDilationSettings.QT_MAX_RUN_FRACTION
    ):
        try:
            return _dilate_binary_runs_qt(mask.shape, kernel_size, *foreground_runs)
        except (RuntimeError, TypeError, ValueError):
            pass
    return _dilate_binary_mask_integral(mask, kernel_size)


def _dilate_sparse_binary_mask_qt(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Draw every sparse binary-mask dilation rectangle in Qt's raster engine."""
    height, width = mask.shape
    anchor = kernel_size // 2
    leading_extent = kernel_size - anchor - 1
    ys, xs = np.nonzero(mask > 0)
    output_image = QImage(width, height, QImage.Format.Format_Grayscale8)
    output_image.fill(0)
    rectangles = [
        QRect(int(x - leading_extent), int(y - leading_extent), kernel_size, kernel_size)
        for y, x in zip(ys, xs)
    ]
    painter = QPainter(output_image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    painter.setPen(QPen(Qt.PenStyle.NoPen))
    painter.setBrush(QBrush(QColor(255, 255, 255)))
    painter.drawRects(rectangles)
    painter.end()

    bits = output_image.bits()
    bits.setsize(output_image.sizeInBytes())
    return np.frombuffer(bits, dtype=np.uint8).reshape(height, output_image.bytesPerLine())[:, :width].copy()


def _dilate_binary_runs_qt(
    shape: tuple[int, int],
    kernel_size: int,
    rows: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """Dilate horizontal foreground runs with Qt raster rectangles.

    Every source run expands into one rectangle instead of one rectangle per
    foreground pixel. Overlapping expanded runs on the same row are merged
    before painting, which keeps drawing-like masks on Qt's fast raster path.
    """
    height, width = shape
    if not rows.size:
        return np.zeros(shape, dtype=np.uint8)

    anchor = kernel_size // 2
    leading_extent = kernel_size - anchor - 1
    expanded_starts = np.maximum(starts - leading_extent, 0)
    expanded_ends = np.minimum(ends + anchor, width - 1)

    group_starts = np.empty(rows.size, dtype=bool)
    group_starts[0] = True
    group_starts[1:] = (rows[1:] != rows[:-1]) | (
        expanded_starts[1:] > expanded_ends[:-1] + 1
    )
    group_indices = np.flatnonzero(group_starts)
    merged_rows = rows[group_indices]
    merged_starts = expanded_starts[group_indices]
    merged_ends = np.maximum.reduceat(expanded_ends, group_indices)
    rectangles = [
        QRect(
            int(start),
            int(row - leading_extent),
            int(end - start + 1),
            kernel_size,
        )
        for row, start, end in zip(merged_rows, merged_starts, merged_ends)
    ]

    output_image = QImage(width, height, QImage.Format.Format_Grayscale8)
    output_image.fill(0)
    painter = QPainter(output_image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    painter.setPen(QPen(Qt.PenStyle.NoPen))
    painter.setBrush(QBrush(QColor(255, 255, 255)))
    painter.drawRects(rectangles)
    painter.end()

    bits = output_image.bits()
    bits.setsize(output_image.sizeInBytes())
    return np.frombuffer(bits, dtype=np.uint8).reshape(height, output_image.bytesPerLine())[:, :width].copy()


def _dilate_binary_mask_integral(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Dilate a dense binary mask through a NumPy summed-area table."""
    anchor = kernel_size // 2
    padded = np.pad(
        mask > 0,
        ((anchor, kernel_size - anchor - 1), (anchor, kernel_size - anchor - 1)),
        constant_values=False,
    )
    integral = np.zeros((padded.shape[0] + 1, padded.shape[1] + 1), dtype=np.uint32)
    integral[1:, 1:] = np.cumsum(
        np.cumsum(padded, axis=0, dtype=np.uint32),
        axis=1,
        dtype=np.uint32,
    )
    window_sums = (
        integral[kernel_size:, kernel_size:]
        - integral[:-kernel_size, kernel_size:]
        - integral[kernel_size:, :-kernel_size]
        + integral[:-kernel_size, :-kernel_size]
    )
    return np.where(window_sums > 0, 255, 0).astype(np.uint8)


def _foreground_run_table(foreground: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return row/start/end arrays for all horizontal foreground runs.

    Finding every row transition in one NumPy operation avoids thousands of
    small ``flatnonzero`` calls on high-DPI pages.  Each row contributes an
    even number of transitions, so the flattened transition indices pair into
    inclusive run bounds without storing every foreground pixel coordinate.
    """
    height, width = foreground.shape
    if not height or not width:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty, empty
    transitions = np.empty((height, width + 1), dtype=bool)
    transitions[:, 0] = foreground[:, 0]
    if width > 1:
        np.not_equal(
            foreground[:, 1:],
            foreground[:, :-1],
            out=transitions[:, 1:width],
        )
    transitions[:, width] = foreground[:, -1]
    transition_indices = np.flatnonzero(transitions)
    if not transition_indices.size:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty, empty
    rows, columns = np.divmod(transition_indices, width + 1)
    return rows[::2], columns[::2], columns[1::2] - 1


def _connected_run_labels_8(
    rows: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    height: int,
) -> tuple[int, np.ndarray]:
    """Assign compact 8-connected component labels to horizontal runs."""
    run_count = rows.size
    if not run_count:
        return 1, np.empty(0, dtype=np.intp)

    row_offsets = np.empty(height + 1, dtype=np.intp)
    row_offsets[0] = 0
    np.cumsum(np.bincount(rows, minlength=height), out=row_offsets[1:])
    parents = np.arange(run_count, dtype=np.intp)

    def find(label: int) -> int:
        root = label
        while parents[root] != root:
            root = int(parents[root])
        while parents[label] != label:
            parent = int(parents[label])
            parents[label] = root
            label = parent
        return root

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root == second_root:
            return
        # Runs are row-major; retaining the earliest root keeps the trees
        # shallow for drawing-like masks and makes labels deterministic.
        if first_root < second_root:
            parents[second_root] = first_root
        else:
            parents[first_root] = second_root

    for row in range(1, height):
        previous = int(row_offsets[row - 1])
        previous_stop = int(row_offsets[row])
        current = previous_stop
        current_stop = int(row_offsets[row + 1])
        while previous < previous_stop and current < current_stop:
            if ends[previous] < starts[current] - 1:
                previous += 1
            elif ends[current] < starts[previous] - 1:
                current += 1
            else:
                union(previous, current)
                if ends[previous] < ends[current]:
                    previous += 1
                elif ends[current] < ends[previous]:
                    current += 1
                else:
                    previous += 1
                    current += 1

    roots = np.fromiter(
        (find(index) for index in range(run_count)),
        dtype=np.intp,
        count=run_count,
    )
    component_roots = np.unique(roots)
    run_labels = np.searchsorted(component_roots, roots) + 1
    return int(component_roots.size + 1), run_labels


def _connected_components_8(mask: np.ndarray) -> tuple[int, np.ndarray]:
    """Return a dense 8-connected label image for callers that need one."""
    foreground = np.ascontiguousarray(mask > 0)
    height, width = foreground.shape
    rows, starts, ends = _foreground_run_table(foreground)
    count, run_labels = _connected_run_labels_8(rows, starts, ends, height)
    labels = np.zeros((height, width), dtype=np.int32)
    for row, start, end, label in zip(rows, starts, ends, run_labels):
        labels[row, start : end + 1] = label
    return count, labels


def _regions(mask: np.ndarray, kind: str, minimum_area: int, merge_distance: int) -> list[DifferenceRegion]:
    """Group nearby ink changes into reviewable regions without inflating area.

    A changed dimension label can have several disconnected character strokes.
    Connected-components on the raw mask would present every character as a
    separate sidebar row.  Components are therefore found on a lightly dilated
    copy, while the returned bounds and area still come from the original mask.
    Regions are returned in stable top-to-bottom, left-to-right order instead
    of depending on a particular labeling backend's internal ID order.
    """
    raw_foreground = np.ascontiguousarray(mask > 0)
    occupied_rows = np.flatnonzero(np.any(raw_foreground, axis=1))
    if not occupied_rows.size:
        return []
    occupied_columns = np.flatnonzero(np.any(raw_foreground, axis=0))
    top, bottom = int(occupied_rows[0]), int(occupied_rows[-1]) + 1
    left, right = int(occupied_columns[0]), int(occupied_columns[-1]) + 1
    raw_foreground = np.ascontiguousarray(raw_foreground[top:bottom, left:right])
    raw_rows, raw_starts, raw_ends = _foreground_run_table(raw_foreground)

    if merge_distance:
        grouped = np.ascontiguousarray(
            _dilate_binary_mask(
                raw_foreground,
                merge_distance,
                foreground_runs=(raw_rows, raw_starts, raw_ends),
            )
            > 0
        )
        grouped_rows, grouped_starts, grouped_ends = _foreground_run_table(grouped)
    else:
        grouped = raw_foreground
        grouped_rows, grouped_starts, grouped_ends = raw_rows, raw_starts, raw_ends

    height, width = grouped.shape
    count, grouped_labels = _connected_run_labels_8(
        grouped_rows,
        grouped_starts,
        grouped_ends,
        height,
    )

    if merge_distance:
        # A raw run is contained in exactly one dilated run.  Flattening each
        # row into a disjoint numeric range lets one vectorized search map all
        # raw runs back to their grouped component labels.
        stride = width + 1
        grouped_end_keys = grouped_rows * stride + grouped_ends
        raw_start_keys = raw_rows * stride + raw_starts
        grouped_indices = np.searchsorted(grouped_end_keys, raw_start_keys)
        raw_labels = grouped_labels[grouped_indices]
    else:
        raw_rows, raw_starts, raw_ends = grouped_rows, grouped_starts, grouped_ends
        raw_labels = grouped_labels

    run_lengths = raw_ends - raw_starts + 1
    areas = np.bincount(raw_labels, weights=run_lengths, minlength=count).astype(np.intp)
    eligible_labels = np.flatnonzero(areas >= minimum_area)
    eligible_labels = eligible_labels[eligible_labels > 0]
    if not eligible_labels.size:
        return []

    min_x = np.full(count, width, dtype=np.intp)
    max_x = np.full(count, -1, dtype=np.intp)
    min_y = np.full(count, height, dtype=np.intp)
    max_y = np.full(count, -1, dtype=np.intp)
    np.minimum.at(min_x, raw_labels, raw_starts)
    np.maximum.at(max_x, raw_labels, raw_ends)
    np.minimum.at(min_y, raw_labels, raw_rows)
    np.maximum.at(max_y, raw_labels, raw_rows)

    regions = [
        DifferenceRegion(
            (
                int(min_x[label] + left),
                int(min_y[label] + top),
                int(max_x[label] - min_x[label] + 1),
                int(max_y[label] - min_y[label] + 1),
            ),
            int(areas[label]),
            kind,
        )
        for label in eligible_labels
    ]
    return sorted(
        regions,
        key=lambda region: (
            region.bbox[1],
            region.bbox[0],
            region.bbox[3],
            region.bbox[2],
            region.area,
        ),
    )


def _layer(mask: np.ndarray, bgr_color: tuple[int, int, int], alpha: int) -> np.ndarray:
    layer = np.zeros((*mask.shape, 4), dtype=np.uint8)
    layer[mask > 0, :3] = bgr_color
    layer[mask > 0, 3] = alpha
    return layer


def compare_page_images(
    old_image: np.ndarray,
    new_image: np.ndarray,
    *,
    tolerance_px: float = 2.0,
    ink_threshold: int = DEFAULT_INK_THRESHOLD,
    minimum_region_area: int = 4,
    region_merge_distance: int = 20,
    overlay_alpha: int = 180,
    progress: ProgressCallback | None = None,
) -> ComparisonResult:
    """Compare two page rasters, returning transparent blue-add/red-remove layers.

    Images may have different sizes and can be gray, BGR, or BGRA.  The new
    image defines the result canvas; the old image is resized then aligned.
    """
    if tolerance_px < 0 or minimum_region_area < 1 or region_merge_distance < 0 or not 0 <= ink_threshold <= 255 or not 0 <= overlay_alpha <= 255:
        raise ValueError("invalid comparison settings")
    _progress(progress, "normalizing images", 0.05)
    old_bgr, new_bgr = _as_bgr(old_image), _as_bgr(new_image)
    _progress(progress, "aligning pages", 0.20)
    old_aligned, alignment = _align_old_to_new(
        old_bgr,
        new_bgr,
        ink_threshold=ink_threshold,
    )
    _progress(progress, "finding ink", 0.55)
    old_ink, new_ink = _ink_mask(old_aligned, ink_threshold), _ink_mask(new_bgr, ink_threshold)
    added_mask = _difference_mask(new_ink, old_ink, tolerance_px)
    removed_mask = _difference_mask(old_ink, new_ink, tolerance_px)
    _progress(progress, "extracting regions", 0.75)
    added_regions = _regions(added_mask, "added", minimum_region_area, region_merge_distance)
    removed_regions = _regions(removed_mask, "removed", minimum_region_area, region_merge_distance)
    # Layers are BGRA here, matching OpenCV/QImage byte order on Windows.
    # Added ink is bright blue; removed ink is bright red.
    added_layer = _layer(added_mask, DifferenceColors.ADDITION_BGR, overlay_alpha)
    removed_layer = _layer(removed_mask, DifferenceColors.REMOVAL_BGR, overlay_alpha)
    old_bgra, new_bgra = _bgr_to_bgra(old_aligned), _bgr_to_bgra(new_bgr)
    added_pixels, removed_pixels = int(np.count_nonzero(added_mask)), int(np.count_nonzero(removed_mask))
    _progress(progress, "comparison complete", 1.0)
    return ComparisonResult(np.ascontiguousarray(old_bgra), np.ascontiguousarray(new_bgra), added_layer, removed_layer, added_mask, removed_mask, added_regions, removed_regions, alignment, new_bgr.shape[1], new_bgr.shape[0], added_pixels, removed_pixels, int(np.count_nonzero((added_mask > 0) | (removed_mask > 0))))


def compare_pdf_pages(
    old_pdf: str | Path,
    new_pdf: str | Path,
    page_index: int = 0,
    *,
    dpi: float = 144.0,
    progress: ProgressCallback | None = None,
    **comparison_options: object,
) -> ComparisonResult:
    """Render matching page indexes from two PDFs and compare them."""
    _progress(progress, "rendering old PDF", 0.0)
    old = render_pdf_page(old_pdf, page_index, dpi=dpi)
    _progress(progress, "rendering new PDF", 0.35)
    new = render_pdf_page(new_pdf, page_index, dpi=dpi)

    def nested(stage: str, fraction: float) -> None:
        _progress(progress, stage, 0.5 + fraction * 0.5)

    return compare_page_images(old.bgra, new.bgra, progress=nested, **comparison_options)


