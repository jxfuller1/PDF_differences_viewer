"""PDF rendering and pixel comparison for native image-based review.

The returned BGRA arrays are deliberately suitable for direct conversion to a
``QImage.Format_ARGB32`` by a PyQt application.  The comparison method uses
the distance between *ink* pixels, rather than a brittle exact RGB comparison:
small rasterisation shifts therefore do not become differences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, degrees, hypot
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
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


class AlignmentSettings:
    """Conservative settings for deciding whether ECC is worth attempting."""

    FAST_REJECT_SHORT_SIDE_PX = 384
    FAST_REJECT_MIN_PHASE_SCORE = 0.20
    FAST_REJECT_MIN_INK_IOU = 0.10
    FAST_REJECT_MAX_ECC_SCORE = 0.20
    FAST_REJECT_ECC_MAX_ITERATIONS = 20
    FAST_REJECT_ECC_EPSILON = 1e-4
    FAST_REJECT_MAX_SHIFT_FRACTION = 0.35


class MaskDilationSettings:
    """Tuning values for grouped sparse change masks."""

    QT_MAX_FOREGROUND_FRACTION = 0.005


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
        rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
        if pixmap.n == 1:
            bgr = _gray_to_bgr(rgb[:, :, 0])
        else:
            bgr = _rgb_to_bgr(rgb[:, :, :3])
    bgra = _bgr_to_bgra(bgr)
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
    """Convert BGR to luma with the standard BT.601 integer coefficients."""
    channels = bgr.astype(np.uint32, copy=False)
    return (
        (channels[:, :, 2] * 299 + channels[:, :, 1] * 587 + channels[:, :, 0] * 114 + 500)
        // 1000
    ).astype(np.uint8)


def _resize_bgr(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize a BGR image with Pillow, preserving the requested exact size."""
    if image.shape[:2] == (target_height, target_width):
        return image.copy()
    source_height, source_width = image.shape[:2]
    resample = (
        Image.Resampling.BOX
        if source_width * source_height > target_width * target_height
        else Image.Resampling.BICUBIC
    )
    rgb = Image.fromarray(np.ascontiguousarray(image[:, :, ::-1]))
    resized_rgb = rgb.resize((target_width, target_height), resample=resample)
    return np.ascontiguousarray(np.asarray(resized_rgb)[:, :, ::-1])


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

    ``findTransformECC`` and ``phaseCorrelate`` provide matrices that map each
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


def _ink_mask(bgr: np.ndarray, ink_threshold: int) -> np.ndarray:
    """Mark non-paper pixels.  White/near-white pages remain safely empty."""
    gray = _bgr_to_gray(bgr)
    return np.where(gray < ink_threshold, 255, 0).astype(np.uint8)


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


def _coarse_ink_iou(
    old_gray: np.ndarray,
    new_gray: np.ndarray,
    shift: tuple[float, float],
    ink_threshold: int,
) -> float:
    """Return raw ink IoU after phase's destination-to-source translation."""
    old_ink = (old_gray < ink_threshold).astype(np.uint8)
    new_ink = new_gray < ink_threshold
    matrix = np.float32([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]])
    old_aligned = cv2.warpAffine(
        old_ink,
        matrix,
        (new_gray.shape[1], new_gray.shape[0]),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    union = np.count_nonzero((old_aligned > 0) | new_ink)
    if not union:
        return 1.0
    intersection = np.count_nonzero((old_aligned > 0) & new_ink)
    return float(intersection / union)


def _coarse_ecc_score(
    old_gray: np.ndarray,
    new_gray: np.ndarray,
    shift: tuple[float, float],
) -> float | None:
    """Return a cheap Euclidean ECC score, or ``None`` when it is inconclusive."""
    matrix = np.float32([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]])
    try:
        score, _ = cv2.findTransformECC(
            new_gray.astype(np.float32) / 255.0,
            old_gray.astype(np.float32) / 255.0,
            matrix,
            cv2.MOTION_EUCLIDEAN,
            (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                AlignmentSettings.FAST_REJECT_ECC_MAX_ITERATIONS,
                AlignmentSettings.FAST_REJECT_ECC_EPSILON,
            ),
        )
    except cv2.error:
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
        shift, phase_score = cv2.phaseCorrelate(
            coarse_new.astype(np.float32), coarse_old.astype(np.float32)
        )
    except cv2.error:
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
        shift, phase_score = cv2.phaseCorrelate(
            new_gray.astype(np.float32), old_gray.astype(np.float32)
        )
        matrix = np.float32([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]])
    except cv2.error:
        shift, phase_score = (0.0, 0.0), 0.0
        matrix = np.eye(2, 3, dtype=np.float32)
    max_shift = max(target_h, target_w) * 0.35
    phase_is_plausible = np.isfinite(phase_score) and phase_score >= 0.20 and np.hypot(*shift) <= max_shift
    try:
        score, matrix = cv2.findTransformECC(
            new_gray.astype(np.float32) / 255.0,
            old_gray.astype(np.float32) / 255.0,
            matrix,
            cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-6),
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
    except cv2.error as exc:
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


def _difference_mask(source_ink: np.ndarray, reference_ink: np.ndarray, tolerance_px: float) -> np.ndarray:
    # distanceTransform measures each source pixel's distance to reference ink.
    distance = cv2.distanceTransform(cv2.bitwise_not(reference_ink), cv2.DIST_L2, 3)
    return np.where((source_ink > 0) & (distance > tolerance_px), 255, 0).astype(np.uint8)


def _dilate_binary_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Dilate a zero/255 mask with OpenCV's rectangular-kernel semantics.

    Sparse masks use Qt's native raster painter to fill the kernel rectangle at
    each ink pixel. Dense masks use a summed-area table instead. Both preserve
    OpenCV's default center anchor for even kernels: a 20-pixel kernel examines
    offsets -10..9 from each output pixel, so a source pixel expands 9 pixels
    up/left and 10 pixels down/right. Out-of-bounds pixels remain background.
    """
    if kernel_size == 1:
        return mask.copy()
    foreground_count = int(np.count_nonzero(mask))
    if foreground_count <= mask.size * MaskDilationSettings.QT_MAX_FOREGROUND_FRACTION:
        try:
            return _dilate_sparse_binary_mask_qt(mask, kernel_size)
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


def _regions(mask: np.ndarray, kind: str, minimum_area: int, merge_distance: int) -> list[DifferenceRegion]:
    """Group nearby ink changes into reviewable regions without inflating area.

    A changed dimension label can have several disconnected character strokes.
    Connected-components on the raw mask would present every character as a
    separate sidebar row.  Components are therefore found on a lightly dilated
    copy, while the returned bounds and area still come from the original mask.
    """
    if not np.any(mask):
        return []
    if merge_distance:
        grouped = _dilate_binary_mask(mask, merge_distance)
    else:
        grouped = mask
    count, labels = cv2.connectedComponents(grouped, connectivity=8)
    # Work only with changed pixels once.  The former implementation compared
    # every page pixel against every component label, which becomes especially
    # expensive at high DPI when a page has many regions.
    ys, xs = np.nonzero(mask)
    pixel_labels = labels[ys, xs]
    areas = np.bincount(pixel_labels, minlength=count)
    eligible_labels = np.flatnonzero(areas >= minimum_area)
    eligible_labels = eligible_labels[eligible_labels > 0]
    if not eligible_labels.size:
        return []

    height, width = mask.shape
    min_x = np.full(count, width, dtype=np.intp)
    max_x = np.full(count, -1, dtype=np.intp)
    min_y = np.full(count, height, dtype=np.intp)
    max_y = np.full(count, -1, dtype=np.intp)
    np.minimum.at(min_x, pixel_labels, xs)
    np.maximum.at(max_x, pixel_labels, xs)
    np.minimum.at(min_y, pixel_labels, ys)
    np.maximum.at(max_y, pixel_labels, ys)

    return [
        DifferenceRegion(
            (
                int(min_x[label]),
                int(min_y[label]),
                int(max_x[label] - min_x[label] + 1),
                int(max_y[label] - min_y[label] + 1),
            ),
            int(areas[label]),
            kind,
        )
        for label in eligible_labels
    ]


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


