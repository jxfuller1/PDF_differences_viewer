"""PDF rendering and pixel comparison independent of any user-interface toolkit.

The returned BGRA arrays are deliberately suitable for direct conversion to a
``QImage.Format_ARGB32`` by a PyQt application.  The comparison method uses
the distance between *ink* pixels, rather than a brittle exact RGB comparison:
small rasterisation shifts therefore do not become differences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import pymupdf as fitz
import numpy as np


ProgressCallback = Callable[[str, float], None]
BBox = tuple[int, int, int, int]  # x, y, width, height


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
        pixmap = document.load_page(page_index).get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0), alpha=False)
        rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
        if pixmap.n == 1:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2BGR)
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    _progress(progress, "page rendered", 1.0)
    return RenderedPage(bgra=np.ascontiguousarray(bgra), page_index=page_index, width=bgra.shape[1], height=bgra.shape[0], dpi=dpi)


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty numpy array")
    if image.ndim == 2:
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("image must have 1, 3, or 4 channels")
    image = image.astype(np.uint8, copy=False)
    return image[:, :, :3].copy()


def _ink_mask(bgr: np.ndarray, ink_threshold: int) -> np.ndarray:
    """Mark non-paper pixels.  White/near-white pages remain safely empty."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return np.where(gray < ink_threshold, 255, 0).astype(np.uint8)


def _align_old_to_new(old_bgr: np.ndarray, new_bgr: np.ndarray) -> tuple[np.ndarray, AlignmentMetadata]:
    target_h, target_w = new_bgr.shape[:2]
    old_h, old_w = old_bgr.shape[:2]
    resized = cv2.resize(old_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA if old_w * old_h > target_w * target_h else cv2.INTER_CUBIC)
    base = AlignmentMetadata("resize", True, original_old_size=(old_w, old_h), target_size=(target_w, target_h), moved=(old_w, old_h) != (target_w, target_h))
    old_gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(new_bgr, cv2.COLOR_BGR2GRAY)
    # ECC is deterministic and particularly effective for scanned/printed pages.
    if old_gray.std() < 1.0 or new_gray.std() < 1.0:
        return resized, AlignmentMetadata(**{**base.__dict__, "message": "blank or near-uniform page; resize only"})
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
        aligned = cv2.warpAffine(resized, matrix, (target_w, target_h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        moved = not np.allclose(matrix, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32), atol=0.15)
        return aligned, AlignmentMetadata("ecc-euclidean", True, float(score), matrix, (old_w, old_h), (target_w, target_h), moved)
    except cv2.error as exc:
        if phase_is_plausible:
            aligned = cv2.warpAffine(
                resized,
                matrix,
                (target_w, target_h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
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
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (merge_distance, merge_distance)
        )
        grouped = cv2.dilate(mask, kernel, iterations=1)
    else:
        grouped = mask
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
    ink_threshold: int = 245,
    minimum_region_area: int = 4,
    region_merge_distance: int = 20,
    overlay_alpha: int = 180,
    progress: ProgressCallback | None = None,
) -> ComparisonResult:
    """Compare two page rasters, returning transparent red-add/blue-remove layers.

    Images may have different sizes and can be gray, BGR, or BGRA.  The new
    image defines the result canvas; the old image is resized then aligned.
    """
    if tolerance_px < 0 or minimum_region_area < 1 or region_merge_distance < 0 or not 0 <= ink_threshold <= 255 or not 0 <= overlay_alpha <= 255:
        raise ValueError("invalid comparison settings")
    _progress(progress, "normalizing images", 0.05)
    old_bgr, new_bgr = _as_bgr(old_image), _as_bgr(new_image)
    _progress(progress, "aligning pages", 0.20)
    old_aligned, alignment = _align_old_to_new(old_bgr, new_bgr)
    _progress(progress, "finding ink", 0.55)
    old_ink, new_ink = _ink_mask(old_aligned, ink_threshold), _ink_mask(new_bgr, ink_threshold)
    added_mask = _difference_mask(new_ink, old_ink, tolerance_px)
    removed_mask = _difference_mask(old_ink, new_ink, tolerance_px)
    _progress(progress, "extracting regions", 0.75)
    added_regions = _regions(added_mask, "added", minimum_region_area, region_merge_distance)
    removed_regions = _regions(removed_mask, "removed", minimum_region_area, region_merge_distance)
    # Red and blue are BGRA here, matching OpenCV/QImage byte order on Windows.
    added_layer, removed_layer = _layer(added_mask, (0, 0, 255), overlay_alpha), _layer(removed_mask, (255, 0, 0), overlay_alpha)
    old_bgra, new_bgra = cv2.cvtColor(old_aligned, cv2.COLOR_BGR2BGRA), cv2.cvtColor(new_bgr, cv2.COLOR_BGR2BGRA)
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
