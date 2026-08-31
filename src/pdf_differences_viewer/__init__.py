"""Native PDF visual-comparison primitives."""

from pathlib import Path
import sys

# Support both package execution and direct same-folder imports.
_PACKAGE_DIRECTORY = str(Path(__file__).resolve().parent)
if _PACKAGE_DIRECTORY not in sys.path:
    sys.path.insert(0, _PACKAGE_DIRECTORY)

import colors as _colors
import engine as _engine

# Keep package-qualified imports referring to the same modules as the flat
# imports above, rather than loading duplicate module objects.
sys.modules.setdefault(f"{__name__}.colors", _colors)
sys.modules.setdefault(f"{__name__}.engine", _engine)

from engine import ComparisonResult, compare_pdf_pages, compare_page_images, render_pdf_page

__all__ = ["ComparisonResult", "compare_pdf_pages", "compare_page_images", "render_pdf_page"]
