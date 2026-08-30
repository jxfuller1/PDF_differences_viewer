# PDF Differences Viewer

A native desktop application for comparing two PDF drawing revisions. It uses
PyMuPDF and OpenCV to rasterize, align, and classify drawing differences, then
uses **PyQt6 `QGraphicsView`** to render the result. There is no Qt WebEngine,
embedded browser, local web server, or browser profile.

![Native viewer architecture](docs/native-viewer-architecture.svg)

## What it does

- Opens an old and a new PDF, with a selectable page in each document.
- Rasterizes and aligns the old drawing to the new one before calculating the
  difference, which reduces false positives caused by small page shifts.
- Displays the old drawing, new drawing, red additions, and blue removals as
  independent `QGraphicsPixmapItem` layers.
- Provides a smooth old-to-new review slider, independent addition/removal
  toggles, pan and wheel zoom, fit/reset controls, and a clickable change list.
- Keeps all image processing local. It does not upload drawings or require an
  API key.

## Run it

Python 3.10+ is required.

```powershell
git clone https://github.com/jxfuller1/PDF_differences_viewer.git
cd PDF_differences_viewer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pdf-differences-viewer
```

You can also start it without installing the command:

```powershell
python -m pdf_differences_viewer
```

## Native viewer controls

| Control | Action |
| --- | --- |
| Mouse wheel | Zoom around the cursor |
| Left-drag | Pan the drawing |
| Click a change | Select it in the change list |
| Double-click a change | Zoom to that change |
| `F` / Fit | Fit the drawing to the viewer |
| `0` / Reset | Return to a 1:1 zoom |

The blend slider starts on the old drawing and ends on the new drawing. Red and
blue difference layers are strongest in the middle, so the endpoints remain
clean for direct revision review.

## Development

```powershell
pip install -e ".[dev]"
pytest
```

## Design

The application deliberately uses Qt's scene graph rather than HTML:

```text
PDFs → OpenCV comparison → QGraphicsScene
                           ├── old drawing pixmap
                           ├── new drawing pixmap
                           ├── red additions pixmap
                           ├── blue removals pixmap
                           └── interactive region overlays
```

The `QGraphicsView` approach lets Qt composite the layers directly, preserves
high-quality pixmap rendering at arbitrary zoom levels, and makes region
selection/panning native Qt interactions.

## Attribution

This is a native-viewer rewrite inspired by
[DrawingContrast](https://github.com/ayumilove/DrawingContrast), whose MIT
license and attribution are reproduced in [NOTICE.md](NOTICE.md).
