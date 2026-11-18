"""High-speed PDF sheet music rasterizer using PyMuPDF (zero Poppler dependency)."""

from __future__ import annotations

from pathlib import Path


def render_pdf_pages(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 300,
) -> list[Path]:
    """Render every page of a sheet music PDF into opaque RGB PNGs for OMR."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Sheet music input must be a .pdf file, got: {pdf_path.suffix}")

    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError(
            "PDF OMR fallback requires the optional 'pdf' extra (`pymupdf`). "
            "Install with: uv sync --extra pdf"
        ) from exc

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scale = dpi / 72.0
    matrix = pymupdf.Matrix(scale, scale)
    image_paths: list[Path] = []

    with pymupdf.open(str(pdf_path)) as doc:
        if len(doc) == 0:
            raise ValueError(f"PDF has zero pages: {pdf_path}")
        for idx, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=matrix, colorspace=pymupdf.csRGB, alpha=False)
            page_path = out_dir / f"page_{idx:03d}.png"
            pix.save(str(page_path))
            image_paths.append(page_path)

    return image_paths
