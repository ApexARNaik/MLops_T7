from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pymupdf
import numpy as np


MAX_PDF_SIZE_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 50
MAX_PAGE_PIXELS = 40_000_000
DEFAULT_DPI = 200


class PDFProcessingError(ValueError):
    """Raised when a PDF is invalid or exceeds safe processing limits."""


def render_pdf_pages(
    path: str | Path,
    *,
    max_file_size_bytes: int = MAX_PDF_SIZE_BYTES,
    max_pages: int = MAX_PDF_PAGES,
    dpi: int = DEFAULT_DPI,
) -> Iterator[Any]:
    pdf_path = Path(path)
    if pdf_path.suffix.lower() != ".pdf":
        raise PDFProcessingError("The uploaded document must be a PDF.")
    if max_file_size_bytes <= 0 or max_pages <= 0 or dpi <= 0:
        raise PDFProcessingError("PDF processing limits must be positive.")

    try:
        file_size = pdf_path.stat().st_size
        with pdf_path.open("rb") as pdf_file:
            header = pdf_file.read(5)
    except OSError as exc:
        raise PDFProcessingError("The PDF could not be read.") from exc

    if file_size > max_file_size_bytes:
        raise PDFProcessingError("The PDF exceeds the maximum allowed file size.")
    if header != b"%PDF-":
        raise PDFProcessingError("The document is not a valid PDF.")

    try:
        document = pymupdf.open(pdf_path)
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise PDFProcessingError("The PDF could not be opened.") from exc

    try:
        if document.is_encrypted:
            raise PDFProcessingError("Encrypted PDFs are not supported.")
        if document.page_count == 0:
            raise PDFProcessingError("The PDF contains no pages.")
        if document.page_count > max_pages:
            raise PDFProcessingError("The PDF exceeds the maximum allowed page count.")

        for page in document:
            width = int(page.rect.width * dpi / 72)
            height = int(page.rect.height * dpi / 72)
            if width <= 0 or height <= 0 or width * height > MAX_PAGE_PIXELS:
                raise PDFProcessingError("A PDF page exceeds the maximum render size.")
            try:
                pixmap = page.get_pixmap(dpi=dpi, alpha=False)
            except (RuntimeError, ValueError) as exc:
                raise PDFProcessingError("A PDF page could not be rendered.") from exc
            yield np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.height, pixmap.width, pixmap.n
            ).copy()
    finally:
        document.close()