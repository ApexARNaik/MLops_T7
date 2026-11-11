import numpy as np
import pytest
import pymupdf

from src.answer_feedback.document import PDFProcessingError, render_pdf_pages
from src.answer_feedback.ocr import (
    EasyOCREngine,
    OCRProcessingError,
    manual_transcription,
    process_all_pdfs,
    requires_human_review,
    segment_answers,
    write_manual_transcription,
)


def create_pdf(path, page_count=1):
    document = pymupdf.open()
    for _ in range(page_count):
        document.new_page().insert_text((72, 72), "A rendered answer page.")
    document.save(path)
    document.close()
    return path


def test_render_pdf_pages_yields_page_images(tmp_path):
    pdf_path = create_pdf(tmp_path / "answer.pdf")

    pages = list(render_pdf_pages(pdf_path))

    assert len(pages) == 1
    assert isinstance(pages[0], np.ndarray)
    assert pages[0].ndim == 3


def test_render_pdf_rejects_non_pdf_extension(tmp_path):
    text_path = tmp_path / "answer.txt"
    text_path.write_text("not a PDF", encoding="utf-8")

    with pytest.raises(PDFProcessingError, match="must be a PDF"):
        list(render_pdf_pages(text_path))


def test_render_pdf_rejects_invalid_pdf_content(tmp_path):
    pdf_path = tmp_path / "answer.pdf"
    pdf_path.write_bytes(b"not a PDF")

    with pytest.raises(PDFProcessingError, match="not a valid PDF"):
        list(render_pdf_pages(pdf_path))


def test_render_pdf_enforces_file_size_limit(tmp_path):
    pdf_path = create_pdf(tmp_path / "answer.pdf")

    with pytest.raises(PDFProcessingError, match="file size"):
        list(render_pdf_pages(pdf_path, max_file_size_bytes=1))


def test_render_pdf_enforces_page_count_limit(tmp_path):
    pdf_path = create_pdf(tmp_path / "answer.pdf", page_count=2)

    with pytest.raises(PDFProcessingError, match="page count"):
        list(render_pdf_pages(pdf_path, max_pages=1))


def test_easyocr_combines_text_and_confidence():
    class Reader:
        def readtext(self, image, *, detail, paragraph):
            assert detail == 1
            assert paragraph is False
            return [([], "first line", 0.8), ([], "second line", 1.0)]

    text, confidence = EasyOCREngine(reader=Reader()).transcribe(np.zeros((2, 2, 3)))

    assert text == "first line\nsecond line"
    assert confidence == pytest.approx(0.9)


def test_easyocr_returns_zero_confidence_for_empty_page():
    class Reader:
        def readtext(self, image, *, detail, paragraph):
            return []

    assert EasyOCREngine(reader=Reader()).transcribe(np.zeros((2, 2, 3))) == ("", 0.0)


def test_easyocr_surfaces_engine_failures():
    class Reader:
        def readtext(self, image, *, detail, paragraph):
            raise RuntimeError("engine failed")

    with pytest.raises(OCRProcessingError, match="failed to process"):
        EasyOCREngine(reader=Reader()).transcribe(np.zeros((2, 2, 3)))


def test_manual_transcription_and_review_conditions():
    assert manual_transcription("  answer text  ") == ("answer text", 1.0)
    assert manual_transcription("  ") == ("", 0.0)
    assert requires_human_review("", 1.0)
    assert requires_human_review("too few", 1.0)
    assert requires_human_review("four meaningful answer words", 0.59)
    assert requires_human_review("four meaningful answer words", float("nan"))
    assert requires_human_review("four meaningful answer words", 1.0, known_question=False)
    assert not requires_human_review("four meaningful answer words", 0.6)


def test_manual_transcription_is_written_with_manual_source(tmp_path):
    output_csv = tmp_path / "interim" / "answers_ocr.csv"

    write_manual_transcription(output_csv, "student-1", "Q21", "typed answer text")

    assert "student-1,Q21,typed answer text,1.0,manual" in output_csv.read_text(
        encoding="utf-8"
    )


def test_segment_answers_uses_only_short_answer_question_ids():
    segmented = segment_answers("Q20. ignored\nQ21. first answer\n22) next answer\nQ36. ignored")

    assert segmented == {"Q21": "first answer", "Q22": "next answer"}


def test_batch_ocr_writes_transcripts_without_printing_answer_text(tmp_path, capsys):
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    create_pdf(input_dir / "student-1.pdf")

    class Engine:
        def transcribe(self, image):
            return "Q21. answer words with enough detail", 0.9

    coverage = process_all_pdfs(input_dir, tmp_path / "interim", engine=Engine())

    output = (tmp_path / "interim" / "answers_ocr.csv").read_text(encoding="utf-8")
    page_output = (
        tmp_path / "interim" / "ocr_pages" / "student-1_page_001.txt"
    ).read_text(encoding="utf-8")
    assert coverage == pytest.approx(1 / 15)
    assert "answer words with enough detail" in output
    assert "answer words with enough detail" in page_output
    assert "answer words with enough detail" not in capsys.readouterr().out
