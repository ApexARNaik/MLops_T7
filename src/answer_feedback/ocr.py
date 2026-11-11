import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol

from src.answer_feedback.document import render_pdf_pages


MIN_OCR_CONFIDENCE_FOR_AUTO_SCORING = 0.6
MIN_ANSWER_WORDS_FOR_AUTO_SCORING = 3
SHORT_ANSWER_QIDS = tuple(f"Q{number}" for number in range(21, 36))
_QUESTION_LABEL = re.compile(r"(?m)^\s*(?:Q\.?\s*)?(\d{1,2})\s*[.)\-:]\s*")


class OCRProcessingError(RuntimeError):
    """Raised when an OCR engine cannot safely transcribe an image."""


class OCREngine(Protocol):
    def transcribe(self, image: Any) -> tuple[str, float]: ...


class EasyOCREngine:
    def __init__(
        self,
        languages: tuple[str, ...] = ("en",),
        *,
        reader: Any | None = None,
    ) -> None:
        self._languages = languages
        self._reader = reader

    def _get_reader(self) -> Any:
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise OCRProcessingError(
                    "EasyOCR is unavailable; install the development OCR dependencies."
                ) from exc
            self._reader = easyocr.Reader(list(self._languages), gpu=False)
        return self._reader

    def transcribe(self, image: Any) -> tuple[str, float]:
        try:
            results = self._get_reader().readtext(image, detail=1, paragraph=False)
        except (OSError, RuntimeError, ValueError) as exc:
            raise OCRProcessingError("The OCR engine failed to process the page.") from exc

        if not isinstance(results, (list, tuple)):
            raise OCRProcessingError("The OCR engine returned an invalid result.")
        text_parts: list[str] = []
        confidences: list[float] = []
        for result in results:
            if not isinstance(result, (list, tuple)) or len(result) < 3:
                raise OCRProcessingError("The OCR engine returned an invalid result.")
            text = str(result[1]).strip()
            try:
                confidence = float(result[2])
            except (TypeError, ValueError, OverflowError) as exc:
                raise OCRProcessingError(
                    "The OCR engine returned an invalid confidence."
                ) from exc
            if not math.isfinite(confidence):
                raise OCRProcessingError("The OCR engine returned an invalid confidence.")
            if text:
                text_parts.append(text)
                confidences.append(min(1.0, max(0.0, confidence)))

        if not text_parts:
            return "", 0.0
        return "\n".join(text_parts), sum(confidences) / len(confidences)


def manual_transcription(text: str) -> tuple[str, float]:
    cleaned_text = text.strip()
    return cleaned_text, 1.0 if cleaned_text else 0.0


def write_manual_transcription(
    output_csv: str | Path,
    student_id: str,
    qid: str,
    text: str,
) -> None:
    if not student_id.strip() or qid not in SHORT_ANSWER_QIDS:
        raise ValueError("A student identifier and supported question ID are required.")

    answer_text, confidence = manual_transcription(text)
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("student_id", "qid", "answer_text", "ocr_conf", "source"),
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "student_id": student_id.strip(),
                "qid": qid,
                "answer_text": answer_text,
                "ocr_conf": confidence,
                "source": "manual",
            }
        )


def requires_human_review(
    text: str | None,
    confidence: float | None,
    *,
    known_question: bool = True,
) -> bool:
    if not known_question or text is None:
        return True
    if len(text.split()) < MIN_ANSWER_WORDS_FOR_AUTO_SCORING:
        return True
    return confidence is not None and (
        not math.isfinite(confidence)
        or not 0 <= confidence <= 1
        or confidence < MIN_OCR_CONFIDENCE_FOR_AUTO_SCORING
    )


def segment_answers(text: str) -> dict[str, str]:
    matches = list(_QUESTION_LABEL.finditer(text))
    segments: dict[str, str] = {}
    for index, match in enumerate(matches):
        question_number = int(match.group(1))
        if not 21 <= question_number <= 35:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        answer = text[start:end].strip()
        if answer:
            segments[f"Q{question_number}"] = answer
    return segments


def process_all_pdfs(
    input_dir: str | Path = Path("data/answer_feedback/raw/Student_Pdf"),
    output_dir: str | Path = Path("data/answer_feedback/interim"),
    *,
    engine: OCREngine | None = None,
) -> float:
    pdf_paths = sorted(Path(input_dir).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError("No student PDFs were found in the input directory.")

    output_path = Path(output_dir)
    pages_path = output_path / "ocr_pages"
    pages_path.mkdir(parents=True, exist_ok=True)
    selected_engine = engine or EasyOCREngine()
    answer_pages: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for pdf_path in pdf_paths:
        for page_number, image in enumerate(render_pdf_pages(pdf_path), start=1):
            page_text, confidence = selected_engine.transcribe(image)
            (pages_path / f"{pdf_path.stem}_page_{page_number:03}.txt").write_text(
                page_text, encoding="utf-8"
            )
            for qid, answer in segment_answers(page_text).items():
                answer_pages[pdf_path.stem][qid].append((answer, confidence))

    rows: list[dict[str, str | float]] = []
    for student_id, questions in answer_pages.items():
        for qid, pages in questions.items():
            rows.append(
                {
                    "student_id": student_id,
                    "qid": qid,
                    "answer_text": " ".join(text for text, _ in pages),
                    "ocr_conf": sum(conf for _, conf in pages) / len(pages),
                    "source": "easyocr",
                }
            )

    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "answers_ocr.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=("student_id", "qid", "answer_text", "ocr_conf", "source"),
        )
        writer.writeheader()
        writer.writerows(rows)

    expected_pairs = len(pdf_paths) * len(SHORT_ANSWER_QIDS)
    found_pairs = sum(
        qid in questions for questions in answer_pages.values() for qid in SHORT_ANSWER_QIDS
    )
    coverage = found_pairs / expected_pairs
    print(f"OCR question coverage: {coverage:.1%}")
    return coverage


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe student PDFs for answer feedback.")
    parser.add_argument("--all", action="store_true", help="Process all PDFs in the default input directory.")
    args = parser.parse_args()
    if not args.all:
        parser.error("Pass --all to process the available student PDFs.")
    process_all_pdfs()


if __name__ == "__main__":
    main()