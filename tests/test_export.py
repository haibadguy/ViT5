"""Test export: Word / PDF / TXT / JSON / CSV giữ đúng dấu tiếng Việt."""

import io
import json

import pytest
from export_utils import (
    export_csv_bytes,
    export_json_bytes,
    export_pdf_bytes,
    export_txt_bytes,
    export_word_bytes,
)

MCQS = [
    {
        "question": "Thủ đô của Việt Nam là thành phố nào?",
        "answer": "Hà Nội",
        "options": ["Huế", "Hà Nội", "Đà Nẵng", "Hải Phòng"],
        "correct_label": "B",
    },
    {
        "question": "Sông nào chảy qua Hà Nội?",
        "answer": "Sông Hồng",
        "options": ["Sông Hồng", "Sông Mã", "Sông Cửu Long", "Sông Đồng Nai"],
        "correct_label": "A",
    },
]


def test_pdf_preserves_vietnamese_diacritics():
    pdf_bytes = export_pdf_bytes(MCQS, title="Đề kiểm tra thử")

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    page1 = reader.pages[0].extract_text()

    assert "Câu 1." in page1
    assert "Thủ đô của Việt Nam" in page1
    assert "Đà Nẵng" in page1


def test_pdf_puts_answers_on_separate_page():
    pdf_bytes = export_pdf_bytes(MCQS)

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) == 2

    page1 = reader.pages[0].extract_text()
    page2 = reader.pages[1].extract_text()

    assert "ĐÁP ÁN" not in page1
    assert "ĐÁP ÁN" in page2
    assert "Câu 1: B" in page2


def test_word_export_keeps_diacritics_and_answer_page():
    from docx import Document

    doc = Document(io.BytesIO(export_word_bytes(MCQS, title="Đề kiểm tra thử")))
    texts = [p.text for p in doc.paragraphs]

    assert any("Câu 1. Thủ đô của Việt Nam" in t for t in texts)
    assert "ĐÁP ÁN" in texts

    table_text = "\n".join(cell.text for row in doc.tables[0].rows for cell in row.cells)
    assert "Câu 1: B" in table_text


def test_txt_export_contains_diacritics_and_answer_key():
    text = export_txt_bytes(MCQS).decode("utf-8")

    assert "Câu 1. Thủ đô của Việt Nam" in text
    assert "ĐÁP ÁN" in text
    assert "Câu 1: B. Hà Nội" in text


def test_json_export_is_canonical_and_unescaped():
    raw = export_json_bytes(MCQS).decode("utf-8")
    data = json.loads(raw)

    assert "Hà Nội" in raw  # không escape thành \u1e…
    assert data[0]["correct_index"] == 1
    assert data[0]["correct_label"] == "B"
    assert data[1]["options"][0] == "Sông Hồng"


def test_csv_export_matches_quizizz_template():
    raw = export_csv_bytes(MCQS)
    assert raw.startswith(b"\xef\xbb\xbf")  # BOM cho Excel

    lines = raw.decode("utf-8-sig").strip().splitlines()
    assert lines[0] == "Question,Option A,Option B,Option C,Option D,Correct"
    assert lines[1].endswith(",B")
    assert "Thủ đô của Việt Nam" in lines[1]


def test_export_rejects_empty_list():
    with pytest.raises(ValueError):
        export_pdf_bytes([])
