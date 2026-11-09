"""Test parse JSON: output thô từ LLM (distractor) và bộ đề JSON (import)."""

import pytest
from distractor import _safe_parse_json
from schema import parse_mcq_list


def test_parse_plain_json_array():
    raw = '["Huế", "Đà Nẵng", "Hải Phòng"]'

    assert _safe_parse_json(raw, 3, "Hà Nội") == ["Huế", "Đà Nẵng", "Hải Phòng"]


def test_parse_strips_markdown_fence_and_duplicates():
    raw = '```json\n["Hà Nội", "Huế", "Huế", "Đà Nẵng"]\n```'

    assert _safe_parse_json(raw, 3, "Hà Nội") == ["Huế", "Đà Nẵng"]


def test_parse_extracts_array_from_chatty_response():
    raw = 'Dưới đây là kết quả:\n["Huế", "Đà Nẵng", "Hải Phòng"]\nChúc bạn học tốt!'

    assert _safe_parse_json(raw, 3, "Hà Nội") == ["Huế", "Đà Nẵng", "Hải Phòng"]


def test_parse_falls_back_to_line_splitting_when_not_json():
    raw = '1. Huế\n2. Đà Nẵng\n3. Hải Phòng'

    assert _safe_parse_json(raw, 3, "Hà Nội") == ["Huế", "Đà Nẵng", "Hải Phòng"]


def test_parse_limits_to_requested_count():
    raw = '["Huế", "Đà Nẵng", "Hải Phòng", "Cần Thơ", "Vinh"]'

    assert _safe_parse_json(raw, 2, "Hà Nội") == ["Huế", "Đà Nẵng"]


def test_parse_missing_field_raises_in_bundle():
    with pytest.raises(ValueError):
        parse_mcq_list([{"question": "Chỉ có câu hỏi?"}])


def test_parse_bundle_from_json_string():
    raw = """
    [
      {"question": "Thủ đô của Việt Nam là gì?", "options": ["Huế", "Hà Nội", "Đà Nẵng", "Vinh"],
       "correct_index": 1}
    ]
    """

    mcqs = parse_mcq_list(raw)

    assert len(mcqs) == 1
    assert mcqs[0].answer == "Hà Nội"
