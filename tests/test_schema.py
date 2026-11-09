"""Test cho schema MCQ: validate, chuyển đổi, tổng hợp báo cáo."""

import pytest
from schema import (
    MCQ,
    REASON_BAD_INDEX,
    REASON_DUPLICATE_OPTION,
    REASON_OPTION_COUNT,
    REASON_QUESTION_TOO_SHORT,
    Rejection,
    build_mcq,
    parse_mcq_list,
    shuffle_mcq,
    summarize,
    validate_batch,
)


def make_options():
    return ["Huế", "Hà Nội", "Đà Nẵng", "Hải Phòng"]


def test_valid_mcq_has_no_errors():
    mcq = MCQ(question="Thủ đô của Việt Nam là gì?", options=make_options(), correct_index=1)

    assert mcq.validate() == []
    assert mcq.answer == "Hà Nội"
    assert mcq.correct_label == "B"


def test_question_too_short_is_rejected():
    mcq = MCQ(question="Gì?", options=make_options(), correct_index=0)

    assert REASON_QUESTION_TOO_SHORT in mcq.validate()


def test_wrong_option_count_is_rejected():
    mcq = MCQ(question="Thủ đô của Việt Nam là gì?", options=["Huế", "Hà Nội"], correct_index=1)

    assert REASON_OPTION_COUNT in mcq.validate()


def test_duplicate_options_are_rejected_case_insensitive():
    mcq = MCQ(
        question="Thủ đô của Việt Nam là gì?",
        options=["Hà Nội", " hà nội ", "Đà Nẵng", "Huế"],
        correct_index=0,
    )

    assert REASON_DUPLICATE_OPTION in mcq.validate()


def test_correct_index_out_of_range_is_rejected():
    mcq = MCQ(question="Thủ đô của Việt Nam là gì?", options=make_options(), correct_index=9)

    assert REASON_BAD_INDEX in mcq.validate()
    assert mcq.correct_label == "?"


def test_from_dict_accepts_app_format_and_round_trips():
    app_dict = {
        "question": "Thủ đô của Việt Nam là gì?",
        "answer": "Hà Nội",
        "options": make_options(),
        "correct_label": "B",
        "source": "ViT5 + LLM",
    }

    mcq = MCQ.from_dict(app_dict)

    assert mcq.correct_index == 1
    assert mcq.to_dict()["correct_label"] == "B"
    assert mcq.to_dict()["answer"] == "Hà Nội"
    assert mcq.to_canonical_dict()["correct_index"] == 1


def test_from_dict_without_options_raises():
    with pytest.raises(ValueError):
        MCQ.from_dict({"question": "Câu hỏi đủ dài?"})


def test_validate_batch_reports_reasons_and_summary():
    good = MCQ(question="Thủ đô của Việt Nam là gì?", options=make_options(), correct_index=1)
    short = MCQ(question="Gì?", options=make_options(), correct_index=0)
    bad_index = MCQ(question="Thành phố nào đông dân nhất?", options=make_options(), correct_index=7)

    kept, rejected = validate_batch([good, short, bad_index])
    report = summarize(3, len(kept), rejected)

    assert kept == [good]
    assert len(rejected) == 2
    assert all(isinstance(item, Rejection) for item in rejected)
    assert "giữ 1 câu" in report
    assert "loại 2" in report


def test_summarize_without_rejections():
    report = summarize(3, 3, [])

    assert "không câu nào bị loại" in report


def test_parse_mcq_list_accepts_bundle_and_validates():
    payload = {
        "version": "1.0",
        "questions": [
            {
                "question": "Thủ đô của Việt Nam là gì?",
                "options": make_options(),
                "correct_index": 1,
            },
            {
                "question": "Sông nào chảy qua Hà Nội?",
                "options": ["Sông Hồng", "Sông Mã", "Sông Đà", "Sông Lam"],
                "correct_label": "A",
            },
        ],
    }

    mcqs = parse_mcq_list(payload)
    kept, rejected = validate_batch(mcqs)

    assert len(kept) == 2
    assert rejected == []
    assert kept[1].correct_label == "A"


# ── build_mcq: ghép Q/A/distractors ────────────────────────────
def test_build_mcq_drops_distractor_equal_to_answer():
    mcq = build_mcq("Thủ đô của Việt Nam là gì?", "Hà Nội", ["hà nội", "Huế", "Đà Nẵng"])

    assert mcq["correct_label"] != "?"  # đáp án đúng phải trỏ đúng chỗ
    assert MCQ.from_dict(mcq).answer == "Hà Nội"
    lowered = [opt.lower() for opt in mcq["options"]]
    assert lowered.count("hà nội") == 1


def test_build_mcq_keeps_four_distractors_when_asked():
    distractors = ["Huế", "Đà Nẵng", "Hải Phòng", "Cần Thơ"]

    mcq = build_mcq("Thủ đô của Việt Nam là gì?", "Hà Nội", distractors, num_options=5)

    assert len(mcq["options"]) == 5
    assert set(distractors).issubset(set(mcq["options"]))
    assert MCQ.from_dict(mcq).validate() == []


def test_build_mcq_pads_when_distractors_missing():
    mcq = build_mcq("Thủ đô của Việt Nam là gì?", "Hà Nội", ["Huế"])

    assert len(mcq["options"]) == 4
    assert MCQ.from_dict(mcq).validate() == []


def test_shuffle_mcq_keeps_same_correct_answer():
    original = {
        "question": "Thủ đô của Việt Nam là gì?",
        "answer": "Hà Nội",
        "options": ["Huế", "Hà Nội", "Đà Nẵng", "Hải Phòng"],
        "correct_label": "B",
        "source": "ViT5 + LLM",
    }

    for _ in range(10):
        shuffled = shuffle_mcq(original)
        assert shuffled["answer"] == "Hà Nội"
        assert shuffled["options"][["A", "B", "C", "D"].index(shuffled["correct_label"])] == "Hà Nội"
        assert sorted(shuffled["options"]) == sorted(original["options"])
