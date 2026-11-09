"""Test generator: không tải model thật, mock toàn bộ phần inference."""

from unittest.mock import patch

import generator
from generator import QAGenerator, _answer_in_context, _is_multitask, _split_sentences

CONTEXT = (
    "Hà Nội là thủ đô của Việt Nam. "
    "Thành phố có lịch sử hơn 1000 năm và từng được gọi là Thăng Long. "
    "Sông Hồng chảy qua phía đông bắc thành phố."
)


def make_generator(model_name="test/vit5-qg-ae"):
    """Tạo QAGenerator nhưng bỏ qua bước load model (không cần torch)."""
    with patch.object(generator.QAGenerator, "_load_local", lambda self: None):
        return QAGenerator(model_name=model_name)


def test_is_multitask_detects_qg_and_ae():
    assert _is_multitask("shnl/vit5-vinewsqa-qg-ae") is True
    assert _is_multitask("namngo/pipeline-vit5-viquad-qg") is False


def test_split_sentences_keeps_vietnamese_punctuation():
    sentences = _split_sentences(CONTEXT)

    assert len(sentences) == 3
    assert sentences[0].startswith("Hà Nội")


def test_answer_in_context_tolerates_case_and_nfc():
    assert _answer_in_context("hà nội", CONTEXT) is True
    assert _answer_in_context("Sài Gòn", CONTEXT) is False


def test_generate_builds_pairs_and_skips_duplicate_questions():
    qa_gen = make_generator()
    qa_gen._extract_answers_multitask = lambda context, need: ["Hà Nội", "Thăng Long", "Sông Hồng"]
    qa_gen._generate_question = lambda context, answer: "Câu hỏi về " + answer

    pairs = qa_gen.generate(CONTEXT, num_pairs=2)

    assert len(pairs) == 2
    assert pairs[0] == {"question": "Câu hỏi về Hà Nội", "answer": "Hà Nội"}
    assert all(isinstance(p["question"], str) for p in pairs)


def test_generate_deduplicates_identical_questions():
    qa_gen = make_generator()
    qa_gen._extract_answers_multitask = lambda context, need: ["Hà Nội", "Thăng Long"]
    qa_gen._generate_question = lambda context, answer: "Câu hỏi giống nhau?"

    pairs = qa_gen.generate(CONTEXT, num_pairs=2)

    assert len(pairs) == 1


def test_generate_returns_empty_on_empty_context():
    qa_gen = make_generator()

    assert qa_gen.generate("   ", num_pairs=3) == []
