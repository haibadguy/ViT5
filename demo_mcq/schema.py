"""
schema.py
─────────
Kiểu dữ liệu MCQ dùng chung cho app, generator và export.

Mục đích:
  • Một nơi duy nhất định nghĩa "một câu trắc nghiệm hợp lệ".
  • Validate trước khi hiển thị / xuất đề → câu hỏng không lọt ra output.
  • Chuyển đổi qua lại giữa format chuẩn (``correct_index``) và format app
    đang dùng (``answer`` + ``correct_label``) mà không phá code cũ.

Dùng:
    from schema import MCQ, validate_batch, summarize

    mcq = MCQ.from_dict({"question": "...", "options": [...], "correct_index": 0})
    assert mcq.validate() == []
    kept, rejected = validate_batch([mcq])
    print(summarize(1, len(kept), rejected))
"""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

LABELS: Tuple[str, ...] = ("A", "B", "C", "D", "E")

# ── ngưỡng validate ────────────────────────────────────────────
MIN_QUESTION_CHARS = 5
# App cho phép 3 hoặc 4 distractor → 4 hoặc 5 lựa chọn
ALLOWED_OPTION_COUNTS: Tuple[int, ...] = (4, 5)

# ── lý do loại (dùng trực tiếp trên UI) ────────────────────────
REASON_QUESTION_TOO_SHORT = "câu hỏi quá ngắn"
REASON_OPTION_COUNT = "sai số lựa chọn"
REASON_EMPTY_OPTION = "lựa chọn rỗng"
REASON_DUPLICATE_OPTION = "lựa chọn bị trùng"
REASON_BAD_INDEX = "vị trí đáp án đúng ngoài phạm vi"


def _clean(text: Any) -> str:
    """Chuẩn hoá khoảng trắng + ép về str, chấp nhận None."""
    return " ".join(str(text if text is not None else "").split()).strip()


@dataclass
class MCQ:
    """Một câu hỏi trắc nghiệm đã chuẩn hoá."""

    question: str
    options: List[Any]
    correct_index: int
    explanation: str = ""

    def __post_init__(self) -> None:
        self.question = _clean(self.question)
        self.options = [_clean(o) for o in (self.options or [])]
        self.explanation = _clean(self.explanation)
        try:
            self.correct_index = int(self.correct_index)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            self.correct_index = -1

    # ── thông tin dẫn xuất ─────────────────────────────────────
    @property
    def answer(self) -> str:
        """Nội dung đáp án đúng (chuỗi rỗng nếu index sai)."""
        if 0 <= self.correct_index < len(self.options):
            return self.options[self.correct_index]
        return ""

    @property
    def correct_label(self) -> str:
        """Nhãn A/B/C/D… của đáp án đúng ("?" nếu index sai)."""
        if 0 <= self.correct_index < len(LABELS):
            return LABELS[self.correct_index]
        return "?"

    # ── validate ───────────────────────────────────────────────
    def validate(
        self,
        allowed_option_counts: Sequence[int] = ALLOWED_OPTION_COUNTS,
    ) -> List[str]:
        """Trả về danh sách lỗi (rỗng = hợp lệ)."""
        errors: List[str] = []

        if len(self.question) < MIN_QUESTION_CHARS:
            errors.append(REASON_QUESTION_TOO_SHORT)

        if len(self.options) not in allowed_option_counts:
            errors.append(REASON_OPTION_COUNT)

        if any(not opt for opt in self.options):
            errors.append(REASON_EMPTY_OPTION)

        filled = [opt.lower() for opt in self.options if opt]
        if len(set(filled)) < len(filled):
            errors.append(REASON_DUPLICATE_OPTION)

        if not 0 <= self.correct_index < len(self.options):
            errors.append(REASON_BAD_INDEX)

        return errors

    def is_valid(self, allowed_option_counts: Sequence[int] = ALLOWED_OPTION_COUNTS) -> bool:
        return not self.validate(allowed_option_counts)

    # ── chuyển đổi ─────────────────────────────────────────────
    def to_dict(self, source: str = "ViT5 + LLM") -> Dict[str, Any]:
        """Format mà app.py / export_utils.py đang dùng."""
        return {
            "question": self.question,
            "answer": self.answer,
            "options": list(self.options),
            "correct_label": self.correct_label,
            "source": source,
            "explanation": self.explanation,
        }

    def to_canonical_dict(self) -> Dict[str, Any]:
        """Format chuẩn để lưu JSON / trao đổi với hệ thống khác."""
        return {
            "question": self.question,
            "options": list(self.options),
            "correct_index": self.correct_index,
            "correct_label": self.correct_label,
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, data: Any) -> "MCQ":
        """
        Nhận dict theo format chuẩn (``correct_index``) *hoặc* format app
        (``answer`` / ``correct_label``). Ném ``ValueError`` nếu thiếu dữ liệu
        tối thiểu (question + options).
        """
        if isinstance(data, MCQ):
            return data
        if not isinstance(data, dict):
            raise ValueError("Mỗi câu hỏi phải là một object JSON")

        question = data.get("question") or data.get("q") or ""
        options = data.get("options")
        if options is None:
            options = data.get("choices")
        if not options or not isinstance(options, (list, tuple)):
            raise ValueError("Thiếu 'options'")

        index = _coerce_index(data.get("correct_index"))
        if index is None:
            index = _coerce_index(data.get("correct_option"))
        if index is None:
            label = data.get("correct_label") or data.get("correct") or data.get("answer_label")
            if label is not None:
                index = _coerce_index(label, labels_allowed=True)
        if index is None and data.get("answer") is not None:
            target = _clean(data["answer"]).lower()
            matches = [i for i, opt in enumerate(options) if _clean(opt).lower() == target]
            index = matches[0] if matches else None

        return cls(
            question=question,
            options=list(options),
            correct_index=index if index is not None else -1,
            explanation=data.get("explanation") or "",
        )


def _coerce_index(value: Any, labels_allowed: bool = False) -> Optional[int]:
    """Đổi 'B' / '2' / 2 → int index; trả None nếu không đọc được."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = _clean(value).upper()
    if not text:
        return None
    if labels_allowed and text in LABELS:
        return LABELS.index(text)
    try:
        return int(text)
    except ValueError:
        return None


@dataclass
class Rejection:
    """Một câu bị loại, kèm lý do để hiển thị cho người dùng."""

    index: int
    reasons: List[str]
    question: str = ""


def validate_batch(
    items: Iterable[Any],
    allowed_option_counts: Sequence[int] = ALLOWED_OPTION_COUNTS,
) -> Tuple[List[MCQ], List[Rejection]]:
    """Validate cả lô → (câu giữ lại, câu bị loại kèm lý do)."""
    kept: List[MCQ] = []
    rejected: List[Rejection] = []

    for position, raw in enumerate(items):
        try:
            mcq = raw if isinstance(raw, MCQ) else MCQ.from_dict(raw)
        except ValueError as exc:
            rejected.append(Rejection(index=position, reasons=[str(exc)]))
            continue

        errors = mcq.validate(allowed_option_counts)
        if errors:
            rejected.append(Rejection(index=position, reasons=errors, question=mcq.question))
        else:
            kept.append(mcq)

    return kept, rejected


def summarize(total: int, kept: int, rejected: Sequence[Rejection]) -> str:
    """Câu thông báo kiểu: 'Sinh 10 câu, giữ 8 câu hợp lệ (loại 2: …)'."""
    if not rejected:
        return f"Sinh {total} câu · giữ {kept} câu · không câu nào bị loại."

    counter: Counter = Counter(reason for item in rejected for reason in set(item.reasons))
    detail = ", ".join(f"{reason} ×{count}" for reason, count in counter.most_common())
    return (
        f"Sinh {total} câu, giữ {kept} câu hợp lệ "
        f"(loại {len(rejected)}: {detail})."
    )


def parse_mcq_list(payload: Any) -> List[MCQ]:
    """
    Parse bộ đề từ JSON (str / list / ``{"questions": [...]}`` ) → ``List[MCQ]``.

    Không validate ở đây — dùng kèm :func:`validate_batch` để biết câu nào hỏng.
    """
    if isinstance(payload, str):
        payload = json.loads(payload)

    if isinstance(payload, dict):
        payload = payload.get("questions") or payload.get("mcq_list") or []

    if not isinstance(payload, list):
        raise ValueError("JSON phải là một list hoặc object có khoá 'questions'")

    return [MCQ.from_dict(item) for item in payload]


# ══════════════════════════════════════════════════════════════
# Ghép câu hỏi: Q + A + distractors → MCQ (xáo thứ tự đáp án)
# ══════════════════════════════════════════════════════════════
PLACEHOLDER_DISTRACTOR = "[Đáp án sai {i}]"


def _clean_distractors(
    distractors: Iterable[Any],
    answer: str,
    needed: int,
) -> List[str]:
    """Bỏ distractor rỗng / trùng đáp án đúng / trùng nhau (không phân biệt hoa thường)."""
    seen = {_clean(answer).lower()}
    cleaned: List[str] = []

    for item in distractors:
        text = _clean(item)
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) == needed:
            break

    return cleaned


def build_mcq(
    question: str,
    answer: str,
    distractors: Iterable[Any],
    num_options: int = 4,
    pad_missing: bool = True,
) -> Dict[str, Any]:
    """
    Ghép ``question + answer + distractors`` thành một MCQ đã xáo thứ tự.

    Args:
        distractors: đáp án nhiễu do LLM sinh (có thể trùng/rỗng).
        num_options: tổng số lựa chọn mong muốn (4 hoặc 5).
        pad_missing: thiếu distractor thì thêm placeholder để câu vẫn xuất được.

    Returns:
        Dict theo format app (``answer`` + ``correct_label``).
    """
    answer = _clean(answer)
    needed = max(1, num_options - 1)
    cleaned = _clean_distractors(distractors, answer, needed)

    if pad_missing:
        while len(cleaned) < needed:
            cleaned.append(PLACEHOLDER_DISTRACTOR.format(i=len(cleaned) + 1))

    options = cleaned + [answer]
    random.shuffle(options)

    mcq = MCQ(question=question, options=options, correct_index=options.index(answer))
    return mcq.to_dict()


def shuffle_mcq(item: Any) -> Dict[str, Any]:
    """Xáo lại thứ tự đáp án của một câu, giữ nguyên đáp án đúng."""
    mcq = item if isinstance(item, MCQ) else MCQ.from_dict(item)
    source = item.get("source", "ViT5 + LLM") if isinstance(item, dict) else "ViT5 + LLM"

    indexed = list(enumerate(mcq.options))
    random.shuffle(indexed)

    options = [opt for _, opt in indexed]
    new_index = next(i for i, (orig, _) in enumerate(indexed) if orig == mcq.correct_index)

    shuffled = MCQ(
        question=mcq.question,
        options=options,
        correct_index=new_index,
        explanation=mcq.explanation,
    )
    return shuffled.to_dict(source=source)
