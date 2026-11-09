"""
export_utils.py
───────────────
Xuất đề thi MCQ ra nhiều định dạng, giữ nguyên dấu tiếng Việt.

Định dạng hỗ trợ:
  • Word (.docx)     – export_word_bytes
  • PDF              – export_pdf_bytes   (font Unicode nhúng sẵn trong repo)
  • TXT              – export_txt_bytes
  • JSON             – export_json_bytes  (format chuẩn, import lại được)
  • CSV (Quizizz…)   – export_csv_bytes

Nguyên tắc: đề thi **không** in đáp án trong phần câu hỏi; đáp án nằm ở trang
riêng (Word/PDF) hoặc mục "ĐÁP ÁN" cuối file (TXT).

Dùng:
    from export_utils import export_word_bytes, export_pdf_bytes
    docx_bytes = export_word_bytes(mcq_list, title="Đề kiểm tra")
    pdf_bytes  = export_pdf_bytes(mcq_list, title="Đề kiểm tra")
"""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

from schema import LABELS, MCQ

FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"
DEFAULT_TITLE = "Đề kiểm tra trắc nghiệm"

# (family, file thường, file đậm) – ưu tiên font nhúng để CI / máy lạ luôn có.
BUNDLED_FONTS: Sequence[Tuple[str, str, str]] = (
    ("DejaVu", "DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
)

SYSTEM_FONTS: Sequence[Tuple[str, str, str]] = (
    ("Arial", r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\arialbd.ttf"),
    ("Calibri", r"C:\Windows\Fonts\calibri.ttf", r"C:\Windows\Fonts\calibrib.ttf"),
    ("Times", r"C:\Windows\Fonts\times.ttf", r"C:\Windows\Fonts\timesbd.ttf"),
    ("SegoeUI", r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\segoeuib.ttf"),
    ("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("Liberation", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
     "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
    ("Arial", "/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
    ("DejaVu", "/System/Library/Fonts/Supplemental/DejaVuSans.ttf",
     "/System/Library/Fonts/Supplemental/DejaVuSans-Bold.ttf"),
)


def _as_mcqs(items: Iterable[Any]) -> List[MCQ]:
    """Chuẩn hoá đầu vào: dict (format app/chuẩn) hoặc MCQ → List[MCQ]."""
    mcqs: List[MCQ] = []
    for item in items:
        mcqs.append(item if isinstance(item, MCQ) else MCQ.from_dict(item))
    return mcqs


def font_candidates() -> List[Tuple[str, str, str]]:
    """Font ưu tiên: font nhúng trong repo trước, rồi font hệ thống."""
    bundled = [
        (family, str(FONT_DIR / regular), str(FONT_DIR / bold))
        for family, regular, bold in BUNDLED_FONTS
    ]
    return bundled + list(SYSTEM_FONTS)


def resolve_unicode_font() -> Tuple[str, str, str]:
    """
    Tìm font TTF hỗ trợ tiếng Việt → (family, regular_path, bold_path).

    Raises:
        RuntimeError: khi không có font Unicode nào (báo rõ cách xử lý).
    """
    for family, regular, bold in font_candidates():
        if os.path.exists(regular):
            return family, regular, bold if os.path.exists(bold) else regular
    raise RuntimeError(
        "Không tìm thấy font Unicode để xuất PDF. Cài DejaVu "
        "(Linux: apt install fonts-dejavu-core) hoặc dùng Word/JSON."
    )


# ══════════════════════════════════════════════════════════════
# WORD EXPORT  (python-docx)
# ══════════════════════════════════════════════════════════════
def export_word_bytes(
    mcq_list: Iterable[Any],
    title: str = DEFAULT_TITLE,
    show_answers_in_body: bool = False,
) -> bytes:
    """
    Trả về bytes của file .docx.

    Args:
        mcq_list: list dict (format app/chuẩn) hoặc list :class:`schema.MCQ`.
        title: tiêu đề đề thi.
        show_answers_in_body: True → in đậm + tô xanh đáp án ngay trong phần
            câu hỏi (mặc định False: đề sạch, đáp án ở trang riêng).
    """
    try:
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Inches, Pt, RGBColor
    except ImportError:
        raise RuntimeError("Chạy: pip install python-docx")

    mcqs = _as_mcqs(mcq_list)
    if not mcqs:
        raise ValueError("Danh sách câu hỏi rỗng – không có gì để xuất.")

    doc = Document()

    # ── Tiêu đề ────────────────────────────────────────────────
    heading = doc.add_heading(title, level=1)
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph("")  # khoảng trắng

    # ── Nội dung câu hỏi ───────────────────────────────────────
    for i, mcq in enumerate(mcqs, 1):
        q_para = doc.add_paragraph()
        q_run = q_para.add_run(f"Câu {i}. {mcq.question}")
        q_run.bold = True
        q_run.font.size = Pt(12)

        for j, opt in enumerate(mcq.options):
            lbl = LABELS[j]
            opt_para = doc.add_paragraph(style="List Bullet")
            opt_para.paragraph_format.left_indent = Inches(0.3)
            run = opt_para.add_run(f"{lbl}. {opt}")
            run.font.size = Pt(11)
            if show_answers_in_body and lbl == mcq.correct_label:
                run.bold = True
                run.font.color.rgb = RGBColor(0x16, 0xA3, 0x4A)  # xanh lá

        doc.add_paragraph("")  # khoảng cách giữa các câu

    # ── Đáp án (trang riêng) ────────────────────────────────────
    doc.add_page_break()
    doc.add_heading("ĐÁP ÁN", level=2)
    ans_lines = [f"Câu {i}: {mcq.correct_label}" for i, mcq in enumerate(mcqs, 1)]

    # 3 cột đáp án
    cols = [ans_lines[k::3] for k in range(3)]
    max_rows = max(len(c) for c in cols)
    table = doc.add_table(rows=max_rows, cols=3)
    table.style = "Table Grid"
    for col_idx, col_data in enumerate(cols):
        for row_idx, cell_text in enumerate(col_data):
            table.cell(row_idx, col_idx).text = cell_text

    # ── Serialize ──────────────────────────────────────────────
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════
# PDF EXPORT  (fpdf2 + font Unicode)
# ══════════════════════════════════════════════════════════════
def _register_unicode_font(pdf) -> str:
    """Đăng ký font TTF hỗ trợ tiếng Việt cho fpdf2, trả về tên family."""
    family, regular, bold = resolve_unicode_font()
    pdf.add_font(family, style="", fname=regular)
    pdf.add_font(family, style="B", fname=bold)
    return family


def export_pdf_bytes(
    mcq_list: Iterable[Any],
    title: str = DEFAULT_TITLE,
    show_answers_in_body: bool = False,
) -> bytes:
    """
    Trả về bytes của file .pdf giữ nguyên dấu tiếng Việt.

    Args:
        mcq_list: list dict (format app/chuẩn) hoặc list :class:`schema.MCQ`.
        title: tiêu đề (in ở header mọi trang).
        show_answers_in_body: True → tô xanh đáp án trong phần câu hỏi.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        raise RuntimeError("Chạy: pip install fpdf2")

    mcqs = _as_mcqs(mcq_list)
    if not mcqs:
        raise ValueError("Danh sách câu hỏi rỗng – không có gì để xuất.")

    class QuizPDF(FPDF):
        def header(self) -> None:
            self.set_font(self.quiz_font, "B", 14)
            self.cell(0, 10, self.quiz_title, align="C", new_x="LMARGIN", new_y="NEXT")
            self.ln(4)

        def footer(self) -> None:
            self.set_y(-15)
            self.set_font(self.quiz_font, "", 8)
            self.cell(0, 10, f"Trang {self.page_no()}", align="C")

    pdf = QuizPDF()
    pdf.quiz_font = _register_unicode_font(pdf)
    pdf.quiz_title = title
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    font = pdf.quiz_font

    # ── Phần đề (không kèm đáp án) ─────────────────────────────
    for i, mcq in enumerate(mcqs, 1):
        pdf.set_text_color(0, 0, 0)
        pdf.set_font(font, "B", 12)
        pdf.multi_cell(0, 8, f"Câu {i}. {mcq.question}", new_x="LMARGIN", new_y="NEXT")

        for j, opt in enumerate(mcq.options):
            lbl = LABELS[j]
            if show_answers_in_body and lbl == mcq.correct_label:
                pdf.set_text_color(22, 163, 74)
                pdf.set_font(font, "B", 11)
            else:
                pdf.set_text_color(55, 65, 81)
                pdf.set_font(font, "", 11)
            pdf.multi_cell(0, 7, f"   {lbl}. {opt}", new_x="LMARGIN", new_y="NEXT")

        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)

    # ── Trang đáp án riêng ─────────────────────────────────────
    pdf.add_page()
    pdf.set_text_color(0, 0, 0)
    pdf.set_font(font, "B", 13)
    pdf.cell(0, 10, "ĐÁP ÁN", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font(font, "", 11)
    for i, mcq in enumerate(mcqs, 1):
        pdf.cell(60, 8, f"Câu {i}: {mcq.correct_label}")
        if i % 3 == 0:
            pdf.ln(8)
    pdf.ln(8)

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════
# TXT EXPORT
# ══════════════════════════════════════════════════════════════
def export_txt_bytes(
    mcq_list: Iterable[Any],
    title: str = DEFAULT_TITLE,
    show_answers: bool = True,
) -> bytes:
    """Trả về bytes (.txt, UTF-8) – đề + bảng đáp án ở cuối (nếu show_answers)."""
    mcqs = _as_mcqs(mcq_list)
    if not mcqs:
        raise ValueError("Danh sách câu hỏi rỗng – không có gì để xuất.")

    lines: List[str] = [title, "=" * len(title), ""]

    for i, mcq in enumerate(mcqs, 1):
        lines.append(f"Câu {i}. {mcq.question}")
        for j, opt in enumerate(mcq.options):
            lines.append(f"   {LABELS[j]}. {opt}")
        lines.append("")

    if show_answers:
        lines.append("ĐÁP ÁN")
        lines.append("-" * 6)
        for i, mcq in enumerate(mcqs, 1):
            lines.append(f"Câu {i}: {mcq.correct_label}. {mcq.answer}")
        lines.append("")

    return "\n".join(lines).encode("utf-8")


# ══════════════════════════════════════════════════════════════
# JSON EXPORT  (format chuẩn – import lại / trao đổi)
# ══════════════════════════════════════════════════════════════
def export_json_bytes(mcq_list: Iterable[Any], indent: int = 2) -> bytes:
    """Trả về bytes JSON (UTF-8, không escape tiếng Việt)."""
    mcqs = _as_mcqs(mcq_list)
    if not mcqs:
        raise ValueError("Danh sách câu hỏi rỗng – không có gì để xuất.")

    payload = [mcq.to_canonical_dict() for mcq in mcqs]
    return json.dumps(payload, ensure_ascii=False, indent=indent).encode("utf-8")


# ══════════════════════════════════════════════════════════════
# CSV EXPORT  (template Quizizz / Google Forms)
# ══════════════════════════════════════════════════════════════
def export_csv_bytes(mcq_list: Iterable[Any]) -> bytes:
    """
    Trả về bytes CSV theo template import Quizizz / Google Forms:

        Question,Option A,Option B,Option C,Option D,Correct

    File có BOM UTF-8 để Excel mở đúng dấu tiếng Việt.
    """
    mcqs = _as_mcqs(mcq_list)
    if not mcqs:
        raise ValueError("Danh sách câu hỏi rỗng – không có gì để xuất.")

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(
        ["Question", "Option A", "Option B", "Option C", "Option D", "Correct"]
    )

    for mcq in mcqs:
        options = list(mcq.options)
        if len(options) > 5:
            raise ValueError("CSV chỉ hỗ trợ tối đa 5 lựa chọn (E) mỗi câu.")
        padded = options + [""] * (5 - len(options))
        writer.writerow([mcq.question, *padded[:4], mcq.correct_label])

    return buf.getvalue().encode("utf-8-sig")

