"""
smoke_stage1.py – chạy THẬT Stage 1 (ViT5 local) rồi build/validate/export.

Không cần API key: chỉ dùng Stage 1 + distractor cố định, để kiểm tra
torch/transformers + model + export có chạy trên máy bạn hay không.

Dùng:
    python scripts/smoke_stage1.py --tiny   # model 5 MB, kiểm tra code path (nhanh)
    python scripts/smoke_stage1.py          # model thật (~1 GB, lần đầu tải lâu)

Ghi chú: model tiny có trọng số ngẫu nhiên nên câu hỏi sinh ra là rác —
mục đích chỉ là xác nhận pipeline không crash và export hoạt động.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo_mcq"))

TINY_MODEL = "hf-internal-testing/tiny-random-t5"
DEFAULT_MODEL = "shnl/vit5-vinewsqa-qg-ae"

CONTEXT = (
    "Hà Nội là thủ đô của nước Cộng hòa Xã hội chủ nghĩa Việt Nam. "
    "Thành phố có lịch sử hơn 1000 năm và từng được gọi là Thăng Long dưới triều đại nhà Lý. "
    "Sông Hồng chảy qua phía đông bắc thành phố, tạo nên vùng đất màu mỡ phù sa. "
    "Dân số Hà Nội hiện nay khoảng 8,4 triệu người."
)

DUMMY_DISTRACTORS = ["Huế", "Đà Nẵng", "Hải Phòng"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Stage 1 (ViT5)")
    parser.add_argument("--tiny", action="store_true", help="dùng model 5 MB thay vì model thật")
    parser.add_argument("--model", default="", help="tên model HF tuỳ chọn")
    parser.add_argument("--num-pairs", type=int, default=3, help="số cặp Q-A muốn sinh")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs"), help="nơi ghi file export")
    args = parser.parse_args()

    model_name = args.model or (TINY_MODEL if args.tiny else DEFAULT_MODEL)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from generator import QAGenerator
    from schema import build_mcq, validate_batch

    print(f"[1/4] Load model: {model_name}")
    started = time.time()
    generator = QAGenerator(model_name=model_name)
    print(f"      xong sau {time.time() - started:.1f}s")

    print(f"[2/4] Sinh câu hỏi (num_pairs={args.num_pairs})")
    t1 = time.time()
    pairs = generator.generate(CONTEXT, num_pairs=args.num_pairs)
    print(f"      {len(pairs)} cặp Q-A trong {time.time() - t1:.1f}s")
    for pair in pairs:
        print(f"      Q: {pair['question'][:80]}")
        print(f"      A: {pair['answer']}")

    if not pairs:
        print("      ⚠️ Không sinh được cặp nào — kiểm tra model/context.")
        return 1

    print("[3/4] Ghép MCQ + validate")
    mcqs = [
        build_mcq(pair["question"], pair["answer"], DUMMY_DISTRACTORS)
        for pair in pairs
    ]
    kept, rejected = validate_batch(mcqs)
    print(f"      giữ {len(kept)} / loại {len(rejected)}")
    for item in rejected:
        print(f"      loại: {item.question[:60]!r} → {', '.join(item.reasons)}")

    print("[4/4] Export Word + PDF")
    from export_utils import export_pdf_bytes, export_word_bytes

    docx_path = out_dir / "smoke_stage1.docx"
    pdf_path = out_dir / "smoke_stage1.pdf"
    docx_path.write_bytes(export_word_bytes(kept, title="Smoke test Stage 1"))
    pdf_path.write_bytes(export_pdf_bytes(kept, title="Smoke test Stage 1"))
    print(f"      {docx_path}")
    print(f"      {pdf_path}")

    print("OK – Stage 1 + validate + export chạy được trên máy này.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
