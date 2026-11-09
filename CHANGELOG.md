# Changelog

## [1.0.1] – 2026-11-10

### Added
- `scripts/smoke_stage1.py`: chạy Stage 1 (ViT5 local) → `build_mcq` → validate →
  export Word/PDF, **không cần API key**; `--tiny` dùng model 5 MB để kiểm tra
  code path trong ~10 giây. Đã chạy thật với torch 2.15.0+cpu / transformers 5.22.2.

### Changed
- Ghim version dependency đã kiểm thử (`requirements.txt`, `requirements-dev.txt`);
  `torch` giữ range vì phải chọn CPU/CUDA theo máy. `openai` / `google-genai` giữ
  range vì chưa chạy được trên máy dev (thiếu key).
- README: bỏ hướng dẫn quay GIF, nói rõ "không có key thì app mở được nhưng sinh
  câu hỏi sẽ báo lỗi cấu hình", thêm mục chạy `smoke_stage1.py`.

### Fixed
- Bỏ `.coverage` bị commit nhầm; ignore `.idea/`, `.vscode/`, `coverage.xml`, `htmlcov/`.

## [1.0.0] – 2026-11-10

### Removed
- **HF Inference API path** trong UI và `generator.py` (`use_api`): tham số này bị bỏ qua
  từ trước và các model ViT5 tiếng Việt không khả dụng qua Inference API. Stage 1 giờ
  luôn chạy local. Bớt một đường code chết, UI không còn lựa chọn gây hiểu nhầm.
- Khối CSS + `load_qa_generator` / `load_distractor_generator` bị khai báo trùng ở cuối
  `demo_mcq/app.py`.

### Added
- `demo_mcq/schema.py`: dataclass `MCQ` + `validate()` + `validate_batch()` + `summarize()`
  + `parse_mcq_list()`. Generator/UI loại câu hỏng và hiển thị lý do
  ("Sinh 10 câu, giữ 8 câu hợp lệ…").
- `demo_mcq/export_utils.py`: `export_txt_bytes`, `export_json_bytes`, `export_csv_bytes`
  (template Quizizz / Google Forms) và tham số `show_answers_in_body`.
- `tests/` mở rộng lên 36 test, chạy offline (mock model, không gọi API).
- `pyproject.toml`: cấu hình ruff + pytest cho CI.
- `demo_mcq/assets/fonts/DejaVuSans{,-Bold}.ttf` (giấy phép Bitstream Vera) để PDF luôn
  có font Unicode, kể cả trong CI.

### Fixed
- **PDF mất dấu tiếng Việt**: tiêu đề/đáp án bị hard-code `"Cau"`, `"DAP AN"` và fallback
  về Helvetica. Đổi sang `"Câu"`, `"ĐÁP ÁN"` và luôn đăng ký font Unicode (bundled +
  hệ thống), báo lỗi rõ nếu không tìm thấy font.
- Đề xuất ra **không còn in đáp án** trong phần câu hỏi; đáp án nằm ở trang riêng
  (`show_answers_in_body=False` mặc định).
- UI thiếu backend **Groq**; giờ có đủ `auto | gemini | openai | groq | ollama`, đọc
  `LLM_BACKEND` và `<BACKEND>_MODEL` từ `.env`.
- **API key không còn bị in ra UI**: ô nhập để trống sẽ dùng key trong `.env` (hiển thị
  dạng mask `AIza…9f2c`); lỗi backend đi qua `friendly_llm_error()` + `scrub_secrets()`.
- `SyntaxWarning` trong `plms/utils.py` (chuỗi `'\[SEP\]'` → raw string).
