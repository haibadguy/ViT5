# Changelog

## [Unreleased] – polish for CV

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
