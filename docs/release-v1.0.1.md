# ViT5 Quiz Generator v1.0.1

Bản vá nhỏ: ghim dependency và thêm script tự kiểm tra Stage 1 (không cần API key).

## Thay đổi

- **Ghim dependency** trong `demo_mcq/requirements.txt` và `requirements-dev.txt`
  theo đúng version đã chạy xanh: streamlit 1.67.0, transformers 5.22.2,
  sentencepiece 0.2.3, accelerate 1.15.0, python-docx 1.2.0, fpdf2 2.9.0,
  pytest 9.2.0, pytest-cov 7.1.0, ruff 0.17.4. `torch` vẫn để range vì phải chọn
  bản CPU/CUDA theo máy.
- **`scripts/smoke_stage1.py`**: chạy Stage 1 (ViT5 local) → ghép MCQ → validate →
  export Word/PDF vào `outputs/`, không cần key. `--tiny` dùng model 5 MB để kiểm tra
  code path trong ~10 giây.
- Dọn repo: bỏ `.coverage` bị commit nhầm, ignore `.idea/` + `.vscode/`, xoá file
  hướng dẫn quay GIF (không dùng).

## Kiểm thử

- 49 test offline, CI xanh trên Python 3.10 và 3.12.
- `scripts/smoke_stage1.py --tiny` đã chạy thật với torch 2.15.0+cpu /
  transformers 5.22.2: load → sinh Q-A → validate → export, dấu tiếng Việt giữ nguyên.

Cài đặt và giới hạn: xem [README](../README.md).
