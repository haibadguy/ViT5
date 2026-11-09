# ViT5 Quiz Generator v1.0.1

Bản vá nhỏ: thêm script tự kiểm tra Stage 1 (không cần API key) và dọn repo.

## Thay đổi

- **`scripts/smoke_stage1.py`**: chạy Stage 1 (ViT5 local) → ghép MCQ → validate →
  export Word/PDF vào `outputs/`, không cần key. `--tiny` dùng model 5 MB để kiểm tra
  code path trong ~10 giây; bỏ `--tiny` để chạy model thật.
- Dọn repo: bỏ `.coverage` bị commit nhầm, ignore `.idea/` + `.vscode/` +
  `coverage.xml` + `htmlcov/`, xoá file hướng dẫn quay GIF (không dùng).
- `requirements*.txt`: giữ **range** thay vì ghim version. (Đã thử ghim và CI đỏ ở
  bước install vì bản ghim không có trên PyPI của runner — ghi lại để không lặp lại.)

## Kiểm thử

- 49 test offline; CI xanh trên Python 3.10 và 3.12.
- `scripts/smoke_stage1.py --tiny` đã chạy thật: load model → sinh Q-A → validate →
  export 2 trang, dấu tiếng Việt giữ nguyên (đọc lại bằng pypdf).

Cài đặt và giới hạn: xem [README](../README.md).
