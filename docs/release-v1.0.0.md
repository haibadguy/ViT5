# ViT5 Quiz Generator v1.0.0

Sinh câu hỏi trắc nghiệm tiếng Việt từ văn bản, dùng ViT5 cho câu hỏi/đáp án đúng
và LLM cho đáp án nhiễu.

## Tính năng

- **Pipeline 2 stage**: ViT5 (`shnl/vit5-vinewsqa-qg-ae`) sinh câu hỏi + đáp án
  đúng, LLM sinh đáp án nhiễu, sau đó chuẩn hoá thành MCQ A/B/C/D.
- **4 backend LLM**: Gemini, OpenAI, Groq, Ollama (chọn qua `LLM_BACKEND` hoặc UI).
- **Validate câu hỏi**: câu thiếu/trùng lựa chọn, sai vị trí đáp án bị loại kèm lý do
  (`Sinh 10 câu, giữ 8 câu hợp lệ…`).
- **Export**: Word / PDF (font Unicode nhúng, đáp án ở trang riêng) / TXT / JSON /
  CSV (Quizizz, Google Forms) — giữ nguyên dấu tiếng Việt.
- **Bảo mật**: API key không in ra UI/log; lỗi 401/429/timeout được diễn giải thân thiện.

## Kiểm thử

- 49 test, chạy **offline** (mock model, không gọi API).
- CI: `ruff check .` + `pytest --cov` trên Python 3.10 và 3.12.

## Chạy thử

```bash
pip install -r demo_mcq/requirements.txt
copy demo_mcq\.env.example demo_mcq\.env    # điền 1 API key
streamlit run demo_mcq/app.py
```

Chi tiết: [README.md](../README.md) · Thay đổi: [CHANGELOG.md](../CHANGELOG.md)
