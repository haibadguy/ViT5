# GIF demo – kịch bản quay (20 giây)

Mục tiêu: người xem hiểu project làm gì trong **5 giây đầu**. Không cần đẹp, cần rõ.

## Chuẩn bị (1 lần)

```powershell
winget install --id NickeManarin.ScreenToGif -e
cd D:\new\ViT5
copy demo_mcq\.env.example demo_mcq\.env    # rồi điền 1 key (Gemini free tier là nhanh nhất)
streamlit run demo_mcq/app.py
```

Trước khi quay:
- [ ] Cửa sổ trình duyệt ~1280×800, zoom 100%, đã tắt bookmark bar.
- [ ] **Không mở** tab/file nào chứa `.env` (key sẽ lộ trong GIF).
- [ ] Dán sẵn đoạn văn mẫu vào clipboard (lấy đoạn "Hà Nội là thủ đô…" trong app, nút 🎲 Ví dụ).
- [ ] Đã bấm thử 1 lần để model ViT5 tải xong (lần đầu tải ~1 GB → GIF sẽ đứng hình 1–2 phút).

## Kịch bản 20 giây

| Giây | Thao tác | Điểm cần thấy |
|---|---|---|
| 0–2 | Trang chủ app | Tiêu đề + sidebar "Stage 1 / Stage 2" |
| 2–6 | Dán đoạn văn 150–200 từ vào ô văn bản | Văn bản tiếng Việt có dấu |
| 6–8 | Sidebar: chọn backend **Gemini**, số câu 5 | Dropdown có đủ 4 backend |
| 8–14 | Bấm **Sinh câu hỏi trắc nghiệm**, thanh Stage 1 → Stage 2 chạy | Progress + "✅ Stage 1 hoàn tất" |
| 14–18 | Cuộn danh sách 2–3 câu hỏi | Câu hỏi + 4 lựa chọn |
| 18–20 | Bấm **Word (.docx)** để tải | Nút download |

## Xuất file

- ScreenToGif → **Save as** → GIF → lưu `D:\new\ViT5\docs\demo.gif`.
- Giới hạn: **≤ 5 MB**, 10–12 fps, rộng ≤ 1000 px (dùng tab *Image → Resize*).
- Nếu > 5 MB: cắt bớt phần cuối, giảm fps xuống 8, hoặc giảm chiều rộng 800 px.

## Nhúng vào README

Thêm ngay dưới badge (dòng ~5 của `README.md`):

```markdown
![demo](docs/demo.gif)
```

## Kiểm tra trước khi commit

- [ ] Mở `docs/demo.gif` bằng trình duyệt/Explorer → chạy mượt.
- [ ] Xem lại từng khung hình: **không có** chuỗi `AIza…`, `sk-…`, `gsk_…`, `hf_…`.
- [ ] Kích thước ≤ 5 MB (`(Get-Item docs\demo.gif).Length`).
- [ ] Nếu sau này quay lại GIF mới: **đổi tên file** (`demo-v2.gif`) — GitHub cache ảnh theo URL.
