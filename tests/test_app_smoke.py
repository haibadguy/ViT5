"""Smoke test UI Streamlit: script phải render không lỗi.

Chỉ chạy khi môi trường có `streamlit` (CI nhẹ không cài streamlit → skip).
Không gọi model/API: chỉ dựng trang ban đầu và kiểm tra sidebar.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="cần streamlit để chạy smoke test UI")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = Path(__file__).resolve().parents[1] / "demo_mcq" / "app.py"


def test_app_renders_without_exception():
    at = AppTest.from_file(str(APP_PATH), default_timeout=90)
    at.run()

    assert not at.exception, at.exception


def test_backend_dropdown_offers_all_backends():
    at = AppTest.from_file(str(APP_PATH), default_timeout=90)
    at.run()

    backend_select = at.sidebar.selectbox[0]

    assert backend_select.options == ["auto", "gemini", "openai", "groq", "ollama"]


def test_hf_inference_api_option_is_gone():
    at = AppTest.from_file(str(APP_PATH), default_timeout=90)
    at.run()

    labels = [radio.label for radio in at.sidebar.radio]
    assert "Chế độ" not in labels
