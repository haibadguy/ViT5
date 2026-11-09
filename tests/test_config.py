"""Test cấu hình: `.env.example` phải khớp với code (chặn drift tài liệu).

Test "tài liệu" nhưng có giá trị thật: nếu ai thêm backend mới mà quên cập nhật
`.env.example`, hoặc đổi tên biến trong code mà quên sửa mẫu, CI sẽ đỏ.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_EXAMPLE = REPO_ROOT / "demo_mcq" / ".env.example"

# Biến bắt buộc phải có trong .env.example để cả 4 backend chạy được
REQUIRED_KEYS = [
    "VIQAG_MODEL",
    "HF_TOKEN",
    "LLM_BACKEND",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "OLLAMA_HOST",
    "OLLAMA_MODEL",
]


def read_env_example() -> str:
    assert ENV_EXAMPLE.exists(), "Thiếu demo_mcq/.env.example"
    return ENV_EXAMPLE.read_text(encoding="utf-8")


def test_env_example_documents_all_backends():
    content = read_env_example()

    for key in REQUIRED_KEYS:
        assert re.search(rf"^{key}=", content, re.MULTILINE), f"Thiếu `{key}=` trong .env.example"


def test_env_example_has_no_real_secrets():
    """Mẫu không được chứa giá trị giống key thật."""
    content = read_env_example()

    assert not re.search(r"AIza[0-9A-Za-z_\-]{10,}", content)
    assert not re.search(r"sk-[A-Za-z0-9]{20,}", content)
    assert not re.search(r"gsk_[A-Za-z0-9]{20,}", content)


def test_env_example_backends_match_code():
    """Mọi backend trong LLM_BACKEND comment phải được code hỗ trợ."""
    distractor_src = (REPO_ROOT / "demo_mcq" / "distractor.py").read_text(encoding="utf-8")

    for backend in ("gemini", "openai", "groq", "ollama"):
        assert f'"{backend}"' in distractor_src, f"distractor.py không hỗ trợ {backend}"


def test_ollama_host_env_variants_are_supported():
    """Cả OLLAMA_HOST và OLLAMA_BASE_URL đều phải được đọc."""
    distractor_src = (REPO_ROOT / "demo_mcq" / "distractor.py").read_text(encoding="utf-8")

    assert "OLLAMA_HOST" in distractor_src
    assert "OLLAMA_BASE_URL" in distractor_src


def test_env_files_are_gitignored():
    """.env thật không bao giờ được commit."""
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert ".env" in gitignore
