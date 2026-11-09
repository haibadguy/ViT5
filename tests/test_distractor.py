"""Test distractor: parse JSON + chọn backend (không gọi API thật)."""

import pytest
from distractor import DistractorGenerator, _safe_parse_json


def test_parse_valid_json_output():
    raw = '["Huế", "Đà Nẵng", "Hải Phòng"]'

    assert _safe_parse_json(raw, 3, "Hà Nội") == [
        "Huế",
        "Đà Nẵng",
        "Hải Phòng",
    ]


def test_parse_json_removes_answer_and_duplicates():
    raw = '```json\n["Hà Nội", "Huế", "Huế", "Đà Nẵng"]\n```'

    assert _safe_parse_json(raw, 3, "Hà Nội") == ["Huế", "Đà Nẵng"]


def test_parse_removes_answer_with_different_case_and_spacing():
    raw = '[" hà nội ", "Huế", "Đà Nẵng"]'

    assert _safe_parse_json(raw, 3, "Hà Nội") == ["Huế", "Đà Nẵng"]


def test_detect_backend_prefers_env_llm_backend(monkeypatch):
    for name in ("GROQ_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "OLLAMA_HOST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_BACKEND", "groq")

    assert DistractorGenerator._detect_backend() == "groq"


def test_detect_backend_raises_without_any_config(monkeypatch):
    for name in ("GROQ_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "OLLAMA_HOST", "LLM_BACKEND"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(DistractorGenerator, "_ollama_alive", staticmethod(lambda host=None: False))

    with pytest.raises(RuntimeError):
        DistractorGenerator._detect_backend()


def test_model_defaults_to_env_override(monkeypatch):
    """Model đọc từ <BACKEND>_MODEL trong .env (không tạo client thật)."""
    import types

    import distractor as distractor_module

    monkeypatch.setenv("GROQ_API_KEY", "gsk_fakefakefakefakefake")
    monkeypatch.setenv("GROQ_MODEL", "llama-test-model")
    monkeypatch.setattr(
        distractor_module,
        "_OpenAICompatBackend",
        lambda api_key, model, base_url=None: types.SimpleNamespace(model=model),
    )

    gen = DistractorGenerator(backend="groq")

    assert gen.backend_name == "groq"
    assert gen._backend.model == "llama-test-model"

