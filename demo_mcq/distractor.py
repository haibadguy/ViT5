"""
distractor.py
─────────────
Sinh 3 đáp án sai (distractors) cho một cặp Q-A tiếng Việt bằng LLM.

Hỗ trợ 3 backend:
  • openai   – OpenAI API (gpt-4o-mini, gpt-3.5-turbo, …)
  • gemini   – Google Gemini API (gemini-1.5-flash, …)
  • ollama   – Ollama chạy local (llama3, qwen2, mistral, …)

Auto-detect backend theo biến môi trường:
  OPENAI_API_KEY  → openai
  GEMINI_API_KEY  → gemini
  OLLAMA_HOST     → ollama  (mặc định http://localhost:11434)

Dùng:
    from distractor import DistractorGenerator
    gen = DistractorGenerator()         # auto-detect
    distractors = gen.generate(question, answer, context)
    # ["Huế", "Đà Nẵng", "Hải Phòng"]
"""

import os
import re
import json
import time
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────── prompt ─────────────────────────────
def _build_prompt(question: str, answer: str, context: str, n: int) -> str:
    ctx_section = f"\nContext:\n{context}\n" if context else ""
    return f"""Bạn là chuyên gia thiết kế đề thi trắc nghiệm tiếng Việt có kinh nghiệm cao.

Câu hỏi: {question}
Đáp án đúng: {answer}{ctx_section}
Hãy tạo **chính xác {n} đáp án sai** (distractors) chất lượng cao theo các tiêu chí nghiêm ngặt sau:
1. Thuộc cùng phạm trù/loại thực thể với đáp án đúng (ví dụ: cùng là địa danh, cùng là năm tháng, cùng là tên người, cùng là khái niệm khoa học…).
2. Hợp lý, gần giống về ngữ nghĩa và có sức nhiễu cao (plausible distractors) nhưng chắc chắn sai.
3. Ngắn gọn, tự nhiên, đúng ngữ pháp tiếng Việt.
4. Không trùng hoặc gần trùng với đáp án đúng và không lặp lại lẫn nhau.

**Quy tắc output cực kỳ nghiêm ngặt**:
- CHỈ trả về đúng **một mảng JSON** chứa chính xác {n} chuỗi string.
- Không được thêm bất kỳ chữ nào khác (không giải thích, không số thứ tự, không "Dưới đây là...", không markdown).
- JSON phải hợp lệ 100%.

Ví dụ output hợp lệ:
["Huế", "Đà Nẵng", "Hải Phòng"]

Bây giờ hãy tạo distractors ngay:"""


# ─────────────────────────── helpers ────────────────────────────
def _safe_parse_json(text: str, n: int, answer: str) -> List[str]:
    """Parse JSON array từ response LLM, có fallback mạnh."""
    # Strip markdown fences nếu có
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()

    # Thử parse trực tiếp
    try:
        data = json.loads(text)
        if isinstance(data, list):
            result = [str(d).strip() for d in data if str(d).strip()]
            return _deduplicate(result, answer, n)
    except json.JSONDecodeError:
        pass

    # Thử tìm array trong text
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                result = [str(d).strip() for d in data if str(d).strip()]
                return _deduplicate(result, answer, n)
        except json.JSONDecodeError:
            pass

    # Fallback: tách dòng hoặc dấu phẩy
    candidates = re.split(r'[\n,;]+', text)
    result = []
    for c in candidates:
        c = re.sub(r'^[\d\.\-\*\s"\']+', '', c).strip().strip('"\'')
        if c and len(c) > 1:
            result.append(c)

    return _deduplicate(result, answer, n)


def _deduplicate(items: List[str], answer: str, n: int) -> List[str]:
    """Loại trùng lặp và loại item giống đáp án đúng."""
    seen   = set()
    result = []
    answer_norm = answer.lower().strip()
    for item in items:
        norm = item.lower().strip()
        if norm not in seen and norm != answer_norm:
            seen.add(norm)
            result.append(item)
        if len(result) == n:
            break
    return result


# ─────────────────────────── backends ───────────────────────────

class _OpenAIBackend:
    def __init__(self, api_key: str, model: str):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("Chạy: pip install openai")
        self.client = OpenAI(api_key=api_key)
        self.model  = model

    def complete(self, prompt: str) -> str:
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=120,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    time.sleep(3 * (attempt + 1))
                else:
                    raise
        raise RuntimeError("OpenAI API rate limit – thử lại sau.")


class _GeminiBackend:
    def __init__(self, api_key: str, model: str):
        try:
            import google.generativeai as genai
        except ImportError:
            raise RuntimeError("Chạy: pip install google-generativeai")
        genai.configure(api_key=api_key)
        # response_mime_type="application/json" buộc Gemini trả về JSON hợp lệ
        # → loại bỏ hoàn toàn lỗi JSONDecodeError
        self.client = genai.GenerativeModel(
            model,
            generation_config={"response_mime_type": "application/json"},
        )
        self.model = model

    def complete(self, prompt: str) -> str:
        for attempt in range(3):
            try:
                resp = self.client.generate_content(prompt)
                return resp.text or ""
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    time.sleep(4 * (attempt + 1))
                else:
                    raise
        raise RuntimeError("Gemini API rate limit – thử lại sau.")


class _OllamaBackend:
    def __init__(self, host: str, model: str):
        self.host  = host.rstrip("/")
        self.model = model

    def complete(self, prompt: str) -> str:
        import requests
        url  = f"{self.host}/api/generate"
        body = {"model": self.model, "prompt": prompt, "stream": False}
        resp = requests.post(url, json=body, timeout=120)
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama lỗi {resp.status_code}: {resp.text[:200]}")
        return resp.json().get("response", "")


# ─────────────────────────── class chính ────────────────────────

class DistractorGenerator:
    """
    Sinh distractors bằng LLM.

    Tham số:
        backend     : "openai" | "gemini" | "ollama" | None (auto-detect).
        model       : Tên model (mặc định theo từng backend).
        api_key     : API key (tự đọc từ env nếu không truyền).
        ollama_host : URL Ollama server (mặc định http://localhost:11434).
    """

    DEFAULTS = {
        "openai": "gpt-4o-mini",
        "gemini": "gemini-1.5-flash",
        "ollama": "llama3",
    }

    def __init__(
        self,
        backend: Optional[str] = None,
        model:   Optional[str] = None,
        api_key: Optional[str] = None,
        ollama_host: str       = "http://localhost:11434",
    ):
        backend = backend or self._detect_backend()
        model   = model   or self.DEFAULTS.get(backend, "gpt-4o-mini")
        self.backend_name = backend

        if backend == "openai":
            key = api_key or os.getenv("OPENAI_API_KEY", "")
            if not key:
                raise ValueError("Cần OPENAI_API_KEY. Thêm vào .env hoặc truyền api_key=…")
            self._backend = _OpenAIBackend(key, model)

        elif backend == "gemini":
            key = api_key or os.getenv("GEMINI_API_KEY", "")
            if not key:
                raise ValueError("Cần GEMINI_API_KEY. Thêm vào .env hoặc truyền api_key=…")
            self._backend = _GeminiBackend(key, model)

        elif backend == "ollama":
            host = os.getenv("OLLAMA_HOST", ollama_host)
            self._backend = _OllamaBackend(host, model)

        else:
            raise ValueError(
                f"Backend '{backend}' không hợp lệ. Chọn: openai | gemini | ollama"
            )

        print(f"[Distractor] Backend: {backend} / model: {model}")

    @staticmethod
    def _detect_backend() -> str:
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("GEMINI_API_KEY"):
            return "gemini"
        if os.getenv("OLLAMA_HOST") or DistractorGenerator._ollama_alive():
            return "ollama"
        raise RuntimeError(
            "Không tìm thấy LLM backend.\n"
            "Vui lòng đặt một trong: OPENAI_API_KEY / GEMINI_API_KEY / OLLAMA_HOST trong file .env"
        )

    @staticmethod
    def _ollama_alive(host: str = "http://localhost:11434") -> bool:
        try:
            import requests
            r = requests.get(f"{host}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        question: str,
        answer: str,
        context: str = "",
        num_distractors: int = 3,
    ) -> List[str]:
        """
        Sinh `num_distractors` đáp án sai cho cặp (question, answer).

        Returns:
            List[str] – ví dụ ["Huế", "Đà Nẵng", "Hải Phòng"]
        """
        prompt   = _build_prompt(question, answer, context, num_distractors)
        raw_text = self._backend.complete(prompt)
        return _safe_parse_json(raw_text, num_distractors, answer)
