"""
generator.py
────────────
Sinh cặp Question–Answer từ đoạn văn tiếng Việt bằng ViT5-qag (ViQAG).

Hai chế độ:
  1. LOCAL  – load model từ Hugging Face Hub về máy (cần ~2 GB RAM/VRAM).
  2. API    – gọi Hugging Face Inference API (không cần GPU, cần HF token).

Dùng:
    from generator import QAGenerator
    gen = QAGenerator()                       # auto-detect chế độ
    pairs = gen.generate("Hà Nội là thủ đô…")
    # [{"question": "…", "answer": "…"}, …]
"""

import os
import re
import sys
import json
import logging
import time
from typing import List, Dict

logging.basicConfig(level=logging.WARNING)

# ─────────────────────────── hằng số ───────────────────────────
DEFAULT_MODEL     = "VietAI/vit5-base-vi-qag"   # fine-tuned cho tiếng Việt QAG
FALLBACK_MODEL    = "VietAI/vit5-base"            # base (thêm prefix thủ công)
MAX_INPUT_TOKENS  = 512
MAX_OUTPUT_TOKENS = 256
TASK_PREFIX       = "generate question and answer: "

# ─────────────────────────── helpers ────────────────────────────
def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _parse_qa_string(raw: str) -> List[Dict[str, str]]:
    """
    ViT5-qag trả về chuỗi dạng:
        "question: Câu hỏi 1?, answer: Đáp án 1 [SEP] question: Câu hỏi 2?, answer: Đáp án 2"
    Hàm này phân tích chuỗi đó thành list[{question, answer}].
    """
    pairs = []
    for chunk in raw.split("[SEP]"):
        chunk = chunk.strip()
        if not chunk:
            continue
        q_match = re.search(r"question:\s*(.+?)(?:,\s*answer:|$)", chunk, re.DOTALL)
        a_match = re.search(r"answer:\s*(.+)$",                   chunk, re.DOTALL)
        if q_match and a_match:
            q = _clean(q_match.group(1))
            a = _clean(a_match.group(1))
            if q and a:
                pairs.append({"question": q, "answer": a})
    return pairs


# ─────────────────────────── class chính ────────────────────────
class QAGenerator:
    """
    Sinh Question-Answer từ context tiếng Việt.

    Tham số:
        model_name  : Tên model HF hub (mặc định VietAI/vit5-base-vi-qag).
        use_api     : True  → dùng HF Inference API.
                      False → load model local (or auto-detect nếu None).
        hf_token    : HF token (cần khi use_api=True hoặc model private).
        device      : "cpu" | "cuda" | "auto" (default "auto").
    """

    def __init__(
        self,
        model_name: str = None,
        use_api: bool = None,
        hf_token: str = None,
        device: str = "auto",
    ):
        self.model_name = model_name or os.getenv("VIQAG_MODEL", DEFAULT_MODEL)
        self.hf_token   = hf_token   or os.getenv("HF_TOKEN", "")
        self.device     = device

        # auto-detect: nếu có HF_TOKEN → API, nếu không → local
        if use_api is None:
            use_api = bool(self.hf_token)

        self.use_api = use_api
        self._model     = None
        self._tokenizer = None

        if not self.use_api:
            self._load_local()

    # ── local ──────────────────────────────────────────────────
    def _load_local(self):
        """Load ViT5 model về máy (lazy, chỉ load 1 lần)."""
        try:
            import torch
            from transformers import AutoTokenizer, T5ForConditionalGeneration
        except ImportError:
            raise RuntimeError(
                "Thiếu thư viện. Chạy:  pip install torch transformers sentencepiece"
            )

        print(f"[Generator] Đang load model '{self.model_name}' (lần đầu ~1 phút)…")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_auth_token=self.hf_token or None,
            legacy=False,
        )

        try:
            self._model = T5ForConditionalGeneration.from_pretrained(
                self.model_name,
                use_auth_token=self.hf_token or None,
            )
        except Exception:
            # fallback nếu model không phải T5 strict
            from transformers import AutoModelForSeq2SeqLM
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                use_auth_token=self.hf_token or None,
            )

        import torch
        if self.device == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            dev = self.device

        self._model.to(dev)
        self._model.eval()
        self._device = dev
        print(f"[Generator] Model đã sẵn sàng trên '{dev}'.")

    def _generate_local(self, context: str) -> str:
        import torch
        # ViT5-qag đã fine-tuned: thêm task prefix
        prompt = TASK_PREFIX + context[:MAX_INPUT_TOKENS * 3]  # rough char limit
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_INPUT_TOKENS,
            truncation=True,
        ).to(self._device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_length=MAX_OUTPUT_TOKENS,
                num_beams=4,
                early_stopping=True,
            )
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # ── HF Inference API ───────────────────────────────────────
    def _generate_api(self, context: str, retries: int = 3) -> str:
        """Gọi Hugging Face Inference API."""
        import requests

        prompt = TASK_PREFIX + context[:1500]
        url    = f"https://api-inference.huggingface.co/models/{self.model_name}"
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": MAX_OUTPUT_TOKENS,
                "num_beams": 4,
            },
            "options": {"wait_for_model": True},
        }

        for attempt in range(retries):
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and data:
                    return data[0].get("generated_text", "")
                return ""
            elif resp.status_code == 503:
                wait = 20 * (attempt + 1)
                print(f"[Generator] Model đang khởi động, chờ {wait}s…")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"HF API lỗi {resp.status_code}: {resp.text[:200]}"
                )
        raise RuntimeError("HF API không phản hồi sau nhiều lần thử.")

    # ── public API ─────────────────────────────────────────────
    def generate(
        self,
        context: str,
        num_pairs: int = 5,
    ) -> List[Dict[str, str]]:
        """
        Sinh tối đa `num_pairs` cặp Q-A từ `context`.

        Returns:
            [{"question": "…", "answer": "…"}, …]
        """
        context = _clean(context)
        if not context:
            return []

        if self.use_api:
            raw = self._generate_api(context)
        else:
            raw = self._generate_local(context)

        pairs = _parse_qa_string(raw)

        # Lọc các cặp có answer rỗng hoặc không nằm trong context
        valid = []
        for p in pairs:
            q, a = p["question"], p["answer"]
            if a and a.lower() in context.lower():
                valid.append(p)

        # Nếu không parse được (model trả raw text), thử lấy toàn bộ
        if not valid:
            valid = pairs  # trả về nguyên raw parse, không lọc ketat

        return valid[:num_pairs]
