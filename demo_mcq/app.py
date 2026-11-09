"""
app.py  –  Giao diện Giáo Viên – Sinh MCQ Tiếng Việt
─────────────────────────────────────────────────────
Pipeline: Văn bản → ViQAG (ViT5) → Q+A → LLM → Distractors → MCQ

Chạy:
    streamlit run app.py
"""

import copy
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

# Module nội bộ nằm cùng thư mục demo_mcq/ (Streamlit thêm thư mục script vào sys.path)
from schema import build_mcq as _build_mcq
from schema import shuffle_mcq as _shuffle_mcq

# ── load .env ──────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")

# ══════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MCQ Generator – Công cụ soạn đề trắc nghiệm",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── tổng thể ── */
[data-testid="stAppViewContainer"] { background: #f7f9fc; }
[data-testid="stSidebar"]          { background: #ffffff; border-right: 1px solid #e5e7eb; }

/* ── card câu hỏi ── */
.mcq-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #3b82f6;
    border-radius: 10px;
    padding: 18px 22px 14px 20px;
    margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.mcq-card.selected { border-left-color: #10b981; }
.mcq-q  { font-size: 1.05rem; font-weight: 700; color: #1e293b; margin-bottom: 10px; }
.opt    { font-size: .97rem; padding: 3px 0; color: #374151; }
.opt-correct { color: #16a34a; font-weight: 700; }

/* ── stage badge ── */
.badge {
    display: inline-block; padding: 2px 10px;
    border-radius: 20px; font-size: .78rem; font-weight: 600;
    margin-left: 6px;
}
.badge-viqag  { background:#dbeafe; color:#1d4ed8; }
.badge-llm    { background:#dcfce7; color:#15803d; }
.badge-done   { background:#fef9c3; color:#92400e; }

/* ── preview panel ── */
.preview-box {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-radius: 8px; padding: 14px 16px; font-size: .9rem;
    white-space: pre-wrap; font-family: monospace;
}

/* ── stage info ── */
.stage-info {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-radius: 8px; padding: 10px 14px;
    font-size: .88rem; color: #1e40af; margin-bottom: 12px;
}

/* ── button full-width ── */
div[data-testid="stButton"] > button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# Constants & helpers
# ══════════════════════════════════════════════════════════════
LABELS  = ["A", "B", "C", "D", "E"]
EXAMPLE = (
    "Hà Nội là thủ đô của nước Cộng hòa Xã hội chủ nghĩa Việt Nam. "
    "Đây là thành phố lớn thứ hai cả nước về dân số sau Thành phố Hồ Chí Minh. "
    "Hà Nội nằm ở vùng đồng bằng sông Hồng, cách bờ biển khoảng 120 km về phía tây. "
    "Thành phố có lịch sử hơn 1000 năm và đã từng được gọi là Thăng Long dưới triều đại nhà Lý. "
    "Dân số Hà Nội hiện nay khoảng 8,4 triệu người. "
    "Hà Nội có nhiều trường đại học lớn như Đại học Quốc gia Hà Nội, Đại học Bách Khoa. "
    "Sông Hồng chảy qua phía đông bắc thành phố, tạo nên vùng đất màu mỡ phù sa."
)

_EXAMPLE_TOPICS = [
    "lịch sử Việt Nam", "địa lý Việt Nam", "văn hóa ẩm thực Việt Nam",
    "khoa học vũ trụ", "môi trường và biến đổi khí hậu", "công nghệ trí tuệ nhân tạo",
    "y học và sức khỏe cộng đồng", "kinh tế Việt Nam", "văn học dân gian Việt Nam",
    "sinh học đại cương", "lịch sử thế giới", "địa lý tự nhiên",
]

def _generate_example_paragraph() -> str:
    """Sinh đoạn văn ví dụ bằng OpenAI (ưu tiên) để giữ Gemini quota cho distractor.
    Fallback về đoạn cứng nếu không có API key hoặc lỗi."""
    topic  = random.choice(_EXAMPLE_TOPICS)
    prompt = (
        f"Hãy viết một đoạn văn tiếng Việt từ 150 đến 250 từ về chủ đề: {topic}. "
        "Đoạn văn cần có nhiều dữ kiện cụ thể (tên, số liệu, địa danh, khái niệm) "
        "để có thể dùng làm ngữ liệu sinh câu hỏi trắc nghiệm. "
        "Chỉ trả về đoạn văn, không có tiêu đề, không giải thích."
    )

    # ── Thử OpenAI trước (không tốn Gemini quota) ──────────────
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp   = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=400,
            )
            text = (resp.choices[0].message.content or "").strip()
            if len(text) > 50:
                return text
        except Exception:
            pass

    # ── Không có OpenAI → dùng đoạn cứng (giữ Gemini cho distractor) ──
    return EXAMPLE

def _init_state():
    defaults = {
        "mcq_list":    [],      # List[Dict] – danh sách câu sau edit
        "history":     [],      # List[List[Dict]] – lịch sử 10 lần gần nhất
        "context_buf": "",      # context hiện tại
        "selected":    set(),   # index câu được chọn để export
        "regen_idx":   None,    # index câu cần regenerate
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ══════════════════════════════════════════════════════════════
# Bảo vệ API key + lỗi thân thiện
# ══════════════════════════════════════════════════════════════
LLM_BACKENDS = ["auto", "gemini", "openai", "groq", "ollama"]
BACKEND_ENV_KEYS = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "groq":   "GROQ_API_KEY",
}

# Bắt các chuỗi giống API key để không bao giờ hiển thị ra UI / log
_KEY_PATTERN = re.compile(
    r"(AIza[0-9A-Za-z_\-]{8,}|sk-[A-Za-z0-9_\-]{8,}|gsk_[A-Za-z0-9_\-]{8,}|hf_[A-Za-z0-9]{8,})"
)


def mask_key(key: Optional[str]) -> str:
    """Che key khi hiển thị: 'AIza…9f2c'. Không bao giờ trả về full key."""
    if not key:
        return "(chưa đặt)"
    key = key.strip()
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}…{key[-4:]}"


def scrub_secrets(text: str) -> str:
    """Xoá mọi chuỗi giống API key khỏi message trước khi hiển thị."""
    return _KEY_PATTERN.sub("***", str(text or ""))


def friendly_llm_error(exc: Exception) -> str:
    """Đổi lỗi backend thành message dễ hiểu, đã scrub key."""
    raw = scrub_secrets(exc)
    low = raw.lower()

    if any(k in low for k in ("invalid api key", "api key not valid", "unauthenticated",
                              "401", "permission_denied", "invalid_api_key")):
        return "API key không hợp lệ hoặc đã bị thu hồi (401). Kiểm tra lại key trong .env rồi thử lại."
    if any(k in low for k in ("429", "quota", "resource_exhausted", "rate limit",
                              "rate_limit", "too many requests")):
        return (
            "Backend đang giới hạn tốc độ hoặc hết quota (429). Đợi ~1 phút, "
            "đổi sang backend khác, hoặc thêm GEMINI_API_KEY_2 để luân phiên key."
        )
    if any(k in low for k in ("connection", "refused", "timed out", "timeout",
                              "unreachable", "getaddrinfo", "max retries")):
        return (
            "Không kết nối được tới backend. Ollama: chạy `ollama serve`. "
            "Cloud: kiểm tra mạng / proxy / VPN."
        )
    if "out of memory" in low or "cuda" in low:
        return "Hết bộ nhớ khi chạy ViT5. Thử VIQAG_MODEL nhỏ hơn hoặc chạy trên CPU."
    return f"{type(exc).__name__}: {raw[:200]}"


def env_key_for(backend: str) -> str:
    """Đọc key tương ứng backend từ env (rỗng nếu không có)."""
    name = BACKEND_ENV_KEYS.get(backend or "")
    return (os.getenv(name, "") or "").strip() if name else ""


def build_mcq(question: str, answer: str, distractors: List[str], num_options: int = 4) -> Dict:
    """Ghép Q + A + distractors → MCQ đã xáo thứ tự (logic nằm ở schema.py)."""
    return _build_mcq(question, answer, distractors, num_options=num_options)


def shuffle_mcq(mcq: Dict) -> Dict:
    """Xáo lại thứ tự đáp án, giữ nguyên đáp án đúng (logic nằm ở schema.py)."""
    return _shuffle_mcq(mcq)


def mcq_to_text(mcq_list: List[Dict], show_ans: bool = False) -> str:
    lines = []
    for i, m in enumerate(mcq_list, 1):
        lines.append(f"Câu {i}. {m['question']}")
        for j, opt in enumerate(m["options"]):
            lbl  = LABELS[j]
            mark = "  ← ĐÁP ÁN" if (show_ans and lbl == m["correct_label"]) else ""
            lines.append(f"   {lbl}. {opt}{mark}")
        lines.append("")
    return "\n".join(lines)


def save_to_history(mcq_list: List[Dict]):
    hist = st.session_state["history"]
    hist.insert(0, copy.deepcopy(mcq_list))
    st.session_state["history"] = hist[:10]


# ══════════════════════════════════════════════════════════════
# Cached model loaders (singleton per session)
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_qa_generator(model_name: str, hf_token: str):
    from generator import QAGenerator
    return QAGenerator(model_name=model_name or None, hf_token=hf_token or None)


@st.cache_resource(show_spinner=False)
def load_distractor_generator(backend: str, model: str, api_key: str, ollama_host: str):
    from distractor import DistractorGenerator
    return DistractorGenerator(
        backend=backend or None,
        model=model or None,
        api_key=api_key or None,
        ollama_host=ollama_host or "http://localhost:11434",
    )


# ══════════════════════════════════════════════════════════════
# SIDEBAR – Cấu hình giáo viên
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Cấu hình")

    # ── ViQAG ──────────────────────────────────────────────────
    with st.expander("🧠 Stage 1 – ViT5 (sinh Q-A)", expanded=False):
        viqag_model = st.text_input(
            "Model",
            value=os.getenv("VIQAG_MODEL", "shnl/vit5-vinewsqa-qg-ae"),
            help="Model ViQAG chạy local (torch + transformers).",
        )
        hf_token_input = st.text_input(
            "HF Token (tuỳ chọn)",
            value=os.getenv("HF_TOKEN", ""),
            type="password",
            help="Chỉ cần với model gated/private. Lấy tại huggingface.co/settings/tokens",
        )
        if hf_token_input:
            st.caption(f"HF Token: `{mask_key(hf_token_input)}`")

    # ── LLM ────────────────────────────────────────────────────
    with st.expander("🤖 Stage 2 – LLM (sinh distractors)", expanded=True):
        _env_backend = (os.getenv("LLM_BACKEND", "") or "auto").strip().lower()
        if _env_backend not in LLM_BACKENDS:
            _env_backend = "auto"

        llm_backend = st.selectbox(
            "Backend",
            LLM_BACKENDS,
            index=LLM_BACKENDS.index(_env_backend),
            help=(
                "auto = dò theo key có trong .env (groq → openai → gemini → ollama). "
                "Chọn tay để ép một backend."
            ),
        )

        env_key_name  = BACKEND_ENV_KEYS.get(llm_backend)
        env_key_value = env_key_for(llm_backend)
        if env_key_name:
            if env_key_value:
                st.caption(f"`{env_key_name}` = {mask_key(env_key_value)} (từ .env)")
            else:
                st.caption(f"⚠️ Chưa có `{env_key_name}` trong .env")
        elif llm_backend == "ollama":
            st.caption("Ollama: cần chạy `ollama serve` trước.")
        else:  # auto
            available = [n for b, n in BACKEND_ENV_KEYS.items() if env_key_for(b)]
            st.caption("Key có sẵn: " + (", ".join(f"`{n}`" for n in available) if available else "chưa có key nào"))

        llm_model = st.text_input(
            "Model name",
            value="",
            placeholder="Để trống = mặc định",
            help="VD: gemini-2.5-flash-lite, gpt-4o-mini, llama-3.3-70b-versatile, llama3",
        )
        llm_api_key = st.text_input(
            "API Key (tuỳ chọn)",
            value="",
            type="password",
            placeholder=(
                f"Để trống = dùng .env ({mask_key(env_key_value)})"
                if env_key_name else "Chỉ cần khi backend yêu cầu key"
            ),
            help="Nhập tay sẽ ghi đè key trong .env. Key không bao giờ được in ra UI hay log.",
        )
        ollama_host = st.text_input(
            "Ollama Host",
            value=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )

    # Key hiệu lực: nhập tay > .env
    llm_api_key_effective = (llm_api_key or "").strip() or env_key_value

    st.divider()

    # ── Tuỳ chọn đề ────────────────────────────────────────────
    st.markdown("### 📐 Tuỳ chọn đề")
    num_pairs       = st.slider("Số câu hỏi muốn sinh", 2, 15, 5)
    num_distractors = st.radio("Số đáp án sai / câu", [3, 4], index=0, horizontal=True)
    difficulty      = st.select_slider(
        "Độ khó distractor",
        options=["Dễ", "Trung bình", "Khó"],
        value="Trung bình",
        help="Ảnh hưởng đến độ nhiễu trong prompt LLM",
    )

    st.divider()

    # ── Lịch sử ────────────────────────────────────────────────
    hist = st.session_state["history"]
    if hist:
        st.markdown(f"### 🕑 Lịch sử ({len(hist)} đề)")
        for i, h in enumerate(hist):
            if st.button(f"Đề #{i+1}  –  {len(h)} câu", key=f"hist_{i}"):
                st.session_state["mcq_list"] = copy.deepcopy(h)
                st.rerun()

    st.divider()
    st.caption("ViT5 + LLM · Vietnamese MCQ Generator")


# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown(
    "# 📝 Công cụ Soạn Đề Trắc Nghiệm Tiếng Việt"
)

# ══════════════════════════════════════════════════════════════
# LAYOUT: 2 cột chính  (input+output) + preview
# ══════════════════════════════════════════════════════════════
col_main, col_preview = st.columns([3, 2], gap="large")

# ──────────────────────────────────────────────────────────────
# CỘT TRÁI – Input & Generate
# ──────────────────────────────────────────────────────────────
with col_main:

    # ── Text input ─────────────────────────────────────────────
    st.markdown("#### 📄 Bước 1 – Nhập đoạn văn bản")

    col_ta, col_ex = st.columns([5, 1])
    with col_ex:
        st.markdown(" ")   # padding
        if st.button(
            "🎲\nVí dụ",
            use_container_width=True,
            help="Sinh đoạn văn ngẫu nhiên (OpenAI nếu có key, không thì dùng đoạn mẫu)",
        ):
            with st.spinner("Đang sinh đoạn văn…"):
                st.session_state["context_input"] = _generate_example_paragraph()
            st.rerun()
    with col_ta:
        context = st.text_area(
            "Nội dung",
            height=200,
            placeholder="Dán đoạn văn từ sách giáo khoa, bài báo, bài giảng…",
            label_visibility="collapsed",
            key="context_input",
        )

    word_count = len(context.split()) if context else 0
    st.caption(f"Số từ: **{word_count}**  |  Khuyến nghị: 100–500 từ")

    # ── Nút Generate ───────────────────────────────────────────
    st.markdown("#### 🚀 Bước 2 – Sinh câu hỏi")
    num_pairs = st.number_input(
        "Số câu hỏi muốn sinh",
        min_value=1,
        max_value=20,
        value=num_pairs,
        step=1,
        help="Số cặp Q-A tối đa sẽ được sinh ra từ đoạn văn",
    )
    generate_btn = st.button(
        "✨  Sinh câu hỏi trắc nghiệm",
        type="primary",
        use_container_width=True,
    )

    # ══════════════════════════════════════════════════════════
    # PIPELINE
    # ══════════════════════════════════════════════════════════
    if generate_btn:
        if not context or len(context.strip()) < 20:
            st.warning("⚠️ Đoạn văn quá ngắn. Nhập ít nhất 20 ký tự.")
        else:
            ctx = context.strip()
            st.session_state["context_buf"] = ctx
            mcq_list_new = []

            # difficulty → prompt modifier
            diff_hint = {
                "Dễ": "gần giống nhau về âm thanh hoặc cấu trúc",
                "Trung bình": "hợp lý và có sức nhiễu cao",
                "Khó": "rất dễ gây nhầm lẫn, tương tự đáp án đúng về ngữ nghĩa",
            }[difficulty]

            # ── Load models ───────────────────────────────────
            t_start = time.perf_counter()
            with st.status("⚙️ Khởi tạo model…", expanded=True) as s:
                try:
                    st.write("🔄 Đang kết nối ViT5…")
                    qa_gen = load_qa_generator(viqag_model, hf_token_input)
                    st.write("🔄 Đang kết nối LLM backend…")
                    dist_gen = load_distractor_generator(
                        None if llm_backend == "auto" else llm_backend,
                        llm_model or None,
                        llm_api_key_effective or None,
                        ollama_host,
                    )
                    s.update(label="✅ Kết nối thành công!", state="complete")
                except Exception as e:
                    s.update(label="❌ Lỗi kết nối", state="error")
                    st.error(f"Không khởi tạo được model: {friendly_llm_error(e)}")
                    st.stop()

            # ── Stage 1: ViQAG ────────────────────────────────
            with st.status(
                "🧠 **Stage 1 · ViT5** – Đang phân tích văn bản và sinh Q-A…",
                expanded=True,
            ) as s:
                try:
                    # Yêu cầu thêm 50% để bù cho lọc chất lượng (trùng lặp, answer leakage…)
                    _request_pairs = max(num_pairs + 2, int(num_pairs * 1.5))
                    qa_pairs = qa_gen.generate(ctx, num_pairs=_request_pairs)
                    if not qa_pairs:
                        s.update(label="⚠️ ViT5 không trả về kết quả", state="error")
                        st.warning(
                            "Không tìm thấy câu hỏi phù hợp.  \n"
                            "Thử: văn bản dài hơn · kiểm tra HF Token · đổi sang Local model."
                        )
                        st.stop()
                    s.update(
                        label=f"✅ Stage 1 hoàn tất – {len(qa_pairs)} cặp Q-A",
                        state="complete",
                    )
                    t_stage1 = time.perf_counter() - t_start
                except Exception as e:
                    s.update(label="❌ Lỗi ViT5", state="error")
                    st.error(f"ViT5 lỗi: {friendly_llm_error(e)}")
                    st.stop()

            # ── Quality filter Q-A pairs ───────────────────
            def _jaccard(q1: str, q2: str) -> float:
                w1 = set(q1.lower().split())
                w2 = set(q2.lower().split())
                if not w1 or not w2:
                    return 0.0
                return len(w1 & w2) / len(w1 | w2)

            def _filter_qa(pairs):
                seen_q = []
                result = []
                for p in pairs:
                    q_, a_ = p["question"], p["answer"]
                    # Bỏ answer quá dài (>15 từ)
                    if len(a_.split()) > 15:
                        continue
                    # Bỏ answer xuất hiện trong question (answer leakage)
                    if a_.lower() in q_.lower():
                        continue
                    # Bỏ câu hỏi gần giống nhau – Jaccard >= 0.55
                    if any(_jaccard(q_, sq) >= 0.55 for sq in seen_q):
                        continue
                    seen_q.append(q_)
                    result.append(p)
                return result

            qa_pairs = _filter_qa(qa_pairs)[:num_pairs]  # giữ tối đa num_pairs sau lọc
            if not qa_pairs:
                st.warning("⚠️ Sau khi lọc chất lượng, không còn câu hỏi phù hợp. Thử đoạn văn khác.")
                st.stop()

            # ── Stage 2: LLM distractors ─────────────────────
            progress_bar = st.progress(0, text="🤖 Stage 2 · LLM – Đang sinh distractors…")
            errors_list  = []

            for i, pair in enumerate(qa_pairs):
                q, a = pair["question"], pair["answer"]
                progress_bar.progress(
                    i / len(qa_pairs),
                    text=f"🤖 Stage 2 · LLM – Câu {i+1}/{len(qa_pairs)}: {q[:45]}…",
                )
                # Throttle: ≥4s between calls to stay within Gemini free-tier rate limit
                if i > 0:
                    time.sleep(4)
                try:
                    augmented_ctx = f"{ctx}\n[Yêu cầu: distractors {diff_hint}]"
                    distractors = dist_gen.generate(
                        question=q, answer=a,
                        context=augmented_ctx,
                        num_distractors=num_distractors,
                    )
                    while len(distractors) < num_distractors:
                        distractors.append(f"[Đáp án sai {len(distractors)+1}]")
                    mcq_list_new.append(
                        build_mcq(q, a, distractors[:num_distractors], num_options=num_distractors + 1)
                    )
                except Exception as e:
                    print(f"[Distractor] FAILED câu {i+1}: {scrub_secrets(e)}")
                    errors_list.append(e)
                    # Fallback: vẫn thêm câu với placeholder distractor
                    placeholders = [f"[Đáp án sai {j+1}]" for j in range(num_distractors)]
                    mcq_list_new.append(
                        build_mcq(q, a, placeholders, num_options=num_distractors + 1)
                    )

            progress_bar.progress(1.0, text="✅ Stage 2 hoàn tất!")
            time.sleep(0.3)
            progress_bar.empty()
            t_stage2 = time.perf_counter() - t_start - t_stage1

            if errors_list:
                st.warning(
                    f"⚠️ LLM distractor gặp lỗi {len(errors_list)}/{len(qa_pairs)} câu "
                    f"(dùng placeholder tạm). "
                    f"Nguyên nhân: {friendly_llm_error(errors_list[0])}"
                )

            # ── Validate MCQ trước khi đưa vào session ────────
            from schema import summarize, validate_batch

            kept, rejected = validate_batch(mcq_list_new)
            total_questions = len(mcq_list_new)
            mcq_list_new = [mcq.to_dict() for mcq in kept]
            st.session_state["validation_report"] = summarize(
                total_questions, len(mcq_list_new), rejected
            )
            st.session_state["stage_times"] = {
                "Khởi tạo + Stage 1 (ViT5)": t_stage1,
                "Stage 2 (LLM)": t_stage2,
            }

            st.session_state["mcq_list"] = mcq_list_new
            st.session_state["selected"] = set(range(len(mcq_list_new)))
            save_to_history(mcq_list_new)
            if mcq_list_new:
                st.success(f"🎉 Hoàn tất! Đã tạo **{len(mcq_list_new)} câu trắc nghiệm**.")
            st.rerun()

    # ══════════════════════════════════════════════════════════
    # DANH SÁCH CÂU HỎI – chỉnh sửa / xóa / regenerate
    # ══════════════════════════════════════════════════════════
    mcq_list: List[Dict] = st.session_state["mcq_list"]

    if mcq_list:
        st.divider()
        st.markdown(f"#### ✏️ Bước 3 – Xem và chỉnh sửa ({len(mcq_list)} câu)")

        # Báo cáo validate + thời gian chạy từng stage (nếu vừa sinh xong)
        _report = st.session_state.get("validation_report")
        if _report:
            st.caption(f"🧪 {_report}")
        _timings = st.session_state.get("stage_times")
        if _timings:
            st.caption(
                "⏱ " + " · ".join(f"{name}: {secs:.1f}s" for name, secs in _timings.items())
            )

        # Thanh hành động nhanh
        act1, act2, act3, act4 = st.columns(4)
        with act1:
            if st.button("☑️ Chọn tất cả"):
                st.session_state["selected"] = set(range(len(mcq_list)))
                st.rerun()
        with act2:
            if st.button("☐ Bỏ chọn tất cả"):
                st.session_state["selected"] = set()
                st.rerun()
        with act3:
            if st.button("🔀 Xáo đáp án"):
                st.session_state["mcq_list"] = [shuffle_mcq(m) for m in mcq_list]
                st.rerun()
        with act4:
            if st.button("➕ Thêm câu trống"):
                st.session_state["mcq_list"].append({
                    "question": "Câu hỏi mới…",
                    "answer": "Đáp án đúng",
                    "options": ["Đáp án đúng", "Sai 1", "Sai 2", "Sai 3"],
                    "correct_label": "A",
                    "source": "thủ công",
                })
                st.session_state["selected"].add(len(st.session_state["mcq_list"]) - 1)
                st.rerun()

        st.markdown("")

        # ── Render từng card ───────────────────────────────────
        to_delete = None

        for idx, mcq in enumerate(mcq_list):
            is_selected = idx in st.session_state["selected"]
            card_cls    = "mcq-card selected" if is_selected else "mcq-card"

            with st.container():
                # Header card: checkbox + số câu + nguồn
                hc1, hc2 = st.columns([1, 11])
                with hc1:
                    checked = st.checkbox(
                        f"Chọn câu {idx+1}", value=is_selected,
                        key=f"sel_{idx}",
                        label_visibility="collapsed",
                    )
                    if checked != is_selected:
                        if checked:
                            st.session_state["selected"].add(idx)
                        else:
                            st.session_state["selected"].discard(idx)
                        st.rerun()

                with hc2:
                    st.markdown(
                        f'<span style="font-weight:700;font-size:1rem;">Câu {idx+1}</span>'
                        f'<span class="badge badge-viqag" style="font-size:.7rem;">'
                        f'{mcq.get("source", "ViT5+LLM")}</span>',
                        unsafe_allow_html=True,
                    )

                # Nội dung câu hỏi (có thể chỉnh)
                with st.expander(f"📝 {mcq['question'][:80]}{'…' if len(mcq['question'])>80 else ''}", expanded=False):
                    new_q = st.text_input(
                        "Nội dung câu hỏi",
                        value=mcq["question"],
                        key=f"q_{idx}",
                    )
                    if new_q != mcq["question"]:
                        st.session_state["mcq_list"][idx]["question"] = new_q
                        st.rerun()

                    st.markdown("**Các lựa chọn** (xanh = đáp án đúng):")
                    for j, opt in enumerate(mcq["options"]):
                        lbl        = LABELS[j]
                        is_correct = (lbl == mcq["correct_label"])
                        c1, c2, c3 = st.columns([1, 8, 2])
                        with c1:
                            st.markdown(
                                f'<span style="color:{"#16a34a" if is_correct else "#6b7280"};'
                                f'font-weight:{"700" if is_correct else "400"}">{lbl}.</span>',
                                unsafe_allow_html=True,
                            )
                        with c2:
                            new_opt = st.text_input(
                                f"opt_{idx}_{j}",
                                value=opt,
                                key=f"opt_{idx}_{j}",
                                label_visibility="collapsed",
                            )
                            if new_opt != opt:
                                old_ans = mcq["answer"]
                                st.session_state["mcq_list"][idx]["options"][j] = new_opt
                                if is_correct:
                                    st.session_state["mcq_list"][idx]["answer"] = new_opt
                                st.rerun()
                        with c3:
                            if not is_correct:
                                if st.button("✔ Đặt đúng", key=f"setcorrect_{idx}_{j}"):
                                    st.session_state["mcq_list"][idx]["correct_label"] = lbl
                                    st.session_state["mcq_list"][idx]["answer"] = opt
                                    st.rerun()

                    # Nút hành động câu
                    rb1, rb2, rb3 = st.columns(3)
                    with rb1:
                        if st.button("🔀 Xáo lại câu này", key=f"shuffle_{idx}"):
                            st.session_state["mcq_list"][idx] = shuffle_mcq(
                                st.session_state["mcq_list"][idx]
                            )
                            st.rerun()
                    with rb2:
                        if st.button("🔁 Regenerate distractors", key=f"regen_{idx}"):
                            st.session_state["regen_idx"] = idx
                            st.rerun()
                    with rb3:
                        if st.button("🗑️ Xóa câu này", key=f"del_{idx}", type="secondary"):
                            to_delete = idx

        # Xóa câu (sau vòng lặp để tránh lỗi index)
        if to_delete is not None:
            st.session_state["mcq_list"].pop(to_delete)
            st.session_state["selected"].discard(to_delete)
            st.session_state["selected"] = {
                i if i < to_delete else i - 1
                for i in st.session_state["selected"] if i != to_delete
            }
            st.rerun()

        # ── Regenerate single question ────────────────────────
        regen_idx = st.session_state.get("regen_idx")
        if regen_idx is not None and 0 <= regen_idx < len(mcq_list):
            st.session_state["regen_idx"] = None
            with st.spinner(f"🔁 Đang regenerate distractors câu {regen_idx+1}…"):
                try:
                    dist_gen = load_distractor_generator(
                        None if llm_backend == "auto" else llm_backend,
                        llm_model or None,
                        llm_api_key_effective or None,
                        ollama_host,
                    )
                    m = mcq_list[regen_idx]
                    new_d = dist_gen.generate(
                        question=m["question"], answer=m["answer"],
                        context=st.session_state.get("context_buf", ""),
                        num_distractors=num_distractors,
                    )
                    while len(new_d) < num_distractors:
                        new_d.append(f"[Đáp án sai {len(new_d)+1}]")
                    st.session_state["mcq_list"][regen_idx] = build_mcq(
                        m["question"], m["answer"], new_d[:num_distractors],
                        num_options=num_distractors + 1,
                    )
                    st.success(f"✅ Đã cập nhật câu {regen_idx+1}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Lỗi regenerate: {friendly_llm_error(e)}")

    else:
        # Màn chào ban đầu
        st.info(
            "**Hướng dẫn nhanh:**\n"
            "1. Dán đoạn văn bản vào ô phía trên\n"
            "2. Bấm **Sinh câu hỏi trắc nghiệm**\n"
            "3. Chỉnh sửa câu hỏi nếu cần\n"
            "4. Export Word/PDF từ cột bên phải"
        )
        with st.expander("ℹ️ Về pipeline ViT5 + LLM"):
            st.markdown("""
| Thành phần | Vai trò |
|---|---|
| **ViT5** | Model fine-tuned `shnl/vit5-vinewsqa-qg-ae` sinh câu hỏi & trích đáp án từ văn bản tiếng Việt |
| **LLM** | Sinh 3–4 đáp án nhiễu (Groq / Gemini / OpenAI / Ollama) |
| **MCQ Builder** | Xáo trộn đáp án, đánh nhãn A/B/C/D |
            """)


# ──────────────────────────────────────────────────────────────
# CỘT PHẢI – Preview & Export
# ──────────────────────────────────────────────────────────────
with col_preview:
    mcq_list: List[Dict] = st.session_state["mcq_list"]
    selected: set        = st.session_state["selected"]
    export_list          = [mcq_list[i] for i in sorted(selected) if i < len(mcq_list)]

    st.markdown("#### 👁️ Xem trước & Xuất đề")

    if export_list:
        st.caption(f"**{len(export_list)}/{len(mcq_list)}** câu được chọn để xuất")

        # Preview text
        preview_txt = mcq_to_text(export_list, show_ans=False)
        st.markdown(
            f'<div class="preview-box">{preview_txt}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("##### 📤 Xuất đề")

        # ── JSON (format chuẩn, import lại được) ───────────────
        try:
            from export_utils import export_json_bytes

            st.download_button(
                "⬇️ JSON (format chuẩn)",
                data=export_json_bytes(export_list),
                file_name="mcq.json",
                mime="application/json",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"JSON: {e}")

        # ── CSV (Quizizz / Google Forms) ───────────────────────
        try:
            from export_utils import export_csv_bytes

            st.download_button(
                "⬇️ CSV (Quizizz / Google Forms)",
                data=export_csv_bytes(export_list),
                file_name="mcq_quizizz.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"CSV: {e}")

        # ── TXT ────────────────────────────────────────────────
        try:
            from export_utils import export_txt_bytes

            st.download_button(
                "⬇️ TXT (kèm đáp án)",
                data=export_txt_bytes(export_list),
                file_name="mcq.txt",
                mime="text/plain",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"TXT: {e}")

        # ── Word ───────────────────────────────────────────────
        try:
            from export_utils import export_word_bytes
            word_bytes = export_word_bytes(export_list)
            st.download_button(
                "⬇️ Word (.docx) – có đáp án trang sau",
                data=word_bytes,
                file_name="mcq.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Word: {e}")

        # ── PDF ────────────────────────────────────────────────
        try:
            from export_utils import export_pdf_bytes
            pdf_bytes = export_pdf_bytes(export_list)
            st.download_button(
                "⬇️ PDF – có đáp án trang sau",
                data=pdf_bytes,
                file_name="mcq.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"PDF: {e}")

        st.divider()

        # ── Đáp án riêng ───────────────────────────────────────
        st.markdown("##### 🔑 Bảng đáp án")
        ans_md = "| Câu | Đáp án |\n|---|---|\n"
        for i, m in enumerate(export_list, 1):
            ans_md += f"| {i} | **{m['correct_label']}**. {m['answer']} |\n"
        st.markdown(ans_md)

    else:
        st.info("Chọn câu hỏi ở cột trái (checkbox ☑) để xem preview và xuất đề.")

        if mcq_list:
            st.markdown(f"Hiện có **{len(mcq_list)} câu** chưa được chọn.")
            if st.button("☑️ Chọn tất cả & preview"):
                st.session_state["selected"] = set(range(len(mcq_list)))
                st.rerun()
