import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time
import json
from datetime import datetime

st.set_page_config(
    page_title="Question Generator",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="collapsed",
)

LIQUID_GLASS_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --glass-bg: rgba(255, 255, 255, 0.45);
    --glass-border: rgba(255, 255, 255, 0.55);
    --glass-shadow: 0 8px 32px rgba(0, 0, 0, 0.06);
    --glass-blur: 18px;
    --text-primary: #1a1a2e;
    --text-secondary: #555770;
    --text-muted: #8b8da3;
    --accent: #6366f1;
    --accent-hover: #4f46e5;
    --radius: 20px;
    --radius-sm: 14px;
    --transition: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

*, *::before, *::after {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #e0e7ff 0%, #fce7f3 25%, #e0f2fe 50%, #ede9fe 75%, #fdf2f8 100%) !important;
    background-attachment: fixed !important;
}

[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.4) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.5) !important;
}

[data-testid="stSidebar"] { display: none !important; }

.block-container {
    max-width: 1080px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 4rem !important;
}

/* ---- BACKGROUND BLOBS ---- */

.bg-blobs {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}

.blob {
    position: absolute;
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.5;
    animation: blobFloat 20s ease-in-out infinite alternate;
}

.blob-1 {
    width: 400px; height: 400px;
    background: radial-gradient(circle, #c7d2fe, #a5b4fc);
    top: -100px; left: -50px;
    animation-duration: 22s;
}

.blob-2 {
    width: 350px; height: 350px;
    background: radial-gradient(circle, #fbcfe8, #f9a8d4);
    top: 30%; right: -80px;
    animation-duration: 18s;
    animation-delay: -5s;
}

.blob-3 {
    width: 300px; height: 300px;
    background: radial-gradient(circle, #bae6fd, #7dd3fc);
    bottom: -60px; left: 30%;
    animation-duration: 25s;
    animation-delay: -10s;
}

.blob-4 {
    width: 250px; height: 250px;
    background: radial-gradient(circle, #ddd6fe, #c4b5fd);
    top: 60%; left: 10%;
    animation-duration: 20s;
    animation-delay: -3s;
}

@keyframes blobFloat {
    0%   { transform: translate(0, 0) scale(1); }
    33%  { transform: translate(30px, -40px) scale(1.05); }
    66%  { transform: translate(-20px, 20px) scale(0.95); }
    100% { transform: translate(10px, -10px) scale(1.02); }
}

/* ---- HERO ---- */

.hero-section {
    text-align: center;
    padding: 40px 0 32px 0;
    position: relative;
}

.hero-badge {
    display: inline-block;
    padding: 6px 16px;
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 100px;
    font-size: 12px;
    font-weight: 600;
    color: var(--accent);
    margin-top: 28px;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: 24px;
}

.hero-title {
    font-size: 44px;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -1px;
    line-height: 1.15;
    margin: 0 0 14px 0;
}

.hero-subtitle {
    font-size: 16px;
    color: var(--text-secondary);
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.7;
}

/* ---- 3D GLASS OBJECT ---- */

.scene-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 30px 0 10px 0;
}

.scene {
    width: 140px;
    height: 140px;
    perspective: 600px;
}

.cube {
    width: 100%;
    height: 100%;
    position: relative;
    transform-style: preserve-3d;
    animation: cubeRotate 12s ease-in-out infinite;
}

.cube__face {
    position: absolute;
    width: 140px;
    height: 140px;
    border: 1.5px solid rgba(255, 255, 255, 0.6);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    box-shadow:
        inset 0 0 30px rgba(255, 255, 255, 0.2),
        0 8px 32px rgba(99, 102, 241, 0.08);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    font-weight: 700;
    color: rgba(99, 102, 241, 0.35);
    letter-spacing: -1px;
}

.cube__face--front  { transform: rotateY(  0deg) translateZ(70px); }
.cube__face--right  { transform: rotateY( 90deg) translateZ(70px); }
.cube__face--back   { transform: rotateY(180deg) translateZ(70px); }
.cube__face--left   { transform: rotateY(-90deg) translateZ(70px); }
.cube__face--top    { transform: rotateX( 90deg) translateZ(70px); }
.cube__face--bottom { transform: rotateX(-90deg) translateZ(70px); }

@keyframes cubeRotate {
    0%   { transform: rotateX(-15deg) rotateY(0deg); }
    25%  { transform: rotateX(5deg) rotateY(90deg); }
    50%  { transform: rotateX(-10deg) rotateY(180deg); }
    75%  { transform: rotateX(8deg) rotateY(270deg); }
    100% { transform: rotateX(-15deg) rotateY(360deg); }
}

/* ---- GLASS CARD ---- */

.glass-card {
    background: var(--glass-bg);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    box-shadow: var(--glass-shadow);
    transition: box-shadow var(--transition), transform var(--transition);
}

.glass-card:hover {
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.08);
}

.card-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
    display: block;
}

/* ---- INPUTS ---- */

textarea, input {
    border-radius: 12px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}

textarea:focus, input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
}

/* ---- SLIDER ---- */

[data-testid="stSlider"] label {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
}

.stSlider > div > div > div[role="slider"] {
    background: var(--accent) !important;
}

/* ---- BUTTON ---- */

.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.3px !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 12px 28px !important;
    background: var(--accent) !important;
    color: #fff !important;
    transition: all var(--transition) !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.30) !important;
    width: 100% !important;
}

.stButton > button:hover {
    background: var(--accent-hover) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.40) !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ---- RESULT ---- */

.result-container {
    background: rgba(255, 255, 255, 0.55);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.6);
    border-radius: var(--radius);
    padding: 28px;
    margin-top: 20px;
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.08);
    position: relative;
    overflow: hidden;
    animation: glassReveal 0.5s ease;
}

.result-container::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #818cf8, #6366f1, #a78bfa);
    border-radius: var(--radius) var(--radius) 0 0;
}

.result-question {
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.55;
    margin: 0;
}

.result-meta {
    display: flex;
    gap: 16px;
    margin-top: 16px;
    padding-top: 14px;
    border-top: 1px solid rgba(0, 0, 0, 0.05);
}

.result-meta-item {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px;
    font-weight: 500;
    color: var(--text-muted);
}

@keyframes glassReveal {
    from { opacity: 0; transform: translateY(16px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

/* ---- STATS ---- */

.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-top: 10px;
}

.stat-box {
    background: rgba(255, 255, 255, 0.4);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.5);
    border-radius: 12px;
    padding: 14px 10px;
    text-align: center;
}

.stat-value {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
}

.stat-label {
    font-size: 10px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 2px;
}

/* ---- HISTORY ---- */

.history-item {
    background: rgba(255, 255, 255, 0.35);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.5);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 8px;
    transition: all var(--transition);
}

.history-item:hover {
    border-color: rgba(99, 102, 241, 0.3);
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.06);
}

.history-q {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0 0 4px 0;
    line-height: 1.5;
}

.history-a {
    font-size: 12px;
    color: var(--text-muted);
    margin: 0;
}

.history-time {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 6px;
}

/* ---- EXAMPLES ---- */

.example-card {
    background: rgba(255, 255, 255, 0.35);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.5);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 6px;
    transition: all var(--transition);
}

.example-card:hover {
    border-color: rgba(99, 102, 241, 0.3);
    background: rgba(255, 255, 255, 0.5);
}

.example-ctx {
    font-size: 13px;
    color: var(--text-secondary);
    margin: 0;
    line-height: 1.6;
}

.example-ans {
    font-size: 12px;
    font-weight: 600;
    color: var(--accent);
    margin: 8px 0 0 0;
}

/* ---- EMPTY STATE ---- */

.empty-state {
    text-align: center;
    padding: 32px 16px;
    color: var(--text-muted);
    font-size: 13px;
    line-height: 1.7;
}

/* ---- DIVIDER ---- */

.divider {
    height: 1px;
    background: rgba(0, 0, 0, 0.06);
    margin: 20px 0;
}

/* ---- DOWNLOAD BUTTON ---- */

[data-testid="stDownloadButton"] > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    border: 1.5px solid rgba(255,255,255,0.6) !important;
    border-radius: var(--radius-sm) !important;
    padding: 8px 16px !important;
    background: rgba(255,255,255,0.4) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    color: var(--text-primary) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    transition: all var(--transition) !important;
}

[data-testid="stDownloadButton"] > button:hover {
    border-color: rgba(99,102,241,0.4) !important;
    color: var(--accent) !important;
}

/* ---- EXPANDER ---- */

[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.5) !important;
    border-radius: var(--radius) !important;
    background: rgba(255,255,255,0.35) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
}

/* ---- FOOTER ---- */

.footer-custom {
    text-align: center;
    font-size: 12px;
    color: var(--text-muted);
    padding: 28px 0 0 0;
    border-top: 1px solid rgba(0, 0, 0, 0.05);
    margin-top: 40px;
    line-height: 1.7;
}

/* ---- MISC ---- */

footer { visibility: hidden !important; }
#MainMenu { visibility: hidden !important; }

.stMarkdown hr {
    border-color: rgba(0,0,0,0.05) !important;
}
</style>
"""

HERO_3D_HTML = """
<div class="bg-blobs">
    <div class="blob blob-1"></div>
    <div class="blob blob-2"></div>
    <div class="blob blob-3"></div>
    <div class="blob blob-4"></div>
</div>

<div class="hero-section">
    <div class="scene-wrapper">
        <div class="scene">
            <div class="cube">
                <div class="cube__face cube__face--front">AQ</div>
                <div class="cube__face cube__face--back">AI</div>
                <div class="cube__face cube__face--right">T5</div>
                <div class="cube__face cube__face--left">NLP</div>
                <div class="cube__face cube__face--top">QG</div>
                <div class="cube__face cube__face--bottom">ML</div>
            </div>
        </div>
    </div>
    <div class="hero-badge">T5 Transformer</div>
    <div class="hero-title">Question Generator</div>
    <div class="hero-subtitle">
        Paste a passage and highlight the answer, the model
        crafts a relevant question in seconds.
    </div>
</div>
"""


@st.cache_resource
def load_model():
    path = "./QG_Model/best_model"
    tok = T5Tokenizer.from_pretrained(path)
    mdl = T5ForConditionalGeneration.from_pretrained(path)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = mdl.to(dev)
    mdl.eval()
    return tok, mdl, dev


def run_inference(context, answer, num_beams, max_len, tokenizer, model, device):
    text = f"generate question: answer: {answer.strip()} context: {context.strip()}"
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(
            inputs["input_ids"],
            max_length=int(max_len),
            num_beams=int(num_beams),
            early_stopping=True,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


if "history" not in st.session_state:
    st.session_state.history = []
if "ctx_value" not in st.session_state:
    st.session_state.ctx_value = ""
if "ans_value" not in st.session_state:
    st.session_state.ans_value = ""

st.markdown(LIQUID_GLASS_STYLES, unsafe_allow_html=True)
st.markdown(HERO_3D_HTML, unsafe_allow_html=True)

tokenizer, model, device = load_model()

col_left, col_right = st.columns([5, 2], gap="large")

with col_left:

    st.markdown('<span class="card-label">Context</span>', unsafe_allow_html=True)
    context = st.text_area(
        "ctx",
        value=st.session_state.ctx_value,
        height=200,
        placeholder="Paste the source paragraph here ...",
        label_visibility="collapsed",
    )

    st.markdown('<span class="card-label">Answer</span>', unsafe_allow_html=True)
    answer = st.text_input(
        "ans",
        value=st.session_state.ans_value,
        placeholder="The target answer found in the context above",
        label_visibility="collapsed",
    )

    c1, c2 = st.columns(2)
    with c1:
        num_beams = st.slider("Beam width", 1, 10, 4)
    with c2:
        max_length = st.slider("Max tokens", 16, 128, 64, step=8)

    generate_clicked = st.button("Generate Question", use_container_width=True, type="primary")

    if generate_clicked:
        if not context.strip() or not answer.strip():
            st.error("Please fill in both the context and the answer.")
        else:
            start = time.time()
            with st.spinner("Generating ..."):
                result = run_inference(context, answer, num_beams, max_length, tokenizer, model, device)
            elapsed = time.time() - start

            ctx_words = len(context.split())

            st.session_state.history.insert(0, {
                "question": result,
                "answer": answer.strip(),
                "context_preview": context.strip()[:120],
                "time": datetime.now().strftime("%H:%M:%S"),
                "elapsed": round(elapsed, 2),
                "beams": num_beams,
            })

            st.markdown(f"""
            <div class="result-container">
                <p class="result-question">{result}</p>
                <div class="result-meta">
                    <span class="result-meta-item">{elapsed:.2f}s</span>
                    <span class="result-meta-item">{num_beams} beams</span>
                    <span class="result-meta-item">{ctx_words} words</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="card-label">Quick Examples</span>', unsafe_allow_html=True)

    examples = [
        (
            "Friedrich Hayek shared the 1974 Nobel Memorial Prize in Economic Sciences with Gunnar Myrdal for his pioneering work in the theory of money and economic fluctuations.",
            "Nobel Memorial Prize",
        ),
        (
            "The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states as protection against various nomadic groups.",
            "protection against various nomadic groups",
        ),
        (
            "An ascospore is a spore contained in an ascus. Typically, a single ascus will contain eight ascospores produced by meiosis followed by a mitotic division.",
            "eight ascospores",
        ),
    ]

    for i, (ctx, ans) in enumerate(examples):
        with st.container():
            st.markdown(f"""
            <div class="example-card">
                <p class="example-ctx">{ctx}</p>
                <p class="example-ans">Answer: {ans}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Use this example", key=f"ex_{i}", use_container_width=True):
                st.session_state.ctx_value = ctx
                st.session_state.ans_value = ans
                st.rerun()


with col_right:

    st.markdown('<span class="card-label">Session Stats</span>', unsafe_allow_html=True)
    total = len(st.session_state.history)
    avg_time = (
        sum(h["elapsed"] for h in st.session_state.history) / total
        if total > 0
        else 0
    )
    st.markdown(f"""
    <div class="stats-grid">
        <div class="stat-box">
            <div class="stat-value">{total}</div>
            <div class="stat-label">Generated</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{avg_time:.1f}s</div>
            <div class="stat-label">Avg time</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">T5</div>
            <div class="stat-label">Model</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="card-label">History</span>', unsafe_allow_html=True)

    if st.session_state.history:
        for h in st.session_state.history[:10]:
            st.markdown(f"""
            <div class="history-item">
                <p class="history-q">{h["question"]}</p>
                <p class="history-a">Answer: {h["answer"]}</p>
                <div class="history-time">{h["time"]}  ·  {h["elapsed"]}s</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        export_data = json.dumps(st.session_state.history, indent=2, ensure_ascii=False)
        st.download_button(
            "Export history as JSON",
            data=export_data,
            file_name="aqg_history.json",
            mime="application/json",
            use_container_width=True,
        )

        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown("""
        <div class="empty-state">
            No questions generated yet.<br>
            Fill in a context and answer, then press Generate.
        </div>
        """, unsafe_allow_html=True)


st.markdown("""
<div class="footer-custom">
    Built with Streamlit and Hugging Face Transformers<br>
    T5-Small fine-tuned for Question Generation
</div>
""", unsafe_allow_html=True)
