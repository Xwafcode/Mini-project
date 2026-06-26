import json
import pickle
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

DATA_DIR   = Path("data/processed")
EVAL_DIR   = Path("data/eval")
MODEL_DIR  = Path("models")
RESULT_DIR = Path("data/results")

st.set_page_config(
    page_title="CBR Putusan Pencurian - PN Malang",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
    :root {
        --white:      #FFFFFF;
        --snow:       #FAFBFF;
        --cloud:      #F0F2FA;
        --lavender:   #E8ECFA;
        --periwinkle: #7C8CF8;
        --iris:       #5B6AE0;
        --deep-iris:  #4452C5;
        --peach:      #F8B4A0;
        --peach-lt:   #FDDDD2;
        --mint:       #7EDCB5;
        --mint-lt:    #D1F5E6;
        --sky:        #7BBFEA;
        --sky-lt:     #D0EAFB;
        --rose:       #E99DB5;
        --text-dark:  #2D3250;
        --text-mid:   #5A5F7E;
        --text-light: #8E92AB;
        --border:     #E2E5F1;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: transparent !important;
        font-family: 'Inter', sans-serif;
        color: var(--text-dark);
    }

    [data-testid="stAppViewContainer"] > .main {
        background: transparent !important;
    }

    [data-testid="block-container"] {
        padding-top: 1rem !important;
        background: transparent !important;
    }

    #bg-canvas {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        z-index: -10;
        pointer-events: none;
    }

    #three-canvas {
        position: fixed;
        bottom: 24px;
        right: 24px;
        width: 150px;
        height: 150px;
        z-index: 100;
        pointer-events: none;
        border-radius: 50%;
        overflow: hidden;
        filter: drop-shadow(0 4px 20px rgba(124,140,248,0.25));
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--snow) 0%, var(--lavender) 100%) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * {
        color: var(--text-dark) !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: 'DM Serif Display', serif !important;
        color: var(--iris) !important;
    }
    [data-testid="stSidebar"] .stSlider > div > div > div {
        background: var(--periwinkle) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: var(--iris) !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] label {
        color: var(--text-mid) !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] .stRadio > div {
        gap: 6px;
    }

    .main-header {
        font-family: 'DM Serif Display', serif;
        font-size: 2.6rem;
        font-weight: 400;
        background: linear-gradient(135deg, var(--iris) 0%, var(--periwinkle) 40%, var(--peach) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        padding: 0.8rem 0 0.4rem;
        animation: headerReveal 1s ease-out both;
    }

    .main-header::after {
        content: '';
        display: block;
        width: 60px;
        height: 3px;
        background: linear-gradient(90deg, var(--iris), var(--peach));
        margin: 0.6rem auto 0;
        border-radius: 2px;
        animation: lineExpand 1.2s ease-out 0.2s both;
    }

    @keyframes headerReveal {
        from { opacity: 0; transform: translateY(-20px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    @keyframes lineExpand {
        from { width: 0; opacity: 0; }
        to   { width: 60px; opacity: 1; }
    }

    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        color: var(--text-mid);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
        letter-spacing: 0.2px;
        line-height: 1.7;
        animation: fadeSlideUp 1.2s ease-out 0.3s both;
    }

    @keyframes fadeSlideUp {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(124,140,248,0.1);
        border: 1px solid rgba(124,140,248,0.25);
        border-radius: 20px;
        padding: 5px 16px;
        font-size: 0.75rem;
        color: var(--iris);
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        animation: badgeGlow 2.5s ease-in-out infinite alternate;
    }

    @keyframes badgeGlow {
        from { box-shadow: 0 0 0 rgba(124,140,248,0); }
        to   { box-shadow: 0 2px 16px rgba(124,140,248,0.2); }
    }

    .metric-card {
        background: linear-gradient(135deg, var(--iris) 0%, var(--periwinkle) 100%);
        border-radius: 16px;
        padding: 1.4rem 1rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 6px 24px rgba(91,106,224,0.25);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: cardPop 0.5s cubic-bezier(0.34,1.56,0.64,1) both;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: -60%; left: -60%;
        width: 220%; height: 220%;
        background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, transparent 55%);
        animation: shimmer 4s linear infinite;
    }

    @keyframes shimmer {
        0%   { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    @keyframes cardPop {
        from { opacity: 0; transform: scale(0.85) translateY(16px); }
        to   { opacity: 1; transform: scale(1) translateY(0); }
    }

    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 12px 36px rgba(91,106,224,0.35);
    }

    .metric-card h3 {
        margin: 0;
        font-size: 0.72rem;
        opacity: 0.85;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        font-weight: 500;
    }

    .metric-card h1 {
        margin: 0.4rem 0 0 0;
        font-size: 1.6rem;
        font-family: 'DM Serif Display', serif;
        font-weight: 400;
    }

    .case-card {
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 2px 8px rgba(45,50,80,0.05);
        transition: all 0.3s cubic-bezier(0.25,0.46,0.45,0.94);
        position: relative;
        overflow: hidden;
    }

    .case-card::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        background: linear-gradient(180deg, var(--periwinkle), var(--peach));
        border-radius: 3px 0 0 3px;
    }

    .case-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 28px rgba(91,106,224,0.12);
        border-color: rgba(124,140,248,0.3);
    }

    .result-box {
        background: rgba(209,245,230,0.4);
        border-left: 3px solid var(--mint);
        padding: 1rem 1.2rem;
        border-radius: 0 10px 10px 0;
        margin: 0.5rem 0;
        transition: transform 0.2s ease;
    }

    .result-box:hover { transform: translateX(3px); }

    .result-box-warn {
        background: rgba(253,221,210,0.4);
        border-left: 3px solid var(--peach);
        padding: 1rem 1.2rem;
        border-radius: 0 10px 10px 0;
        margin: 0.5rem 0;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--iris) 0%, var(--periwinkle) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.3px !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 16px rgba(91,106,224,0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(91,106,224,0.4) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255,255,255,0.7) !important;
        border-radius: 10px !important;
        padding: 4px !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid var(--border) !important;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 7px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        color: var(--text-mid) !important;
        transition: all 0.25s ease !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--iris), var(--periwinkle)) !important;
        color: white !important;
        box-shadow: 0 3px 12px rgba(91,106,224,0.3) !important;
    }

    .stTextArea textarea {
        font-size: 0.85rem !important;
        background: rgba(255,255,255,0.85) !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        color: var(--text-dark) !important;
        transition: border-color 0.3s ease, box-shadow 0.3s ease !important;
    }

    .stTextArea textarea:focus {
        border-color: var(--periwinkle) !important;
        box-shadow: 0 0 0 3px rgba(124,140,248,0.12) !important;
    }

    .stSelectbox > div > div {
        background: rgba(255,255,255,0.85) !important;
        border: 1.5px solid var(--border) !important;
        border-radius: 10px !important;
    }

    .stDataFrame {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 12px rgba(45,50,80,0.06) !important;
    }

    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.75) !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        color: var(--iris) !important;
        border: 1px solid var(--border) !important;
    }

    hr {
        border-color: var(--border) !important;
    }

    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: var(--cloud); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--periwinkle), var(--peach));
        border-radius: 4px;
    }

    .stSpinner > div {
        border-top-color: var(--periwinkle) !important;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(14px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    .fade-in { animation: fadeIn 0.6s ease-out both; }

    #MainMenu, header, footer { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }
</style>

<canvas id="bg-canvas"></canvas>
<canvas id="three-canvas"></canvas>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function() {
    const canvas = document.getElementById('bg-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let W, H;
    function resize() {
        W = canvas.width  = window.innerWidth;
        H = canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    const NUM_STARS = 100;
    const stars = Array.from({length: NUM_STARS}, () => ({
        x:     Math.random() * window.innerWidth,
        y:     Math.random() * window.innerHeight,
        r:     Math.random() * 1.6 + 0.3,
        alpha: Math.random(),
        speed: Math.random() * 0.008 + 0.002,
        hue:   [240, 260, 20, 160, 340][Math.floor(Math.random()*5)]
    }));

    class FallingStar {
        constructor() { this.reset(); }
        reset() {
            this.x     = Math.random() * W * 1.4;
            this.y     = -10;
            this.speed = Math.random() * 2.5 + 1.5;
            this.angle = Math.PI / 4 + (Math.random() - 0.5) * 0.25;
            this.alpha = 0;
            this.fade  = Math.random() * 0.015 + 0.008;
            this.phase = 'in';
            this.trail = [];
            this.hue   = [240, 20, 160, 0, 280][Math.floor(Math.random()*5)];
            this.size  = Math.random() * 1.5 + 0.6;
        }
        update() {
            this.x += Math.cos(this.angle) * this.speed;
            this.y += Math.sin(this.angle) * this.speed;
            this.trail.push({x: this.x, y: this.y});
            if (this.trail.length > 20) this.trail.shift();
            if (this.phase === 'in') {
                this.alpha = Math.min(1, this.alpha + this.fade);
                if (this.alpha >= 0.85) this.phase = 'out';
            } else {
                this.alpha = Math.max(0, this.alpha - this.fade * 0.6);
            }
            if (this.y > H + 80 || this.alpha <= 0) this.reset();
        }
        draw(ctx) {
            if (this.trail.length < 2) return;
            for (let i = 1; i < this.trail.length; i++) {
                const t = i / this.trail.length;
                ctx.beginPath();
                ctx.moveTo(this.trail[i-1].x, this.trail[i-1].y);
                ctx.lineTo(this.trail[i].x,   this.trail[i].y);
                ctx.strokeStyle = `hsla(${this.hue},70%,70%,${this.alpha * t * 0.7})`;
                ctx.lineWidth   = this.size * t;
                ctx.lineCap     = 'round';
                ctx.stroke();
            }
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size * 1.8, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${this.hue},80%,88%,${this.alpha * 0.8})`;
            ctx.fill();
        }
    }

    const fallingStars = Array.from({length: 10}, () => {
        const s = new FallingStar();
        s.y = Math.random() * H;
        return s;
    });

    function drawBatik(ctx, w, h) {
        ctx.save();
        ctx.globalAlpha = 0.035;
        const spacing = 70;
        for (let x = 0; x < w + spacing; x += spacing) {
            for (let y = 0; y < h + spacing; y += spacing) {
                const hue = ((x + y) / 3) % 360;
                ctx.strokeStyle = `hsla(${hue}, 30%, 60%, 0.5)`;
                ctx.lineWidth = 0.6;
                ctx.beginPath();
                ctx.moveTo(x, y - 16);
                ctx.lineTo(x + 16, y);
                ctx.lineTo(x, y + 16);
                ctx.lineTo(x - 16, y);
                ctx.closePath();
                ctx.stroke();
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, Math.PI * 2);
                ctx.stroke();
                for (let a = 0; a < 4; a++) {
                    ctx.save();
                    ctx.translate(x, y);
                    ctx.rotate(a * Math.PI / 2);
                    ctx.beginPath();
                    ctx.ellipse(10, 0, 6, 3, 0, 0, Math.PI * 2);
                    ctx.stroke();
                    ctx.restore();
                }
            }
        }
        ctx.restore();
    }

    function animate() {
        requestAnimationFrame(animate);
        const grad = ctx.createLinearGradient(0, 0, W * 0.3, H);
        grad.addColorStop(0,   '#FAFBFF');
        grad.addColorStop(0.3, '#F0F2FA');
        grad.addColorStop(0.6, '#E8ECFA');
        grad.addColorStop(1,   '#F5F0FB');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, W, H);

        drawBatik(ctx, W, H);

        stars.forEach(s => {
            s.alpha += s.speed;
            if (s.alpha > 1 || s.alpha < 0) s.speed = -s.speed;
            ctx.beginPath();
            ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${s.hue},60%,72%,${Math.max(0,Math.min(1,s.alpha))*0.5})`;
            ctx.fill();
        });

        fallingStars.forEach(s => { s.update(); s.draw(ctx); });
        ctx.globalAlpha = 1;
    }
    animate();

    const tc = document.getElementById('three-canvas');
    if (!tc || typeof THREE === 'undefined') return;

    const renderer = new THREE.WebGLRenderer({ canvas: tc, alpha: true, antialias: true });
    renderer.setSize(150, 150);
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0x000000, 0);

    const scene  = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    camera.position.set(0, 0, 4);

    scene.add(new THREE.AmbientLight(0xfafbff, 0.8));
    const d1 = new THREE.DirectionalLight(0x7C8CF8, 1.4);
    d1.position.set(3, 4, 5); scene.add(d1);
    const d2 = new THREE.DirectionalLight(0xF8B4A0, 0.9);
    d2.position.set(-3, -2, 2); scene.add(d2);
    const pl = new THREE.PointLight(0x7EDCB5, 1, 10);
    pl.position.set(0, 2, 2); scene.add(pl);

    const kGeo = new THREE.TorusKnotGeometry(0.75, 0.22, 100, 16, 2, 3);
    const kMat = new THREE.MeshPhongMaterial({
        color: 0x7C8CF8, emissive: 0x2a2a60, specular: 0xffffff,
        shininess: 90, transparent: true, opacity: 0.88
    });
    const knot = new THREE.Mesh(kGeo, kMat);
    scene.add(knot);

    const sPos = [[1.3,0.5,0],[-1.3,-0.4,0.4],[0.3,1.4,0.2],[-0.5,-1.3,-0.2]];
    const cols = [0xF8B4A0, 0x7EDCB5, 0x7BBFEA, 0xE99DB5];
    const spheres = sPos.map((p, i) => {
        const g = new THREE.SphereGeometry(0.1 + i*0.025, 16, 16);
        const m = new THREE.MeshPhongMaterial({
            color: cols[i], emissive: cols[i], emissiveIntensity: 0.25,
            shininess: 50, transparent: true, opacity: 0.8
        });
        const mesh = new THREE.Mesh(g, m);
        mesh.position.set(...p); scene.add(mesh);
        return mesh;
    });

    const rGeo = new THREE.TorusGeometry(1.4, 0.03, 8, 72);
    const rMat = new THREE.MeshBasicMaterial({color: 0x7C8CF8, transparent: true, opacity: 0.25});
    const ring = new THREE.Mesh(rGeo, rMat);
    ring.rotation.x = Math.PI / 4; scene.add(ring);

    let t = 0;
    function anim3d() {
        requestAnimationFrame(anim3d);
        t += 0.015;
        knot.rotation.x = t * 0.6;
        knot.rotation.y = t;
        ring.rotation.y = t * 0.4;
        spheres.forEach((m, i) => {
            m.position.x = Math.cos(t + i*1.57) * 1.3;
            m.position.y = Math.sin(t*0.7 + i*1.57) * 1.1;
            m.position.z = Math.sin(t*0.4 + i) * 0.35;
        });
        pl.position.x = Math.cos(t*0.5) * 2.2;
        pl.position.y = Math.sin(t*0.35) * 1.8;
        renderer.render(scene, camera);
    }
    anim3d();

    document.addEventListener('mousemove', (e) => {
        const mx = (e.clientX / window.innerWidth  - 0.5) * 2;
        const my = (e.clientY / window.innerHeight - 0.5) * 2;
        camera.position.x = mx * 0.5;
        camera.position.y = -my * 0.5;
        camera.lookAt(0, 0, 0);
    });

    document.addEventListener('click', (e) => {
        const palette = ['#7C8CF8','#F8B4A0','#7EDCB5','#7BBFEA','#E99DB5'];
        for (let i = 0; i < 7; i++) {
            const d = document.createElement('div');
            const sz = 5 + Math.random()*5;
            d.style.cssText = `
                position:fixed;left:${e.clientX}px;top:${e.clientY}px;
                width:${sz}px;height:${sz}px;
                border-radius:50%;pointer-events:none;z-index:9999;
                background:${palette[Math.floor(Math.random()*5)]};
                transform:translate(-50%,-50%);opacity:0.9;
                transition:all 0.7s cubic-bezier(0.23,1,0.32,1);
            `;
            document.body.appendChild(d);
            requestAnimationFrame(() => {
                const ang = (i / 7) * Math.PI * 2;
                const dist = 35 + Math.random() * 50;
                d.style.left = `${e.clientX + Math.cos(ang)*dist}px`;
                d.style.top  = `${e.clientY + Math.sin(ang)*dist}px`;
                d.style.opacity = '0';
                d.style.transform = 'translate(-50%,-50%) scale(0)';
            });
            setTimeout(() => d.remove(), 800);
        }
    });
})();
</script>
""", unsafe_allow_html=True)


@st.cache_data
def load_cases():
    with open(DATA_DIR / "cases.json", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_tfidf():
    with open(MODEL_DIR / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    cases = load_cases()
    texts = [clean_text(c["text_full"]) for c in cases]
    vectors = vectorizer.transform(texts)
    return vectorizer, vectors


@st.cache_data
def load_embeddings():
    return np.load(MODEL_DIR / "indobert_embeddings.npy")


@st.cache_resource
def load_indobert():
    import torch
    from transformers import AutoModel, AutoTokenizer
    MODEL_NAME = "indobenchmark/indobert-base-p1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return tokenizer, model, device


INDO_COMMON_WORDS = {
    "ada", "adalah", "adanya", "agar", "akan", "akibat", "akhir",
    "alasan", "amar", "anak", "antara", "apa", "apabila",
    "atas", "atau", "ayat",
    "bagi", "bagian", "bahwa", "baik", "barang", "baru",
    "batu", "beberapa", "belum", "benar", "berapa",
    "berdasarkan", "berdiri", "berikut", "berkas", "bermaksud",
    "bersalah", "bersama", "berupa", "besar", "biasa", "biaya",
    "bila", "bisa", "boleh", "buah", "bukan", "bulan",
    "cara", "cukup",
    "dahulu", "dalam", "dan", "dapat", "dari", "datang",
    "dengan", "demi", "demikian", "demikianlah",
    "denda", "depan", "desa", "dimana", "dimaksud",
    "diatur", "diancam", "didakwa", "didakwakan", "dijatuhkan",
    "dinyatakan", "diperoleh", "dipersidangan",
    "duduk", "dusun",
    "empat", "enam",
    "fakta",
    "hak", "hal", "halaman", "hanya", "hari", "hakim",
    "harus", "haruslah", "hasil", "hendak", "hukum", "hukuman",
    "ini", "islam", "itu",
    "jadi", "jalan", "jaksa", "jam", "jenis",
    "jika", "juga", "jumlah",
    "kabupaten", "kalau", "kami", "karena", "karenanya",
    "kasus", "kata", "keadilan", "kebangsaan",
    "kecamatan", "kedua", "kelamin",
    "kelurahan", "keluarga", "kemudian", "kepada",
    "kepadanya", "kepentingan", "keperluan",
    "kerja", "kerugian", "kesadaran",
    "keterangan", "ketentuan", "ketiga", "ketika",
    "ketua", "ketuhanan",
    "kewajiban", "khususnya",
    "korban", "kota", "kuasa", "kunci", "kurang",
    "lagi", "lahir", "lain", "lalu", "lama",
    "langsung", "lebih", "lengkap", "lima",
    "maha", "maka", "maksud", "malang",
    "mampu", "mana", "masa", "masing",
    "masih", "masuk", "masyarakat", "maupun",
    "majelis", "meja",
    "melakukan", "melanggar", "melawan",
    "melepaskan", "memang", "membebaskan",
    "membawa", "membayar", "memberi", "memberikan",
    "memenuhi", "memiliki",
    "memohon", "mempertimbangkan", "memperhatikan",
    "memutuskan",
    "mendapat", "mendekati", "mendorong",
    "mendengar", "menerangkan",
    "mengadili", "mengajukan", "mengakui", "mengalami",
    "mengambil", "mengenai", "mengetahui",
    "menggunakan", "mengingat", "menguasai",
    "menimbang", "menikmati", "menjadi",
    "menjatuhkan", "menuju",
    "menurut", "menyatakan", "menyesal", "menyesali",
    "meresahkan", "merupakan", "meskipun",
    "milik", "miliki", "motor", "mulai",
    "nama", "namun", "negeri", "nomor",
    "orang", "oleh",
    "pada", "paling", "panitera", "para", "parkir",
    "pasal", "pekerjaan",
    "pelaku", "pemaaf", "pembenar",
    "pembacaan", "pemeriksaan",
    "pencurian", "pendapat", "penetapan",
    "pengganti", "pengetahuan",
    "penjara", "penjeraan",
    "penuntut", "penunjukan",
    "penyesuaian", "penyidik",
    "peradilan", "peraturan",
    "perbuatan", "perbuatannya",
    "perkara", "perlu", "pernah", "persidangan",
    "pertama", "pertimbangan", "perundang",
    "pidana", "pihak", "pokoknya",
    "pula", "pukul", "putusan",
    "rasa", "rekaman", "republik", "resto",
    "ringan", "rupiah",
    "saat", "saja", "saksi", "salah",
    "sama", "sampai", "sangat", "sarana",
    "satu", "sebagai", "sebagaimana",
    "sebagian", "sebelum", "sebelumnya",
    "secara", "sedang", "sedangkan",
    "sehingga", "sejak", "sejumlah", "sekira", "sekitar",
    "selain", "selaku", "selama", "selanjutnya",
    "seluruh", "seluruhnya",
    "semua", "sendiri", "sependapat",
    "sepeda", "sepengetahuan",
    "seperti", "sepatutnya",
    "serta", "sesuai", "sesuatu",
    "setelah", "setiap", "setimpal",
    "sifat", "sopan",
    "suatu", "sudah", "sumpah", "supaya", "surat",
    "tahun", "tanggal", "tanpa", "telah",
    "tempat", "tentang", "tentunya",
    "terbukti", "terhadap",
    "terdakwa", "terlebih", "termasuk", "ternyata",
    "tersebut", "terungkap",
    "tetap", "tetapi",
    "tidak", "tindak", "tinggal",
    "tujuan", "tulang", "tuntutan",
    "tujuh", "tiga", "tingkat",
    "umum", "umur", "undang",
    "unsur", "untuk", "utama",
    "wajib", "waktu", "walaupun", "warna", "wib",
}


def _remove_page_boundaries(raw_text):
    import re
    lines = raw_text.split('\n')
    content_lines = []
    in_disclaimer = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.lower() == 'disclaimer':
            in_disclaimer = True
            continue
        if in_disclaimer:
            if 'ext.318)' in stripped.lower():
                in_disclaimer = False
            continue
        if re.match(r'^halaman\s+\d+$', stripped, re.IGNORECASE):
            continue
        stripped = re.sub(
            r'\s*putusan\s+nomor\s+\d+/pid\.\w+/\d{4}/pn\s+\w+\s*$',
            '', stripped, flags=re.IGNORECASE
        )
        if stripped:
            content_lines.append(stripped)

    return ' '.join(content_lines)


def _split_merged_words(text, word_set, min_token_len=6, min_part_len=3):
    import re
    tokens = text.split()
    result = []
    for token in tokens:
        alpha_only = re.sub(r'[^a-z]', '', token)
        if len(alpha_only) <= min_token_len or alpha_only in word_set:
            result.append(token)
            continue
        best_split = None
        best_score = -1
        for i in range(min_part_len, len(alpha_only) - min_part_len + 1):
            left = alpha_only[:i]
            right = alpha_only[i:]
            if left in word_set and right in word_set:
                score = len(left) + len(right) + min(len(left), len(right))
                if score > best_score:
                    best_score = score
                    best_split = (left, right)
        if best_split:
            result.append(best_split[0])
            sub = _split_merged_words(best_split[1], word_set, min_token_len, min_part_len)
            result.append(sub)
        else:
            result.append(token)
    return ' '.join(result)


def clean_text(text):
    import re
    text = _remove_page_boundaries(text)
    text = text.lower()

    def fix_spaced_chars(m):
        return m.group(0).replace(' ', '')
    text = re.sub(r'(?<!\w)([a-z] ){3,}[a-z](?!\w)', fix_spaced_chars, text)

    text = re.sub(r';', '; ', text)
    text = re.sub(r'(?<=[a-z])\.(?=[a-z])', '. ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    text = _split_merged_words(text, INDO_COMMON_WORDS)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def get_embedding(text, tokenizer, model, device):
    import torch
    # Tokenize completely without truncation
    tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    
    chunk_size = 510  # Max length 512 minus 2 (for CLS and SEP)
    chunks = [tokens[j:j+chunk_size] for j in range(0, len(tokens), chunk_size)]
    
    chunk_embs = []
    for chunk in chunks:
        input_ids = torch.cat([
            torch.tensor([tokenizer.cls_token_id]), 
            chunk, 
            torch.tensor([tokenizer.sep_token_id])
        ]).unsqueeze(0).to(device)
        
        attention_mask = torch.ones_like(input_ids).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
        mask = attention_mask.unsqueeze(-1).float()
        emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        chunk_embs.append(emb)
        
    if chunk_embs:
        return torch.stack(chunk_embs).mean(dim=0).cpu().numpy()[0]
    else:
        return np.zeros(model.config.hidden_size)


def retrieve_tfidf(query, vectorizer, all_vectors, cases, k=5):
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, all_vectors)[0]
    top_k = np.argsort(sims)[::-1][:k]
    return [(cases[i], float(sims[i])) for i in top_k]


def retrieve_bert(query, tokenizer, model, device, all_embeddings, cases, k=5):
    q_emb = get_embedding(query, tokenizer, model, device).reshape(1, -1)
    sims = cosine_similarity(q_emb, all_embeddings)[0]
    top_k = np.argsort(sims)[::-1][:k]
    return [(cases[i], float(sims[i])) for i in top_k]


def majority_vote(results, key):
    values = [c.get(key, "") for c, _ in results if c.get(key)]
    if not values:
        return "tidak_diketahui"
    return Counter(values).most_common(1)[0][0]


def weighted_vote(results, key):
    scores = {}
    for c, sim in results:
        val = c.get(key, "")
        if val:
            scores[val] = scores.get(val, 0.0) + (sim ** 5)
    if not scores:
        return "tidak_diketahui"
    return max(scores, key=scores.get)


def predict_vonis_bulan(results, voting_method):
    if not results:
        return "?"
    if voting_method == "Majority Vote":
        # Average of all k results
        total_months = sum(c.get("vonis_bulan", 0) for c, _ in results)
        return f"{int(round(total_months / len(results)))} bulan"
    else:
        # Weighted average using sim^5
        total_weight = sum((sim ** 5) for _, sim in results)
        if total_weight == 0:
            return "?"
        weighted_months = sum(c.get("vonis_bulan", 0) * (sim ** 5) for c, sim in results)
        return f"{int(round(weighted_months / total_weight))} bulan"


def render_case_card(case, sim, rank):
    pasal_color = "#7EDCB5" if case["label_pasal"] == "pasal_362" else "#7C8CF8"
    vonis_bulan = case.get('vonis_bulan', 0)
    if vonis_bulan < 12:
        vonis_color = "#7EDCB5"
    elif vonis_bulan <= 36:
        vonis_color = "#7BBFEA"
    else:
        vonis_color = "#E99DB5"

    bar_width = max(int(sim * 100), 5)
    sim_pct   = f"{sim * 100:.1f}%"

    rank_bg = "#5B6AE0" if rank <= 3 else "#8E92AB"

    st.markdown(f"""
    <div class="case-card" style="animation: fadeIn 0.5s ease-out {rank * 0.08}s both;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div style="flex:1;">
                <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px;">
                    <span style="display:inline-flex;align-items:center;justify-content:center;width:26px;height:26px;border-radius:8px;background:{rank_bg};color:white;font-size:0.72rem;font-weight:700;font-family:'Inter',sans-serif;">{rank}</span>
                    <code style="background:rgba(124,140,248,0.1);color:#5B6AE0;padding:3px 9px;border-radius:6px;font-size:0.78rem;font-family:'Inter',sans-serif;">{case['case_id']}</code>
                    <span style="background:{pasal_color};color:white;padding:3px 10px;border-radius:10px;font-size:0.7rem;font-weight:600;letter-spacing:0.4px;">
                        {case['label_pasal'].replace('_',' ').upper()}
                    </span>
                    <span style="background:{vonis_color};color:white;padding:3px 10px;border-radius:10px;font-size:0.7rem;font-weight:600;">
                        Vonis {vonis_bulan} bln
                    </span>
                </div>
                <div style="font-size:0.78rem;color:#5A5F7E;line-height:1.5;">
                    <span style="margin-right:14px;"><strong style='color:#2D3250;'>No. Perkara:</strong> {case.get('no_perkara','-')}</span>
                    <span><strong style='color:#2D3250;'>Amar:</strong> {case.get('amar_putusan','-')[:60]}{'...' if len(str(case.get('amar_putusan',''))) > 60 else ''}</span>
                </div>
            </div>
            <div style="text-align:right;min-width:70px;">
                <div style="font-family:'DM Serif Display',serif;font-weight:400;font-size:1.3rem;color:#5B6AE0;">{sim:.3f}</div>
                <div style="font-size:0.65rem;color:#8E92AB;letter-spacing:0.8px;text-transform:uppercase;">Similarity</div>
            </div>
        </div>
        <div style="margin-top:8px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
                <span style="font-size:0.68rem;color:#8E92AB;">Tingkat Kesamaan</span>
                <span style="font-size:0.68rem;color:#5B6AE0;font-weight:600;">{sim_pct}</span>
            </div>
            <div style="background:rgba(124,140,248,0.1);height:6px;border-radius:4px;overflow:hidden;">
                <div style="width:{bar_width}%;height:6px;border-radius:4px;background:linear-gradient(90deg,#5B6AE0,#7C8CF8,#F8B4A0);transition:width 0.8s ease;"></div>
            </div>
        </div>
        <details style="margin-top:8px;">
            <summary style="cursor:pointer;font-size:0.76rem;color:#5B6AE0;font-weight:500;letter-spacing:0.3px;user-select:none;">Lihat ringkasan fakta</summary>
            <p style="font-size:0.75rem;color:#5A5F7E;margin-top:6px;line-height:1.6;background:rgba(124,140,248,0.04);padding:8px 12px;border-radius:8px;">
                {case.get('ringkasan_fakta','Tidak tersedia')[:500]}{'...' if len(str(case.get('ringkasan_fakta',''))) > 500 else ''}
            </p>
        </details>
    </div>
    """, unsafe_allow_html=True)


def page_predict():
    st.markdown("""
    <div style="text-align:center;padding:1.5rem 0 0.5rem;">
        <div class="hero-badge">Case-Based Reasoning System</div>
    </div>
    <div class="main-header">Putusan Pencurian</div>
    <div class="sub-header">
        Sistem analisis cerdas putusan pidana pencurian &mdash; Pengadilan Negeri Malang<br>
        <span style="color:#7EDCB5;font-weight:600;">Pasal 362</span> (Pencurian Biasa) &nbsp;vs&nbsp;
        <span style="color:#7C8CF8;font-weight:600;">Pasal 363</span> (Pencurian dengan Pemberatan)
    </div>
    """, unsafe_allow_html=True)

    cases = load_cases()
    vectorizer, all_vectors = load_tfidf()
    all_embeddings = load_embeddings()

    with st.sidebar:
        st.header("Pengaturan")
        k = st.slider("Jumlah kasus serupa (k)", 1, 10, 5)
        method = st.radio("Metode Retrieval", ["TF-IDF", "IndoBERT", "Keduanya"], index=2)
        voting = st.radio("Metode Voting", ["Majority Vote", "Weighted Vote"], index=1)

        st.divider()
        st.header("Info Case Base")
        st.metric("Total Kasus", len(cases))
        pasal_362 = sum(1 for c in cases if c["label_pasal"] == "pasal_362")
        pasal_363 = sum(1 for c in cases if c["label_pasal"] == "pasal_363")
        col1, col2 = st.columns(2)
        col1.metric("Pasal 362", pasal_362)
        col2.metric("Pasal 363", pasal_363)

    tab1, tab2, tab3 = st.tabs(["Prediksi Kasus Baru", "Contoh Pengujian", "Metrik Evaluasi"])

    with tab1:
        st.subheader("Pilih Contoh Kasus (Opsional)")
        contoh_options = ["-- Ketik/Upload Sendiri --"]
        
        import random
        random.seed(42)
        s_363 = random.sample([c for c in cases if c['label_pasal'] == 'pasal_363'], 2)
        s_362 = random.sample([c for c in cases if c['label_pasal'] == 'pasal_362'], 2)
        samples = s_363 + s_362
        
        sample_dict = {}
        for i, s in enumerate(samples, 1):
            label = f"Contoh {i}: {s['label_pasal']} (Vonis {s.get('vonis_bulan','?')} bln)"
            contoh_options.append(label)
            sample_dict[label] = s
            
        selected_contoh = st.selectbox("Gunakan kasus dari data latih sebagai contoh:", contoh_options)
        
        default_val = ""
        if selected_contoh != "-- Ketik/Upload Sendiri --":
            s_data = sample_dict[selected_contoh]
            default_val = s_data["text_full"]
            st.info(f"**Prediksi Seharusnya:** {s_data['label_pasal']} | **Vonis:** {s_data.get('vonis_bulan','?')} bulan")
            
        st.subheader("Masukkan Teks Putusan")
        query_text = st.text_area(
            "Paste teks putusan pengadilan di sini:",
            height=250,
            value=default_val,
            placeholder="Contoh: PUTUSAN Nomor xxx/Pid.B/2024/PN Mlg DEMI KEADILAN BERDASARKAN KETUHANAN YANG MAHA ESA ..."
        )

        uploaded_file = st.file_uploader("Atau upload file .txt", type=["txt"])
        if uploaded_file is not None:
            query_text = uploaded_file.read().decode("utf-8", errors="ignore")
            st.success(f"File '{uploaded_file.name}' berhasil dimuat ({len(query_text)} karakter)")

        if st.button("Analisis Kasus", type="primary", use_container_width=True):
            if not query_text or len(query_text.strip()) < 50:
                st.error("Teks terlalu pendek. Masukkan minimal 50 karakter teks putusan.")
                return

            cleaned = clean_text(query_text)

            tfidf_results = []
            bert_results = []

            with st.spinner("Menganalisis kasus..."):
                if method in ["TF-IDF", "Keduanya"]:
                    tfidf_results = retrieve_tfidf(cleaned, vectorizer, all_vectors, cases, k)

                if method in ["IndoBERT", "Keduanya"]:
                    tokenizer, model, device = load_indobert()
                    bert_results = retrieve_bert(cleaned, tokenizer, model, device, all_embeddings, cases, k)

            vote_fn = majority_vote if voting == "Majority Vote" else weighted_vote

            col1, col2, col3, col4 = st.columns(4)
            if tfidf_results:
                pred_pasal = vote_fn(tfidf_results, "label_pasal")
                pred_vonis = predict_vonis_bulan(tfidf_results, voting)
                col1.markdown(f"""<div class="metric-card"><h3>TF-IDF - Pasal</h3><h1>{pred_pasal}</h1></div>""", unsafe_allow_html=True)
                col2.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#f093fb,#f5576c);"><h3>TF-IDF - Vonis</h3><h1>{pred_vonis}</h1></div>""", unsafe_allow_html=True)

            if bert_results:
                pred_pasal_b = vote_fn(bert_results, "label_pasal")
                pred_vonis_b = predict_vonis_bulan(bert_results, voting)
                col3.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#4facfe,#00f2fe);"><h3>IndoBERT - Pasal</h3><h1>{pred_pasal_b}</h1></div>""", unsafe_allow_html=True)
                col4.markdown(f"""<div class="metric-card" style="background:linear-gradient(135deg,#43e97b,#38f9d7);"><h3>IndoBERT - Vonis</h3><h1>{pred_vonis_b}</h1></div>""", unsafe_allow_html=True)

            st.divider()

            if method == "Keduanya":
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("TF-IDF Top-K")
                    for i, (case, sim) in enumerate(tfidf_results):
                        render_case_card(case, sim, i + 1)
                with c2:
                    st.subheader("IndoBERT Top-K")
                    for i, (case, sim) in enumerate(bert_results):
                        render_case_card(case, sim, i + 1)
            elif tfidf_results:
                st.subheader("TF-IDF Top-K")
                for i, (case, sim) in enumerate(tfidf_results):
                    render_case_card(case, sim, i + 1)
            elif bert_results:
                st.subheader("IndoBERT Top-K")
                for i, (case, sim) in enumerate(bert_results):
                    render_case_card(case, sim, i + 1)

    with tab2:
        st.subheader("Contoh Pengujian dengan Data dari Case Base")
        st.info("Pilih salah satu kasus dari case base untuk melihat bagaimana sistem bekerja. "
                "Kasus yang dipilih akan digunakan sebagai query dan dibandingkan dengan seluruh case base.")

        case_ids = [c["case_id"] for c in cases]
        selected_id = st.selectbox("Pilih kasus:", case_ids, index=0)
        selected_case = next(c for c in cases if c["case_id"] == selected_id)

        col1, col2 = st.columns(2)
        col1.metric("Pasal", selected_case["label_pasal"])
        col2.metric("Vonis (bulan)", f"{selected_case.get('vonis_bulan', '?')} bulan")

        with st.expander("Lihat teks putusan (500 karakter pertama)"):
            st.text(selected_case["text_full"][:500])

        if st.button("Jalankan Pencarian Serupa", type="primary"):
            cleaned = clean_text(selected_case["text_full"])

            with st.spinner("Menganalisis..."):
                tfidf_res = retrieve_tfidf(cleaned, vectorizer, all_vectors, cases, k=5)
                tokenizer, model, device = load_indobert()
                bert_res = retrieve_bert(cleaned, tokenizer, model, device, all_embeddings, cases, k=5)

            st.success(f"Ground Truth: **{selected_case['label_pasal']}** | "
                       f"Vonis: **{selected_case.get('vonis_bulan','?')} bulan**")

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("TF-IDF Top-5")
                for i, (case, sim) in enumerate(tfidf_res):
                    render_case_card(case, sim, i + 1)
            with c2:
                st.subheader("IndoBERT Top-5")
                for i, (case, sim) in enumerate(bert_res):
                    render_case_card(case, sim, i + 1)

    with tab3:
        st.subheader("Metrik Evaluasi Model")

        if (EVAL_DIR / "retrieval_metrics.csv").exists():
            df_ret = pd.read_csv(EVAL_DIR / "retrieval_metrics.csv")
            st.markdown("#### Retrieval Metrics (Tahap 3)")
            st.dataframe(df_ret, use_container_width=True, hide_index=True)

        if (EVAL_DIR / "prediction_metrics.csv").exists():
            df_pred = pd.read_csv(EVAL_DIR / "prediction_metrics.csv")
            st.markdown("#### Prediction Metrics (Tahap 4-5)")
            st.dataframe(df_pred, use_container_width=True, hide_index=True)

        if (EVAL_DIR / "evaluation_chart.png").exists():
            st.markdown("#### Grafik Perbandingan Model")
            st.image(str(EVAL_DIR / "evaluation_chart.png"), use_container_width=True)

        st.markdown("#### Ringkasan Temuan")
        st.markdown("""
        | Aspek | Temuan |
        |---|---|
        | **Model terbaik (Pasal)** | TF-IDF + SVM (F1=0.9161) |
        | **Rekomendasi** | Gunakan TF-IDF untuk teks ringkasan, perbaiki ekstraksi data untuk model IndoBERT |
        """)


if __name__ == "__main__":
    page_predict()
