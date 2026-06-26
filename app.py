import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

KURS_USD_TO_IDR = 16_800

st.set_page_config(
    page_title="HealthCost AI — Prediksi Biaya Asuransi",
    page_icon="H",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Outfit:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary: #06060a;
    --bg-secondary: #0c0c14;
    --bg-card: rgba(14,14,22,0.88);
    --bg-card-solid: #0e0e16;
    --accent-gold: #d4a844;
    --accent-yellow: #e8c252;
    --accent-warm: #c4883a;
    --accent-gradient: linear-gradient(135deg, #e8c252 0%, #d4a844 35%, #c4883a 70%, #a86832 100%);
    --text-primary: #e8e6e3;
    --text-secondary: #8a8a96;
    --text-muted: #55556a;
    --border-color: rgba(212,168,68,0.08);
    --border-glow: rgba(212,168,68,0.22);
    --glass-border: rgba(212,168,68,0.1);
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Outfit', sans-serif !important;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 700px 500px at 15% 15%, rgba(212,168,68,0.04) 0%, transparent 70%),
        radial-gradient(ellipse 500px 600px at 85% 85%, rgba(168,104,50,0.03) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: var(--bg-secondary) !important; }

.main .block-container {
    padding-top: 1rem !important;
    max-width: 780px !important;
    position: relative;
    z-index: 1;
}

h1, h2, h3, h4, h5, h6, p, label {
    color: var(--text-primary) !important;
    font-family: 'Outfit', sans-serif !important;
}

.main .block-container div,
.main .block-container span {
    font-family: 'Outfit', sans-serif !important;
}

[data-testid="stExpander"] {
    border: 1px solid var(--glass-border) !important;
    border-radius: 14px !important;
    background: var(--bg-card) !important;
    overflow: hidden;
    margin-bottom: 1rem;
}

[data-testid="stExpander"] summary {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 0.8rem 1.2rem !important;
}

[data-testid="stExpander"] summary:hover {
    color: var(--accent-gold) !important;
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    border-top: 1px solid var(--border-color) !important;
}

.batik-left {
    position: fixed;
    top: 0; left: 0;
    width: 80px; height: 100vh;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}

.batik-left::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        repeating-linear-gradient(
            180deg,
            transparent 0px,
            transparent 30px,
            rgba(212,168,68,0.03) 30px,
            rgba(212,168,68,0.03) 31px
        ),
        repeating-linear-gradient(
            135deg,
            transparent 0px,
            transparent 20px,
            rgba(212,168,68,0.04) 20px,
            rgba(212,168,68,0.04) 21px
        ),
        repeating-linear-gradient(
            45deg,
            transparent 0px,
            transparent 20px,
            rgba(212,168,68,0.04) 20px,
            rgba(212,168,68,0.04) 21px
        );
}

.batik-left::after {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle 8px at 40px 25px, rgba(212,168,68,0.06) 0%, transparent 100%),
        radial-gradient(circle 5px at 40px 25px, rgba(212,168,68,0.04) 0%, transparent 100%),
        radial-gradient(circle 8px at 40px 75px, rgba(212,168,68,0.06) 0%, transparent 100%),
        radial-gradient(circle 5px at 40px 75px, rgba(212,168,68,0.04) 0%, transparent 100%),
        radial-gradient(circle 8px at 40px 125px, rgba(212,168,68,0.06) 0%, transparent 100%),
        radial-gradient(circle 5px at 40px 125px, rgba(212,168,68,0.04) 0%, transparent 100%),
        radial-gradient(circle 8px at 40px 175px, rgba(212,168,68,0.06) 0%, transparent 100%),
        radial-gradient(circle 5px at 40px 175px, rgba(212,168,68,0.04) 0%, transparent 100%);
    background-size: 80px 200px;
}

.batik-right {
    position: fixed;
    top: 0; right: 0;
    width: 80px; height: 100vh;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}

.batik-right::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        repeating-linear-gradient(
            180deg,
            transparent 0px,
            transparent 40px,
            rgba(212,168,68,0.025) 40px,
            rgba(212,168,68,0.025) 41px
        ),
        repeating-linear-gradient(
            135deg,
            transparent 0px,
            transparent 25px,
            rgba(212,168,68,0.03) 25px,
            rgba(212,168,68,0.03) 26px
        ),
        repeating-linear-gradient(
            45deg,
            transparent 0px,
            transparent 25px,
            rgba(212,168,68,0.03) 25px,
            rgba(212,168,68,0.03) 26px
        );
}

.batik-right::after {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle 6px at 40px 50px, rgba(212,168,68,0.05) 0%, transparent 100%),
        radial-gradient(circle 3px at 40px 50px, rgba(212,168,68,0.03) 0%, transparent 100%),
        radial-gradient(circle 10px at 40px 50px, rgba(212,168,68,0.03) 0%, transparent 100%),
        radial-gradient(circle 6px at 40px 150px, rgba(212,168,68,0.05) 0%, transparent 100%),
        radial-gradient(circle 3px at 40px 150px, rgba(212,168,68,0.03) 0%, transparent 100%),
        radial-gradient(circle 10px at 40px 150px, rgba(212,168,68,0.03) 0%, transparent 100%);
    background-size: 80px 200px;
}

.batik-bottom {
    position: fixed;
    bottom: 0; left: 0;
    width: 100%; height: 70px;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}

.batik-bottom::before {
    content: '';
    position: absolute;
    inset: 0;
    background:
        repeating-linear-gradient(
            90deg,
            transparent 0px,
            transparent 30px,
            rgba(212,168,68,0.03) 30px,
            rgba(212,168,68,0.03) 31px
        ),
        repeating-linear-gradient(
            135deg,
            transparent 0px,
            transparent 18px,
            rgba(212,168,68,0.035) 18px,
            rgba(212,168,68,0.035) 19px
        ),
        repeating-linear-gradient(
            45deg,
            transparent 0px,
            transparent 18px,
            rgba(212,168,68,0.035) 18px,
            rgba(212,168,68,0.035) 19px
        );
}

.batik-bottom::after {
    content: '';
    position: absolute;
    inset: 0;
    background:
        radial-gradient(circle 6px at 50px 35px, rgba(212,168,68,0.05) 0%, transparent 100%),
        radial-gradient(circle 3px at 50px 35px, rgba(212,168,68,0.03) 0%, transparent 100%),
        radial-gradient(circle 6px at 150px 35px, rgba(212,168,68,0.05) 0%, transparent 100%),
        radial-gradient(circle 3px at 150px 35px, rgba(212,168,68,0.03) 0%, transparent 100%),
        radial-gradient(circle 6px at 250px 35px, rgba(212,168,68,0.05) 0%, transparent 100%),
        radial-gradient(circle 3px at 250px 35px, rgba(212,168,68,0.03) 0%, transparent 100%);
    background-size: 300px 70px;
}

.scan-line {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 1px;
    background: linear-gradient(90deg, transparent 10%, rgba(212,168,68,0.2) 50%, transparent 90%);
    animation: scanMove 8s linear infinite;
    pointer-events: none;
    z-index: 9999;
}

@keyframes scanMove {
    0%   { top: -2px; }
    100% { top: 100%; }
}

.hero-container {
    text-align: center;
    padding: 2.5rem 1rem 1.2rem 1rem;
    animation: heroAppear 1s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

@keyframes heroAppear {
    from { opacity: 0; transform: translateY(-25px); }
    to   { opacity: 1; transform: translateY(0); }
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 20px;
    background: rgba(212,168,68,0.06);
    border: 1px solid rgba(212,168,68,0.15);
    border-radius: 50px;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    color: var(--accent-gold) !important;
    -webkit-text-fill-color: var(--accent-gold) !important;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    margin-bottom: 1.4rem;
    animation: badgePulse 4s ease-in-out infinite;
}

.hero-badge-dot {
    width: 6px; height: 6px;
    background: var(--accent-gold);
    border-radius: 50%;
    display: inline-block;
    animation: dotPulse 2s ease-in-out infinite;
}

@keyframes dotPulse {
    0%, 100% { opacity: 0.4; transform: scale(0.8); }
    50%      { opacity: 1;   transform: scale(1.2); }
}

@keyframes badgePulse {
    0%, 100% { box-shadow: 0 0 8px rgba(212,168,68,0.05); }
    50%      { box-shadow: 0 0 20px rgba(212,168,68,0.12); }
}

.hero-icon-wrap {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 88px; height: 88px;
    margin-bottom: 1rem;
}

.hero-icon-symbol {
    font-family: 'Orbitron', sans-serif;
    font-size: 2rem;
    font-weight: 900;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    position: relative;
    z-index: 2;
    animation: iconFloat 4s ease-in-out infinite;
}

@keyframes iconFloat {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-6px); }
}

.hero-icon-ring {
    position: absolute;
    top: 50%; left: 50%;
    width: 72px; height: 72px;
    border: 1.5px solid rgba(212,168,68,0.15);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    animation: ringPulse 3s ease-in-out infinite;
}

.hero-icon-ring-outer {
    position: absolute;
    top: 50%; left: 50%;
    width: 88px; height: 88px;
    border: 1px solid rgba(212,168,68,0.08);
    border-radius: 50%;
    transform: translate(-50%, -50%);
    animation: ringPulse 3s ease-in-out infinite 0.6s;
}

.hero-icon-corner {
    position: absolute;
    width: 14px; height: 14px;
    border: 1.5px solid rgba(212,168,68,0.2);
}

.hero-icon-corner.tl { top: 2px; left: 2px; border-right: none; border-bottom: none; }
.hero-icon-corner.tr { top: 2px; right: 2px; border-left: none; border-bottom: none; }
.hero-icon-corner.bl { bottom: 2px; left: 2px; border-right: none; border-top: none; }
.hero-icon-corner.br { bottom: 2px; right: 2px; border-left: none; border-top: none; }

@keyframes ringPulse {
    0%, 100% { transform: translate(-50%, -50%) scale(1);    opacity: 0.4; }
    50%      { transform: translate(-50%, -50%) scale(1.08); opacity: 0.9; }
}

.hero-title {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 2.4rem !important;
    font-weight: 800 !important;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem !important;
    letter-spacing: 4px;
}

.hero-subtitle {
    font-size: 0.88rem !important;
    color: var(--text-secondary) !important;
    -webkit-text-fill-color: var(--text-secondary) !important;
    font-weight: 400;
    letter-spacing: 0.3px;
    line-height: 1.6;
    max-width: 500px;
    margin: 0 auto;
    opacity: 0;
    animation: subtitleFade 1s ease forwards 0.4s;
}

@keyframes subtitleFade { to { opacity: 1; } }

.hero-divider {
    width: 50px; height: 2px;
    background: var(--accent-gradient);
    margin: 1.2rem auto 0 auto;
    border-radius: 1px;
    transform: scaleX(0);
    animation: dividerExpand 0.8s ease forwards 0.7s;
}

@keyframes dividerExpand { to { transform: scaleX(1); } }

.flow-container {
    padding: 1.2rem 0.5rem;
    margin: 0.2rem 0 1.5rem 0;
    opacity: 0;
    animation: fadeInUp 0.8s ease forwards 0.3s;
}

.flow-label {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.6rem !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    -webkit-text-fill-color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 3px;
    text-align: center;
    margin-bottom: 1.2rem !important;
}

.flow-track {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
}

.flow-node {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
}

.flow-node-box {
    width: 50px; height: 50px;
    background: var(--bg-card-solid);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}

.flow-node-icon {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.55rem;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    transition: all 0.3s;
}

.flow-node:hover .flow-node-box {
    border-color: var(--accent-gold);
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(212,168,68,0.15);
}

.flow-node:hover .flow-node-icon {
    color: var(--accent-gold) !important;
    -webkit-text-fill-color: var(--accent-gold) !important;
}

.flow-node-text {
    font-size: 0.56rem !important;
    font-weight: 600 !important;
    color: var(--text-muted) !important;
    -webkit-text-fill-color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    text-align: center;
    width: 68px;
    transition: all 0.3s;
}

.flow-node:hover .flow-node-text {
    color: var(--accent-gold) !important;
    -webkit-text-fill-color: var(--accent-gold) !important;
}

.flow-connector {
    display: flex;
    align-items: center;
    padding: 0 4px;
    margin-bottom: 22px;
    position: relative;
    height: 5px;
}

.flow-connector-line {
    width: 26px; height: 1px;
    background: rgba(212,168,68,0.12);
    position: relative;
}

.flow-connector-line::after {
    content: '';
    position: absolute;
    right: -4px; top: -3px;
    width: 6px; height: 6px;
    border: solid rgba(212,168,68,0.25);
    border-width: 0 1.5px 1.5px 0;
    transform: rotate(-45deg);
}

.flow-connector-dot {
    position: absolute;
    top: -1px; left: 0;
    width: 8px; height: 3px;
    background: var(--accent-gold);
    border-radius: 2px;
    filter: blur(1px);
    opacity: 0;
}

.flow-track .flow-connector:nth-of-type(1) .flow-connector-dot { animation: connPulse 6s linear infinite 0.0s; }
.flow-track .flow-connector:nth-of-type(2) .flow-connector-dot { animation: connPulse 6s linear infinite 1.2s; }
.flow-track .flow-connector:nth-of-type(3) .flow-connector-dot { animation: connPulse 6s linear infinite 2.4s; }
.flow-track .flow-connector:nth-of-type(4) .flow-connector-dot { animation: connPulse 6s linear infinite 3.6s; }

@keyframes connPulse {
    0%  { left: -4px; opacity: 0; }
    5%  { opacity: 1; }
    20% { left: 100%; opacity: 0; }
    100%{ left: 100%; opacity: 0; }
}

@keyframes flowLoop {
    0%    { border-color: var(--border-color); box-shadow: none; }
    10%   { border-color: var(--accent-gold);  box-shadow: 0 0 18px rgba(212,168,68,0.2); }
    25%   { border-color: var(--border-color); box-shadow: none; }
    100%  { border-color: var(--border-color); box-shadow: none; }
}

@keyframes flowLoopText {
    0%    { color: var(--text-muted); -webkit-text-fill-color: var(--text-muted); }
    10%   { color: var(--accent-gold); -webkit-text-fill-color: var(--accent-gold); }
    25%   { color: var(--text-muted); -webkit-text-fill-color: var(--text-muted); }
    100%  { color: var(--text-muted); -webkit-text-fill-color: var(--text-muted); }
}

.flow-track .flow-node:nth-child(1) .flow-node-box  { animation: flowLoop 6s ease-in-out infinite 0.0s; }
.flow-track .flow-node:nth-child(3) .flow-node-box  { animation: flowLoop 6s ease-in-out infinite 1.2s; }
.flow-track .flow-node:nth-child(5) .flow-node-box  { animation: flowLoop 6s ease-in-out infinite 2.4s; }
.flow-track .flow-node:nth-child(7) .flow-node-box  { animation: flowLoop 6s ease-in-out infinite 3.6s; }
.flow-track .flow-node:nth-child(9) .flow-node-box  { animation: flowLoop 6s ease-in-out infinite 4.8s; }

.flow-track .flow-node:nth-child(1) .flow-node-icon { animation: flowLoopText 6s ease-in-out infinite 0.0s; }
.flow-track .flow-node:nth-child(3) .flow-node-icon { animation: flowLoopText 6s ease-in-out infinite 1.2s; }
.flow-track .flow-node:nth-child(5) .flow-node-icon { animation: flowLoopText 6s ease-in-out infinite 2.4s; }
.flow-track .flow-node:nth-child(7) .flow-node-icon { animation: flowLoopText 6s ease-in-out infinite 3.6s; }
.flow-track .flow-node:nth-child(9) .flow-node-icon { animation: flowLoopText 6s ease-in-out infinite 4.8s; }

.flow-track .flow-node:nth-child(1) .flow-node-text { animation: flowLoopText 6s ease-in-out infinite 0.0s; }
.flow-track .flow-node:nth-child(3) .flow-node-text { animation: flowLoopText 6s ease-in-out infinite 1.2s; }
.flow-track .flow-node:nth-child(5) .flow-node-text { animation: flowLoopText 6s ease-in-out infinite 2.4s; }
.flow-track .flow-node:nth-child(7) .flow-node-text { animation: flowLoopText 6s ease-in-out infinite 3.6s; }
.flow-track .flow-node:nth-child(9) .flow-node-text { animation: flowLoopText 6s ease-in-out infinite 4.8s; }

.card {
    background: var(--bg-card);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid var(--glass-border);
    border-radius: 18px;
    padding: 2rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    animation: cardSlideIn 0.7s ease forwards;
    opacity: 0;
    transform: translateY(16px);
}

.card:nth-of-type(1) { animation-delay: 0.15s; }
.card:nth-of-type(2) { animation-delay: 0.3s; }

@keyframes cardSlideIn {
    to { opacity: 1; transform: translateY(0); }
}

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, rgba(212,168,68,0.2) 50%, transparent 100%);
    opacity: 0;
    transition: opacity 0.4s;
}

.card:hover::before { opacity: 1; }

.card:hover {
    border-color: var(--border-glow);
    box-shadow: 0 8px 40px rgba(212,168,68,0.05), inset 0 1px 0 rgba(212,168,68,0.06);
    transform: translateY(-2px);
}

.card-title {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.62rem !important;
    font-weight: 600 !important;
    color: var(--accent-gold) !important;
    -webkit-text-fill-color: var(--accent-gold) !important;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 1.4rem !important;
    display: flex;
    align-items: center;
    gap: 12px;
}

.card-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border-glow), transparent 80%);
}

.result-card {
    background: linear-gradient(145deg, rgba(212,168,68,0.06) 0%, rgba(168,104,50,0.03) 50%, rgba(212,168,68,0.06) 100%);
    border: 1px solid rgba(212,168,68,0.15);
    border-radius: 22px;
    padding: 2.8rem 2rem;
    text-align: center;
    margin: 1.5rem 0;
    position: relative;
    overflow: hidden;
    animation: resultReveal 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

.result-card::before {
    content: '';
    position: absolute;
    top: -1px; left: -1px; right: -1px; bottom: -1px;
    border-radius: 23px;
    background: linear-gradient(135deg, rgba(212,168,68,0.15), transparent 40%, transparent 60%, rgba(212,168,68,0.1));
    z-index: -1;
    animation: borderShimmer 6s ease-in-out infinite;
}

@keyframes borderShimmer {
    0%, 100% { opacity: 0.5; }
    50%      { opacity: 1; }
}

@keyframes resultReveal {
    from { opacity: 0; transform: scale(0.95) translateY(15px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
}

.result-label {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.6rem !important;
    font-weight: 600 !important;
    color: var(--accent-gold) !important;
    -webkit-text-fill-color: var(--accent-gold) !important;
    text-transform: uppercase;
    letter-spacing: 3.5px;
    margin-bottom: 1rem !important;
}

.result-amount {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 900 !important;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0.5rem 0 !important;
    letter-spacing: 2px;
    opacity: 0;
    transform: scale(0.6);
    animation: amountPop 0.7s cubic-bezier(0.16, 1, 0.3, 1) forwards 0.3s;
}

@keyframes amountPop {
    to { opacity: 1; transform: scale(1); }
}

.result-usd {
    font-size: 0.8rem !important;
    color: var(--text-muted) !important;
    -webkit-text-fill-color: var(--text-muted) !important;
    letter-spacing: 1px;
    margin-top: 0.2rem !important;
    opacity: 0;
    animation: subtitleFade 0.5s ease forwards 0.6s;
}

.result-divider {
    width: 40px; height: 1px;
    background: var(--accent-gradient);
    margin: 1rem auto;
    border-radius: 1px;
}

.result-note {
    font-size: 0.78rem !important;
    color: var(--text-muted) !important;
    -webkit-text-fill-color: var(--text-muted) !important;
    letter-spacing: 0.5px;
}

.bmi-calc-card {
    background: rgba(14,14,22,0.7);
    border: 1px solid var(--glass-border);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    animation: cardSlideIn 0.5s ease forwards;
}

.bmi-calc-title {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.58rem !important;
    font-weight: 600 !important;
    color: var(--accent-gold) !important;
    -webkit-text-fill-color: var(--accent-gold) !important;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    margin-bottom: 0.8rem !important;
    display: flex;
    align-items: center;
    gap: 10px;
}

.bmi-calc-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border-glow), transparent 80%);
}

.bmi-result-box {
    background: rgba(10,10,16,0.6);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 0.8rem;
    animation: fadeInUp 0.4s ease forwards;
}

.bmi-result-left {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.bmi-result-label {
    font-size: 0.58rem !important;
    color: var(--text-muted) !important;
    -webkit-text-fill-color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 500 !important;
}

.bmi-result-value {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.bmi-category {
    padding: 5px 14px;
    border-radius: 8px;
    font-size: 0.68rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.bmi-underweight {
    background: rgba(96,165,250,0.1);
    border: 1px solid rgba(96,165,250,0.2);
    color: #60a5fa !important;
    -webkit-text-fill-color: #60a5fa !important;
}

.bmi-normal {
    background: rgba(74,222,128,0.1);
    border: 1px solid rgba(74,222,128,0.2);
    color: #4ade80 !important;
    -webkit-text-fill-color: #4ade80 !important;
}

.bmi-overweight {
    background: rgba(251,191,36,0.1);
    border: 1px solid rgba(251,191,36,0.2);
    color: #fbbf24 !important;
    -webkit-text-fill-color: #fbbf24 !important;
}

.bmi-obese {
    background: rgba(248,113,113,0.1);
    border: 1px solid rgba(248,113,113,0.2);
    color: #f87171 !important;
    -webkit-text-fill-color: #f87171 !important;
}

.bmi-use-btn {
    margin-top: 0.6rem;
    font-size: 0.7rem !important;
    color: var(--accent-gold) !important;
    -webkit-text-fill-color: var(--accent-gold) !important;
    letter-spacing: 0.5px;
}

.detail-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 0.6rem;
    margin-top: 1rem;
}

.detail-item {
    background: rgba(10,10,16,0.5);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 0.9rem;
    transition: all 0.35s ease;
    position: relative;
    overflow: hidden;
}

.detail-item::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 1px;
    background: var(--accent-gradient);
    transform: scaleX(0);
    transition: transform 0.4s ease;
    transform-origin: left;
}

.detail-item:hover::after { transform: scaleX(1); }

.detail-item:hover {
    border-color: rgba(212,168,68,0.18);
    transform: translateY(-2px);
}

.detail-item-label {
    font-size: 0.58rem !important;
    color: var(--text-muted) !important;
    -webkit-text-fill-color: var(--text-muted) !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.25rem !important;
    font-weight: 500 !important;
}

.detail-item-value {
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    -webkit-text-fill-color: var(--text-primary) !important;
}

.alert-box {
    border-radius: 14px;
    padding: 1rem 1.3rem;
    margin-top: 0.7rem;
    font-size: 0.82rem;
    display: flex;
    align-items: center;
    gap: 14px;
    backdrop-filter: blur(10px);
    opacity: 0;
    transform: translateX(-15px);
    animation: alertSlide 0.5s ease forwards;
}

.alert-box:nth-child(1) { animation-delay: 0.5s; }
.alert-box:nth-child(2) { animation-delay: 0.7s; }

@keyframes alertSlide {
    to { opacity: 1; transform: translateX(0); }
}

.alert-warning {
    background: rgba(212,168,68,0.06);
    border: 1px solid rgba(212,168,68,0.12);
    color: var(--accent-yellow) !important;
    -webkit-text-fill-color: var(--accent-yellow) !important;
}

.alert-info {
    background: rgba(100,140,200,0.06);
    border: 1px solid rgba(100,140,200,0.12);
    color: #7ba3cf !important;
    -webkit-text-fill-color: #7ba3cf !important;
}

.alert-icon {
    width: 28px; height: 28px;
    min-width: 28px;
    background: rgba(212,168,68,0.1);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Orbitron', sans-serif;
    font-size: 0.6rem;
    font-weight: 700;
    color: var(--accent-gold);
}

.alert-icon-info {
    background: rgba(100,140,200,0.1);
    color: #7ba3cf;
}

.footer {
    text-align: center;
    padding: 2.5rem 0 2rem 0;
    margin-top: 2rem;
    position: relative;
    opacity: 0;
    animation: footerFade 1s ease forwards 0.5s;
}

.footer::before {
    content: '';
    position: absolute;
    top: 0; left: 15%; right: 15%;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(212,168,68,0.15), transparent);
}

.footer-brand {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem !important;
    letter-spacing: 3px;
}

.footer-text {
    font-size: 0.68rem !important;
    color: var(--text-muted) !important;
    -webkit-text-fill-color: var(--text-muted) !important;
    letter-spacing: 0.3px;
    line-height: 1.8;
}

@keyframes footerFade { to { opacity: 1; } }

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

[data-testid="stNumberInput"] > div > div > input,
[data-testid="stSelectbox"] > div > div {
    background-color: rgba(10,10,16,0.9) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
    font-family: 'Outfit', sans-serif !important;
}

[data-testid="stNumberInput"] > div > div > input:focus {
    border-color: rgba(212,168,68,0.4) !important;
    box-shadow: 0 0 0 3px rgba(212,168,68,0.08), 0 0 15px rgba(212,168,68,0.05) !important;
}

div[data-testid="stSelectbox"] > div > div {
    background-color: rgba(10,10,16,0.9) !important;
}

button[kind="primary"] {
    background: var(--accent-gradient) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.75rem !important;
    padding: 0.9rem 2rem !important;
    letter-spacing: 2px;
    text-transform: uppercase;
    transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1) !important;
    position: relative;
    overflow: hidden;
}

button[kind="primary"]::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
    transition: left 0.6s ease;
}

button[kind="primary"]:hover::before { left: 100%; }

button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(212,168,68,0.3), 0 0 50px rgba(212,168,68,0.08) !important;
}

button[kind="primary"]:active {
    transform: translateY(0) scale(0.98) !important;
}

label {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: var(--text-secondary) !important;
    -webkit-text-fill-color: var(--text-secondary) !important;
    letter-spacing: 0.3px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="batik-left"></div>
<div class="batik-right"></div>
<div class="batik-bottom"></div>
<div class="scan-line"></div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(base_dir, 'model/rf_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(base_dir, 'model/columns.pkl'), 'rb') as f:
        columns = pickle.load(f)
    return model, columns

base_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(base_dir, 'model/rf_model.pkl')):
    st.error("Model belum disimpan. Jalankan save_model.py terlebih dahulu.")
    st.stop()

model, feature_columns = load_model()

st.markdown("""
<div class="hero-container">
    <div class="hero-badge">
        <span class="hero-badge-dot"></span>
        AI-POWERED PREDICTION
    </div>
    <div class="hero-icon-wrap">
        <div class="hero-icon-corner tl"></div>
        <div class="hero-icon-corner tr"></div>
        <div class="hero-icon-corner bl"></div>
        <div class="hero-icon-corner br"></div>
        <span class="hero-icon-symbol">HC</span>
        <div class="hero-icon-ring"></div>
        <div class="hero-icon-ring-outer"></div>
    </div>
    <div class="hero-title">HEALTHCOST AI</div>
    <div class="hero-subtitle">Prediksi biaya asuransi kesehatan tahunan Anda secara instan menggunakan kecerdasan buatan</div>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="flow-container">
    <div class="flow-label">Alur Prediksi</div>
    <div class="flow-track">
        <div class="flow-node">
            <div class="flow-node-box">
                <span class="flow-node-icon">IN</span>
            </div>
            <div class="flow-node-text">Input Data</div>
        </div>
        <div class="flow-connector">
            <div class="flow-connector-line"></div>
            <div class="flow-connector-dot"></div>
        </div>
        <div class="flow-node">
            <div class="flow-node-box">
                <span class="flow-node-icon">PP</span>
            </div>
            <div class="flow-node-text">Preprocessing</div>
        </div>
        <div class="flow-connector">
            <div class="flow-connector-line"></div>
            <div class="flow-connector-dot"></div>
        </div>
        <div class="flow-node">
            <div class="flow-node-box">
                <span class="flow-node-icon">RF</span>
            </div>
            <div class="flow-node-text">Random Forest</div>
        </div>
        <div class="flow-connector">
            <div class="flow-connector-line"></div>
            <div class="flow-connector-dot"></div>
        </div>
        <div class="flow-node">
            <div class="flow-node-box">
                <span class="flow-node-icon">AN</span>
            </div>
            <div class="flow-node-text">Analisis</div>
        </div>
        <div class="flow-connector">
            <div class="flow-connector-line"></div>
            <div class="flow-connector-dot"></div>
        </div>
        <div class="flow-node">
            <div class="flow-node-box">
                <span class="flow-node-icon">Rp</span>
            </div>
            <div class="flow-node-text">Hasil Prediksi</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if 'calculated_bmi' not in st.session_state:
    st.session_state.calculated_bmi = None

with st.expander("Belum tahu BMI Anda? Hitung di sini", expanded=False):
    st.markdown('<div class="bmi-calc-card"><div class="bmi-calc-title">KALKULATOR BMI</div>', unsafe_allow_html=True)
    bc1, bc2 = st.columns(2)
    with bc1:
        tinggi_cm = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=250, value=170, step=1, key="tinggi")
    with bc2:
        berat_kg = st.number_input("Berat Badan (kg)", min_value=20.0, max_value=250.0, value=65.0, step=0.5, key="berat")

    tinggi_m = tinggi_cm / 100
    calc_bmi = round(berat_kg / (tinggi_m ** 2), 1)

    if calc_bmi < 18.5:
        kategori = "Underweight"
        css_class = "bmi-underweight"
    elif calc_bmi < 25:
        kategori = "Normal"
        css_class = "bmi-normal"
    elif calc_bmi < 30:
        kategori = "Overweight"
        css_class = "bmi-overweight"
    else:
        kategori = "Obese"
        css_class = "bmi-obese"

    st.markdown(f"""
    <div class="bmi-result-box">
        <div class="bmi-result-left">
            <div class="bmi-result-label">BMI Anda</div>
            <div class="bmi-result-value">{calc_bmi}</div>
        </div>
        <div class="bmi-category {css_class}">{kategori}</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Gunakan BMI ini", key="use_bmi"):
        st.session_state.calculated_bmi = calc_bmi
        st.rerun()

default_bmi = st.session_state.calculated_bmi if st.session_state.calculated_bmi else 25.0

st.markdown('<div class="card"><div class="card-title">DATA PRIBADI</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Umur", min_value=18, max_value=100, value=30, step=1)
    sex = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
with col2:
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=default_bmi, step=0.1)
    children = st.number_input("Jumlah Anak Tanggungan", min_value=0, max_value=10, value=0, step=1)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card"><div class="card-title">INFORMASI KESEHATAN</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    smoker = st.selectbox("Status Merokok", ["Tidak Merokok", "Perokok"])
with col4:
    region = st.selectbox("Wilayah", ["Southwest", "Southeast", "Northwest", "Northeast"])
st.markdown('</div>', unsafe_allow_html=True)

sex_encoded = 1 if sex == "Laki-laki" else 0
smoker_encoded = 1 if smoker == "Perokok" else 0
region_northwest = 1 if region == "Northwest" else 0
region_southeast = 1 if region == "Southeast" else 0
region_southwest = 1 if region == "Southwest" else 0
age_squared = age ** 2
bmi_smoker = bmi * smoker_encoded

input_data = pd.DataFrame([{
    'age': age,
    'sex': sex_encoded,
    'bmi': bmi,
    'children': children,
    'smoker': smoker_encoded,
    'region_northwest': region_northwest,
    'region_southeast': region_southeast,
    'region_southwest': region_southwest,
    'age_squared': age_squared,
    'bmi_smoker': bmi_smoker,
}])
input_data = input_data[feature_columns]

st.markdown("<br>", unsafe_allow_html=True)

if st.button("HITUNG ESTIMASI BIAYA", type="primary", use_container_width=True):
    prediction_usd = model.predict(input_data)[0]
    prediction_idr = prediction_usd * KURS_USD_TO_IDR

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Estimasi Biaya Asuransi Tahunan</div>
        <div class="result-amount">${prediction_usd:,.0f} USD</div>
        <div class="result-usd">setara Rp {prediction_idr:,.0f} </div>
        <div class="result-divider"></div>
        <div class="result-note">per tahun &mdash; diprediksi oleh Random Forest AI Engine</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="card" style="animation-delay: 0.3s;">
        <div class="card-title">RINGKASAN INPUT</div>
        <div class="detail-grid">
            <div class="detail-item">
                <div class="detail-item-label">Umur</div>
                <div class="detail-item-value">{age} tahun</div>
            </div>
            <div class="detail-item">
                <div class="detail-item-label">Jenis Kelamin</div>
                <div class="detail-item-value">{sex}</div>
            </div>
            <div class="detail-item">
                <div class="detail-item-label">BMI</div>
                <div class="detail-item-value">{bmi:.1f}</div>
            </div>
            <div class="detail-item">
                <div class="detail-item-label">Jumlah Anak</div>
                <div class="detail-item-value">{children}</div>
            </div>
            <div class="detail-item">
                <div class="detail-item-label">Status Merokok</div>
                <div class="detail-item-value">{smoker}</div>
            </div>
            <div class="detail-item">
                <div class="detail-item-label">Wilayah</div>
                <div class="detail-item-value">{region}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    alerts_html = ""
    if smoker_encoded == 1:
        alerts_html += """<div class="alert-box alert-warning">
            <div class="alert-icon">!</div>
            Status <b>Perokok</b> adalah faktor utama yang menaikkan biaya asuransi secara signifikan.
        </div>"""
    if bmi > 30:
        alerts_html += """<div class="alert-box alert-info">
            <div class="alert-icon alert-icon-info">i</div>
            BMI di atas 30 (Obesitas) berkontribusi pada kenaikan biaya, terutama untuk perokok.
        </div>"""

    if alerts_html:
        st.markdown(alerts_html, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <div class="footer-brand">HEALTHCOST AI</div>
    <div class="footer-text">Powered by Random Forest Regressor &bull; R&sup2; Score: 0.8944</div>
    <div class="footer-text">Dataset: Medical Insurance Cost &bull; Tugas Akhir Data Mining</div>
</div>
""", unsafe_allow_html=True)
