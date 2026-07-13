"""
AlphaLens — Autonomous Quant Research Terminal
A unique AI-powered research interface for quantitative finance discovery.
"""
import os
import sys
import json
import time
import logging
import html
from datetime import datetime
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv(Path(project_root) / ".env")

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

import yfinance as yf
import pandas as pd
import numpy as np
import re
from alphalens.agents.memory import AgentMemoryEngine
from alphalens.core.utils import run_sync

logger = logging.getLogger(__name__)
memory_engine = AgentMemoryEngine()

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AlphaLens — Quant Research Terminal",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Unique AlphaLens Design — Research Terminal Aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #030712;
    --bg-secondary: #070b19;
    --bg-card: rgba(15, 23, 42, 0.45);
    --border: rgba(30, 58, 138, 0.35);
    --border-hover: rgba(0, 240, 255, 0.45);
    --accent-cyan: #00f0ff;
    --accent-emerald: #00ff88;
    --accent-violet: #d946ef;
    --accent-amber: #fbbf24;
    --text-primary: #f3f4f6;
    --text-secondary: #cbd5e1;
    --text-muted: #4b5563;
    --glass-grad: linear-gradient(135deg, rgba(30, 41, 59, 0.4), rgba(15, 23, 42, 0.4));
}

* { font-family: 'Outfit', 'Inter', sans-serif; }
code, .mono { font-family: 'JetBrains Mono', monospace; }

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: var(--bg-primary);
}
::-webkit-scrollbar-thumb {
    background: rgba(30, 58, 138, 0.35);
    border-radius: 99px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--accent-cyan);
}

.main {
    background-color: var(--bg-primary);
    background-image: 
        radial-gradient(circle at 50% 0%, rgba(30, 58, 138, 0.3) 0%, transparent 70%),
        linear-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.015) 1px, transparent 1px);
    background-size: 100% 100%, 36px 36px, 36px 36px;
    background-position: center top, center center, center center;
}

section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

/* Animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(16px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes logoEffect {
    0% {
        transform: scale(1) rotate(0deg);
        filter: drop-shadow(0 0 10px rgba(0, 240, 255, 0.4)) drop-shadow(0 0 20px rgba(0, 240, 255, 0.2));
    }
    50% {
        transform: scale(1.08) rotate(180deg);
        filter: drop-shadow(0 0 20px rgba(217, 70, 239, 0.5)) drop-shadow(0 0 40px rgba(217, 70, 239, 0.25));
    }
    100% {
        transform: scale(1) rotate(360deg);
        filter: drop-shadow(0 0 10px rgba(0, 240, 255, 0.4)) drop-shadow(0 0 20px rgba(0, 240, 255, 0.2));
    }
}

/* -------- Landing Screen -------- */
.landing-container {
    text-align: center;
    padding: 5rem 1rem 3rem 1rem;
    animation: fadeInUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) both;
}
.landing-logo {
    font-size: 4.2rem;
    margin-bottom: 1rem;
    display: inline-block;
    animation: logoEffect 12s infinite cubic-bezier(0.4, 0, 0.2, 1);
}
.landing-title {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: var(--text-primary);
    margin-bottom: 0.6rem;
    text-shadow: 0 0 40px rgba(0, 240, 255, 0.15);
}
.landing-title span {
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-violet));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900;
}
.landing-tagline {
    color: var(--text-secondary);
    font-size: 1.15rem;
    font-weight: 300;
    margin-bottom: 3.5rem;
    letter-spacing: 0.02em;
    opacity: 0.9;
}

/* Mode selector cards */
.mode-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    max-width: 820px;
    margin: 0 auto 3rem auto;
}
.mode-card {
    background: var(--glass-grad);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 28px 22px;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.03);
    animation: fadeInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
}
.mode-card:hover {
    transform: translateY(-6px);
}
.mode-card.discovery:hover {
    border-color: var(--accent-cyan);
    box-shadow: 0 10px 30px rgba(0, 240, 255, 0.15), inset 0 1px 1px rgba(0, 240, 255, 0.1);
}
.mode-card.causal:hover {
    border-color: var(--accent-violet);
    box-shadow: 0 10px 30px rgba(217, 70, 239, 0.15), inset 0 1px 1px rgba(217, 70, 239, 0.1);
}
.mode-card.risk:hover {
    border-color: var(--accent-emerald);
    box-shadow: 0 10px 30px rgba(0, 255, 136, 0.15), inset 0 1px 1px rgba(0, 255, 136, 0.1);
}

.mode-icon { 
    font-size: 2.2rem; 
    margin-bottom: 14px;
}
.mode-card.discovery .mode-icon {
    background: linear-gradient(135deg, var(--accent-cyan), #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.mode-card.causal .mode-icon {
    background: linear-gradient(135deg, var(--accent-violet), #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.mode-card.risk .mode-icon {
    background: linear-gradient(135deg, var(--accent-emerald), #059669);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.mode-label {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 8px;
    letter-spacing: -0.01em;
}
.mode-desc {
    font-size: 0.8rem;
    color: var(--text-secondary);
    line-height: 1.5;
    opacity: 0.85;
}

/* Terminal-style status bar */
.status-bar {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 40px;
    padding: 12px 32px;
    border: 1px solid var(--border);
    border-radius: 99px;
    margin: 0 auto 3.5rem auto;
    max-width: 600px;
    background: rgba(11, 21, 45, 0.4);
    backdrop-filter: blur(16px);
    box-shadow: 
        0 10px 30px rgba(0, 0, 0, 0.3),
        inset 0 1px 1px rgba(255, 255, 255, 0.03);
}
.status-item {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 500;
}
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
}
.status-dot.green { 
    background: var(--accent-emerald); 
    box-shadow: 0 0 12px var(--accent-emerald);
}
.status-dot.cyan { 
    background: var(--accent-cyan);
    box-shadow: 0 0 12px var(--accent-cyan);
}
.status-dot.amber { 
    background: var(--accent-amber);
    box-shadow: 0 0 12px var(--accent-amber);
}

/* -------- Chat Messages -------- */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 16px 0;
    animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
}
.msg-user-bubble {
    background: linear-gradient(135deg, rgba(30, 41, 88, 0.8), rgba(26, 31, 58, 0.8));
    backdrop-filter: blur(12px);
    border: 1px solid rgba(0, 240, 255, 0.25);
    border-radius: 20px 20px 4px 20px;
    padding: 14px 22px;
    color: var(--text-primary);
    max-width: 75%;
    font-size: 0.95rem;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
}
.msg-ai {
    margin: 20px 0;
    animation: fadeInUp 0.4s cubic-bezier(0.16, 1, 0.3, 1) both;
}
.msg-ai-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}
.msg-ai-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--accent-cyan);
    background: rgba(0, 240, 255, 0.08);
    border: 1px solid rgba(0, 240, 255, 0.25);
    border-radius: 6px;
    padding: 3px 10px;
    letter-spacing: 0.08em;
}
.msg-ai-time {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
}
.msg-ai-bubble {
    background: var(--glass-grad);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent-cyan);
    border-radius: 4px 16px 16px 4px;
    padding: 20px 26px;
    color: var(--text-secondary);
    font-size: 0.95rem;
    line-height: 1.8;
    max-width: 85%;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
}

/* -------- Sidebar -------- */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.2rem;
}
.sidebar-logo-icon {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-violet));
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    box-shadow: 0 0 15px rgba(0, 240, 255, 0.25);
}
.sidebar-logo-text {
    font-weight: 800;
    font-size: 1.15rem;
    color: var(--text-primary);
    letter-spacing: -0.02em;
}
.sidebar-logo-ver {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent-emerald);
    font-weight: 600;
}
.sidebar-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 1.8rem 0 0.8rem 0;
}

/* Glassmorphic overriding for sidebar buttons */
div[data-testid="stSidebar"] button {
    background: rgba(30, 41, 59, 0.25) !important;
    border: 1px solid rgba(255, 255, 255, 0.04) !important;
    color: var(--text-secondary) !important;
    border-radius: 10px !important;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    text-align: left !important;
    font-size: 0.82rem !important;
    padding: 10px 14px !important;
}
div[data-testid="stSidebar"] button:hover {
    border-color: rgba(0, 240, 255, 0.4) !important;
    background: rgba(0, 240, 255, 0.04) !important;
    color: var(--accent-cyan) !important;
    box-shadow: 0 4px 15px rgba(0, 240, 255, 0.08) !important;
}

/* Glowing Primary Button styling */
div[data-testid="stSidebar"] button[kind="primary"] {
    background: linear-gradient(135deg, rgba(236, 72, 153, 0.15), rgba(139, 92, 246, 0.15)) !important;
    border: 1px solid rgba(236, 72, 153, 0.4) !important;
    color: var(--text-primary) !important;
    box-shadow: 0 4px 20px rgba(236, 72, 153, 0.1) !important;
    font-weight: 600 !important;
}
div[data-testid="stSidebar"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, rgba(236, 72, 153, 0.22), rgba(139, 92, 246, 0.22)) !important;
    border-color: rgba(236, 72, 153, 0.7) !important;
    box-shadow: 0 0 25px rgba(236, 72, 153, 0.25) !important;
}

/* Chat Input Styling */
div[data-testid="stChatInput"] {
    border-radius: 99px !important;
    border: 1px solid var(--border) !important;
    background: rgba(15, 23, 42, 0.7) !important;
    backdrop-filter: blur(16px) !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4) !important;
}

/* Metric cards row */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 1.2rem 0;
}
.metric-card {
    background: var(--glass-grad);
    backdrop-filter: blur(8px);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
    transition: all 0.25s ease;
}
.metric-card:hover {
    border-color: rgba(217, 70, 239, 0.4);
    box-shadow: 0 0 15px rgba(217, 70, 239, 0.1);
}
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 800;
    color: var(--text-primary);
    margin-top: 4px;
}
.metric-change {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    margin-top: 4px;
    font-weight: 600;
}
.metric-change.up { color: var(--accent-emerald); }
.metric-change.down { color: #f43f5e; }

/* Hide streamlit chrome */
#MainMenu, header, footer { visibility: hidden; }
.stDeployButton { display: none; }
</style>

""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
@st.cache_resource
def get_llm():
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key_here":
        return None
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.4,
        groq_api_key=api_key,
    )


# ---------------------------------------------------------------------------
# Stock Tools
# ---------------------------------------------------------------------------
def fetch_stock(symbol: str, period: str = "3mo") -> dict:
    try:
        tk = yf.Ticker(symbol.upper())
        hist = tk.history(period=period)
        info = tk.info
        if hist.empty:
            return {"error": f"No data for {symbol}"}
        cp = float(hist["Close"].iloc[-1])
        pp = float(hist["Close"].iloc[0])
        chg = ((cp - pp) / pp) * 100
        return {
            "symbol": symbol.upper(),
            "name": info.get("longName", symbol.upper()),
            "price": cp, "change": chg,
            "high52": info.get("fiftyTwoWeekHigh"),
            "low52": info.get("fiftyTwoWeekLow"),
            "mcap": info.get("marketCap"),
            "pe": info.get("trailingPE"),
            "vol": info.get("volume"),
            "hist": hist,
        }
    except Exception as e:
        return {"error": str(e)}


def make_chart(hist, symbol):
    fig = go.Figure()
    # Area chart with gradient feel
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["Close"],
        mode="lines", name=symbol,
        line=dict(color="#00f0ff", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(0, 240, 255, 0.04)",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)", plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(color="#cbd5e1", family="JetBrains Mono", size=10),
        xaxis=dict(showgrid=False, linecolor="rgba(30, 58, 138, 0.3)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(30, 58, 138, 0.15)", linecolor="rgba(30, 58, 138, 0.3)"),
        margin=dict(l=45, r=15, t=10, b=25),
        height=280,
        showlegend=False,
    )
    return fig


def detect_ticker(text: str) -> str | None:
    known = {"AAPL", "TSLA", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA",
             "AMD", "NFLX", "SPY", "QQQ", "DIS", "BA", "JPM", "GS", "V", "MA",
             "PYPL", "SQ", "COIN", "INTC", "CRM", "UBER", "ABNB", "PLTR", "SOFI",
             "NIO", "RIVN", "LCID", "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"}
    words = text.upper().split()
    for w in words:
        c = w.strip(".,!?'\"")
        if c in known:
            return c
    kws = ["stock", "price", "share", "ticker", "chart", "buy", "sell", "trade"]
    for w in words:
        c = w.strip(".,!?'\"")
        if 1 <= len(c) <= 5 and c.isalpha() and any(k in text.lower() for k in kws):
            return c
    return None


# ---------------------------------------------------------------------------
# System Prompt — AlphaLens personality (NOT FinAgent)
# ---------------------------------------------------------------------------
ALPHALENS_SYSTEM = """You are AlphaLens, a quantitative research AI terminal.

You are NOT a general chatbot. You are a specialized quant research assistant focused on:
1. **Alpha Discovery**: Finding tradeable signals in financial data
2. **Causal Analysis**: Distinguishing correlation from causation in market relationships
3. **Risk Quantification**: CVaR, drawdown analysis, tail risk, factor exposures
4. **Portfolio Science**: Black-Litterman, mean-variance optimization, risk parity
5. **Statistical Signal Processing**: Information coefficient, signal decay, half-life estimation
6. **Market Microstructure**: Order flow, bid-ask dynamics, Kyle's lambda

Your communication style:
- Precise and data-driven, like a quant researcher writing a research note
- Use exact numbers and formulas when relevant
- Structure responses with clear headers and bullet points
- Reference academic papers and methodologies where appropriate
- Always distinguish between statistical significance and economic significance
- When discussing stocks, present the quantitative view — not just "buy/sell" opinions

When given stock data, analyze it through a quant lens:
- Compute and discuss risk-adjusted metrics
- Note any regime changes or structural breaks
- Discuss factor exposures (momentum, value, quality, volatility)
- Frame everything as testable hypotheses

You speak in a technical but clear manner. Think of yourself as a senior quant at a systematic hedge fund.
"""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="sidebar-logo">'
        '<div class="sidebar-logo-icon">◈</div>'
        '<div>'
        '<div class="sidebar-logo-text">AlphaLens</div>'
        '<div class="sidebar-logo-ver">v0.1.0 · LLaMA-70B</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    if st.button("⊕  New Research Session", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.charts = {}
        st.rerun()

    st.markdown('<div class="sidebar-label">Research Modules</div>', unsafe_allow_html=True)

    modules = [
        ("◈", "Alpha Discovery", "Hypothesis generation & testing"),
        ("⊿", "Causal Engine", "PC-algorithm & DML validation"),
        ("◇", "Risk Analytics", "CVaR, drawdown, tail risk"),
        ("▣", "Portfolio Lab", "Black-Litterman & optimization"),
        ("⫘", "Signal Scanner", "IC, ICIR, half-life metrics"),
        ("🧠", "Deep Learning", "TFT, N-BEATS, PatchTST ensemble"),
        ("🕸", "GNN (GAT)", "Cross-asset graph attention"),
        ("📊", "Regime Detection", "HMM bull/bear/high-vol"),
        ("🔍", "Explainability", "SHAP & causal attribution"),
    ]
    for icon, name, desc in modules:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;padding:6px 0;">'
            f'<span style="color:var(--accent-cyan);font-size:1rem;">{icon}</span>'
            f'<div><div style="color:var(--text-primary);font-size:0.82rem;font-weight:600;">{name}</div>'
            f'<div style="color:var(--text-muted);font-size:0.72rem;">{desc}</div></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown('<div class="sidebar-label">Quick Research</div>', unsafe_allow_html=True)

    queries = [
        "Analyze NVDA momentum factor",
        "Explain CVaR optimization",
        "What is Kyle's lambda?",
        "Black-Litterman vs MVO",
    ]
    for q in queries:
        if st.button(f"→ {q}", key=f"q_{q}", use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;padding:8px 0;">'
        '<div style="font-family:JetBrains Mono;font-size:0.6rem;color:var(--text-muted);">'
        'GROQ · LLaMA 3.3 70B · TFT · N-BEATS · PatchTST · GAT</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown('<div class="sidebar-label">Pipeline Control</div>', unsafe_allow_html=True)

    if st.button("🚀 Run Full Pipeline", use_container_width=True, type="primary", key="run_pipeline_btn"):
        st.session_state.run_pipeline = True
        st.rerun()

# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "charts" not in st.session_state:
    st.session_state.charts = {}

# ---------------------------------------------------------------------------
# Pipeline Execution Handler
# ---------------------------------------------------------------------------
if st.session_state.get("run_pipeline", False):
    st.session_state.run_pipeline = False

    st.markdown(
        '<div class="landing-container">'
        '<div class="landing-logo">◈</div>'
        '<div class="landing-title"><span>AlphaLens</span> Full Pipeline</div>'
        '<div class="landing-tagline">Running 7-agent autonomous research pipeline...</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Pipeline progress
    progress_bar = st.progress(0, text="Initializing pipeline...")
    status_area = st.empty()

    try:
        from alphalens.orchestration.graph import run_pipeline as _run_pipeline

        progress_bar.progress(10, text="📚 Literature Agent — generating hypothesis...")
        result = _run_pipeline(predictor_variable="momentum_12_1", target_asset_class="US_EQUITY")
        progress_bar.progress(100, text="✅ Pipeline complete!")

        # Display results
        st.markdown("---")

        # Pipeline Status
        routing = result.get("routing_decision", "UNKNOWN")
        status_color = "var(--accent-emerald)" if routing == "ACCEPTED" else "#f43f5e"
        st.markdown(
            f'<div style="text-align:center;padding:20px;">'
            f'<div style="font-size:1.4rem;font-weight:700;color:{status_color};">'
            f'{"✅ HYPOTHESIS ACCEPTED" if routing == "ACCEPTED" else "❌ HYPOTHESIS REJECTED"}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sharpe Ratio", f"{result.get('sharpe_ratio', 0):.2f}")
        with col2:
            st.metric("p-value", f"{result.get('p_value', 1.0):.4f}")
        with col3:
            st.metric("ATE", f"{result.get('ate_magnitude', 0):.4f}")
        with col4:
            regime = result.get("current_regime", "N/A")
            st.metric("Regime", regime.upper() if regime else "N/A")

        # Portfolio Weights
        weights = result.get("portfolio_weights", {})
        if weights:
            st.markdown("### 💼 Portfolio Allocation")
            fig_port = go.Figure(data=[go.Pie(
                labels=list(weights.keys()),
                values=list(weights.values()),
                hole=0.45,
                marker=dict(colors=["#00f0ff", "#d946ef", "#00ff88", "#fbbf24", "#f43f5e"]),
                textinfo="label+percent",
                textfont=dict(color="#f3f4f6", size=11),
            )])
            fig_port.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1"),
                showlegend=False,
                height=300,
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_port, use_container_width=True)

        # Ensemble Predictions
        ensemble = result.get("ensemble_predictions", {})
        if ensemble:
            st.markdown("### 🧠 Ensemble Forecast (TFT + N-BEATS + PatchTST)")
            horizons = sorted(ensemble.keys(), key=lambda x: int(x))
            vals = [float(ensemble[h]) for h in horizons]
            fig_ens = go.Figure(data=[go.Bar(
                x=[f"{h}d" for h in horizons],
                y=vals,
                marker=dict(color=["#00f0ff", "#d946ef", "#00ff88"]),
                text=[f"{v:.6f}" for v in vals],
                textposition="outside",
                textfont=dict(color="#f3f4f6", size=10),
            )])
            fig_ens.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1", family="JetBrains Mono", size=10),
                xaxis=dict(showgrid=False, title="Forecast Horizon"),
                yaxis=dict(showgrid=True, gridcolor="rgba(30,58,138,0.15)", title="Predicted Return"),
                height=250,
                margin=dict(t=10, b=40, l=50, r=10),
            )
            st.plotly_chart(fig_ens, use_container_width=True)

        # Model Contributions
        contributions = result.get("model_contributions", {})
        if contributions:
            st.markdown("### 📊 Model Ensemble Weights")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("TFT", f"{contributions.get('tft', 0):.1%}")
            with col_b:
                st.metric("N-BEATS", f"{contributions.get('nbeats', 0):.1%}")
            with col_c:
                st.metric("PatchTST", f"{contributions.get('patchtst', 0):.1%}")

        # Graph Edges (GAT)
        graph_edges = result.get("graph_edges", [])
        if graph_edges:
            st.markdown("### 🕸️ Cross-Asset Graph (GAT)")
            edge_text = ", ".join([f"{e.get('source','?')}↔{e.get('target','?')}" for e in graph_edges[:10]])
            st.markdown(f"*{len(graph_edges)} edges discovered:* {edge_text}")

        # Regime Probabilities
        regime_probs = result.get("regime_probabilities", {})
        if regime_probs:
            st.markdown("### 📈 Regime Probabilities")
            fig_regime = go.Figure(data=[go.Bar(
                x=list(regime_probs.keys()),
                y=list(regime_probs.values()),
                marker=dict(color=["#00ff88", "#f43f5e", "#fbbf24"]),
                text=[f"{v:.1%}" for v in regime_probs.values()],
                textposition="outside",
                textfont=dict(color="#f3f4f6", size=10),
            )])
            fig_regime.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#cbd5e1", family="JetBrains Mono", size=10),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(30,58,138,0.15)", range=[0, 1]),
                height=220,
                margin=dict(t=10, b=30, l=40, r=10),
            )
            st.plotly_chart(fig_regime, use_container_width=True)

        # Rejection info
        if result.get("rejection_reason"):
            st.warning(f"**Rejection Reason:** {result['rejection_reason']}")
            st.info(f"**Iterations:** {result.get('iteration', 0)}")

    except Exception as e:
        progress_bar.progress(100, text="❌ Pipeline failed")
        st.error(f"Pipeline execution failed: {e}")
        logger.error(f"Pipeline execution error: {e}", exc_info=True)

    st.stop()

# ---------------------------------------------------------------------------
# Main Area
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    # ===== LANDING SCREEN =====
    st.markdown(
        '<div class="landing-container">'
        '<div class="landing-logo">◈</div>'
        '<div class="landing-title"><span>AlphaLens</span> Research Terminal</div>'
        '<div class="landing-tagline">Autonomous quantitative research powered by causal AI</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Status bar
    st.markdown(
        '<div class="status-bar">'
        '<div class="status-item"><span class="status-dot green"></span> LLM Connected</div>'
        '<div class="status-item"><span class="status-dot cyan"></span> Groq Inference</div>'
        '<div class="status-item"><span class="status-dot amber"></span> Market Data Ready</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Mode cards
    st.markdown(
        '<div class="mode-grid">'
        '<div class="mode-card discovery">'
        '  <div class="mode-icon">◈</div>'
        '  <div class="mode-label">Alpha Discovery</div>'
        '  <div class="mode-desc">Generate & validate trading hypotheses from research</div>'
        '</div>'
        '<div class="mode-card causal">'
        '  <div class="mode-icon">⊿</div>'
        '  <div class="mode-label">Causal Analysis</div>'
        '  <div class="mode-desc">Distinguish causation from correlation in signals</div>'
        '</div>'
        '<div class="mode-card risk">'
        '  <div class="mode-icon">◇</div>'
        '  <div class="mode-label">Risk & Portfolio</div>'
        '  <div class="mode-desc">CVaR optimization, Black-Litterman, risk parity</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

else:
    # ===== CHAT HISTORY =====
    for i, msg in enumerate(st.session_state.messages):
        escaped_content = html.escape(msg["content"])
        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-user"><div class="msg-user-bubble">{escaped_content}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            ts = msg.get("ts", "")
            st.markdown(
                f'<div class="msg-ai">'
                f'<div class="msg-ai-header">'
                f'<span class="msg-ai-tag">ALPHALENS</span>'
                f'<span class="msg-ai-time">{ts}</span>'
                f'</div>'
                f'<div class="msg-ai-bubble">{escaped_content}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            ck = f"chart_{i}"
            if ck in st.session_state.charts:
                st.plotly_chart(st.session_state.charts[ck], use_container_width=True)

            # Show metrics if present
            mk = f"metrics_{i}"
            if mk in st.session_state:
                data = st.session_state[mk]
                st.markdown(
                    f'<div class="metric-row">'
                    f'<div class="metric-card">'
                    f'  <div class="metric-label">Price</div>'
                    f'  <div class="metric-value">${data["price"]:.2f}</div>'
                    f'  <div class="metric-change {"up" if data["change"] >= 0 else "down"}">'
                    f'    {"▲" if data["change"] >= 0 else "▼"} {data["change"]:+.2f}%</div>'
                    f'</div>'
                    f'<div class="metric-card">'
                    f'  <div class="metric-label">Market Cap</div>'
                    f'  <div class="metric-value">${data.get("mcap", 0)/1e9:.1f}B</div>'
                    f'</div>'
                    f'<div class="metric-card">'
                    f'  <div class="metric-label">P/E Ratio</div>'
                    f'  <div class="metric-value">{data.get("pe", "N/A")}</div>'
                    f'</div>'
                    f'<div class="metric-card">'
                    f'  <div class="metric-label">52W Range</div>'
                    f'  <div class="metric-value" style="font-size:0.9rem;">${data.get("low52", 0):.0f}-${data.get("high52", 0):.0f}</div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Chat Input
# ---------------------------------------------------------------------------
pending = st.session_state.pop("pending_query", None)
user_input = st.chat_input("Research query — e.g. 'analyze NVDA momentum' or 'explain CVaR'") or pending

if user_input:
    llm = get_llm()
    if not llm:
        st.error("⚠️ GROQ_API_KEY not configured. Please set GROQ_API_KEY in your .env file.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})

    # Detect stock ticker
    ticker = detect_ticker(user_input)
    stock_ctx = ""
    stock_data = None

    if ticker:
        with st.spinner(f"◈ Fetching market data for {ticker}..."):
            stock_data = fetch_stock(ticker)
        if "error" not in stock_data:
            stock_ctx = (
                f"\n\n[LIVE MARKET DATA — {stock_data['symbol']}]\n"
                f"Name: {stock_data['name']}\n"
                f"Price: ${stock_data['price']:.2f}\n"
                f"3M Change: {stock_data['change']:+.2f}%\n"
                f"52W High: ${stock_data.get('high52', 0)}\n"
                f"52W Low: ${stock_data.get('low52', 0)}\n"
                f"Market Cap: ${stock_data.get('mcap', 0):,.0f}\n"
                f"P/E: {stock_data.get('pe', 'N/A')}\n"
                f"Volume: {stock_data.get('vol', 0):,.0f}\n"
            )

    # Try to extract and save the user's name if they introduced themselves
    name_match = re.search(
        r"(?:my name is|call me|i am|i'm)\s+([a-zA-Z\s]{2,15})",
        user_input.lower(),
    )
    if name_match:
        extracted_name = name_match.group(1).strip().title()
        try:
            run_sync(
                memory_engine.store_semantic_fact(
                    "user_interface", "user_profile", {"name": extracted_name}
                )
            )
        except Exception as e:
            logger.warning(f"Failed to persist user name in database: {e}")

    # Fetch stored user profile from database
    user_name = "User"
    try:
        user_profile = run_sync(
            memory_engine.get_semantic_fact("user_interface", "user_profile")
        )
        if user_profile and "name" in user_profile:
            user_name = user_profile["name"]
    except Exception as e:
        logger.warning(f"Failed to load user profile: {e}")

    # Inject user name into system instructions so LLM knows who they are talking to
    customized_system_prompt = (
        ALPHALENS_SYSTEM + f"\n\nThe user's name is: {user_name}. Keep responses regarding user name and greetings/introductions in simple, direct, plain text without any bolding, headers, or markdown formatting."
    )

    # Build LangChain messages
    lc_msgs = [SystemMessage(content=customized_system_prompt)]
    for m in st.session_state.messages[-10:]:
        if m["role"] == "user":
            lc_msgs.append(HumanMessage(content=m["content"]))
        else:
            lc_msgs.append(AIMessage(content=m["content"]))

    if stock_ctx:
        lc_msgs[-1] = HumanMessage(content=user_input + stock_ctx)

    # LLM call
    with st.spinner("◈ Processing research query..."):
        try:
            resp = llm.invoke(lc_msgs)
            ai_text = resp.content.strip()
        except Exception as e:
            st.error(f"⚠️ Groq LLM invocation failed: {e}")
            st.stop()

    now = datetime.now().strftime("%H:%M:%S")
    idx = len(st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": ai_text, "ts": now})

    # Store chart and metrics if stock data
    if stock_data and "error" not in stock_data:
        st.session_state.charts[f"chart_{idx}"] = make_chart(stock_data["hist"], stock_data["symbol"])
        st.session_state[f"metrics_{idx}"] = stock_data

    st.rerun()
