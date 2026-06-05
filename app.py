import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests

# App Layout Configuration
st.set_page_config(page_title="Alpha Engine", layout="wide", initial_sidebar_state="expanded")

# Premium Theme UI Styles
st.markdown("""
    <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 1.5rem;}
        h1 {font-weight: 800; letter-spacing: -1px;}
        .stMetric {background-color: #1e2230; padding: 15px; border-radius: 10px; border: 1px solid #2e3440;}
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Alpha Engine: Premium Momentum Scanner")
st.markdown("Multi-asset quantitative confluence tracking dashboard.")

# ==============================================================================
# 1. EXPANDED UNIFIED WATCHLIST
# ==============================================================================
default_tickers = [
    "MRVL", "PLUG", "RGTI", "IREN", "QBTS", "CRWV", "MSTR", "CIFR", "IONQ", "HOOD", 
    "CLSK", "WULF", "QUBT", "CORZ", "NBIS", "BE", "TSM", "CBRS", "HUT", "TTWO", 
    "CAT", "MU", "META", "TSLA", "AVGO", "MSFT", "GOOGL", "AAPL", "AMZN", "HON", 
    "NVDA", "ORCL", "CRM", "PLTR", "VVV", "MET", "RARE", "BTC-USD", "ETH-USD", 
    "BNB-USD", "ZEC-USD", "XMR-USD", "SOL-USD", "QNT-USD", "LTC-USD", "DASH-USD", 
    "LINK-USD", "INJ-USD", "ICP-USD", "NEAR-USD", "XRP-USD", "XLM-USD", "DOGE-USD", 
    "JUP-USD", "WIF-USD", "BONK-USD", "SHIB-USD", "GC=F", "SI=F", "PL=F", "PA=F", 
    "CL=F", "BZ=F", "HG=F", "LIT", "GLTR", "PALL", "REMX", "SIL", "BOTZ", "IGV", 
    "AIQU", "REXC"
]

# ==============================================================================
# 2. MODERNIZED SIDEBAR CONTROLS
# ==============================================================================
st.sidebar.header("🎛️ Control Panel")

# Score Filter Slider
score_range = st.sidebar.slider("Filter Confluence Score", min_value=-4, max_value=4, value=(-4, 4), step=1)

with st.sidebar.expander("📝 Edit Watchlist Assets", expanded=False):
    t_input = st.sidebar.text_area("Tickers (Comma Separated)", ", ".join(default_tickers), height=250)
watchlist = [t.strip().upper() for t in t_input.split(",") if t.strip()]

lookback = st.sidebar.selectbox("Lookback Data Window", ["6mo", "3mo", "1y"], index=0)

# ==============================================================================
# 3. QUANTITATIVE MATHEMATICAL CALCULATIONS (INDEX PRESERVED)
# ==============================================================================
def calculate_indicators(df):
    if len(df) < 30: return None
    
    # Trend Baselines (Medium & Short-Term)
    df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # RSI Engine
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ema_g = gain.ewm(com=13, adjust=False).mean()
    ema_l = loss.ewm(com=13, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (ema_g / (ema_l + 1e-10))))
    
    # MACD Engine
    e12 = df['Close'].ewm(span=12, adjust=False).mean()
    e26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Hist'] = (e12 - e26) - (e12 - e26).ewm(span=9, adjust=False).mean()
    
    # True Range & ADX Engine (Using pure DataFrame assignment to prevent index shifts)
    df['TR'] = pd.concat([
        df['High'] - df['Low'], 
        (df['High'] - df['Close'].shift()).abs(), 
        (df['Low'] - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    
    up_move = df['High'].diff()
    down_move = -df['Low'].diff()
    
    df['+DM'] = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    df['-DM'] = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_smooth = df['TR'].ewm(com=13, adjust=False).mean()
    p_di = 100 * (df['+DM'].ewm(com=13, adjust=False).mean() / (tr_smooth + 1e-10))
    m_di = 100 * (df['-DM'].ewm(com=13, adjust=False).mean() / (tr_smooth + 1e-10))
    
    dx = 100 * (p_di - m_di).abs() / (p_di + m_di + 1e-10)
    df['ADX'] = dx.ewm(com=13, adjust=False).mean()
    
    return df

def process_state_and_score(df):
    if len(df) < 3: 
        return 0, "⚪ Neutral"
        
    r = df.iloc[-1]   # Current Candle Row
    p = df.iloc[-2]   # Prior Candle Row
    
    # Balanced 4-Signal Confluence Score Matrix (-4 to +4)
    s = (1 if r['Close'] > r['EMA20'] else -1) + \
        (1 if r['Close'] > r['EMA5'] else -1) + \
        (1 if r['RSI'] > 50 else -1) + \
        (1 if r['MACD_Hist'] > 0 else -1)
        
    # Multi-State Status Flow Architecture
    if r['ADX'] < 15:
        status = "⚪ Neutral"
    elif r['Close'] > r['EMA20'] and p['Close'] <= p['EMA20']:
        status = "🚀 Bullish Flip"
    elif r['Close'] < r['EMA20'] and p['Close'] >= p['EMA20']:
        status = "🩸 Bearish Flip"
    elif r['Close'] > r['EMA20']:
        status = "🟢 Bullish"
    else:
        status = "🔴 Bearish"
