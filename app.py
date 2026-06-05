import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests

# Page Layout Initialization
st.set_page_config(
    page_title="Alpha Engine", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("🛡️ Alpha Engine: Premium Momentum Scanner")
st.markdown("Multi-asset quantitative confluence tracking dashboard.")

st.markdown("""
    <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 1.5rem;}
        h1 {font-weight: 800; letter-spacing: -1px;}
        .stMetric {
            background-color: rgba(128, 128, 128, 0.05); 
            padding: 15px; 
            border-radius: 10px; 
            border: 1px solid rgba(128, 128, 128, 0.2);
        }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. WATCHLIST DEFINITION & STATE PERSISTENCE
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

if "persisted_tickers" not in st.session_state:
    st.session_state["persisted_tickers"] = ", ".join(default_tickers)

# ==============================================================================
# 2. SIDEBAR INTERACTIVE FILTERS
# ==============================================================================
st.sidebar.header("🎛️ Control Panel")
score_selection = st.sidebar.slider("Filter Confluence Score", min_value=-6, max_value=6, value=(-6, 6), step=1)
min_score, max_score = score_selection if isinstance(score_selection, tuple) else (score_selection, score_selection)

with st.sidebar.expander("📝 Edit Watchlist Assets", expanded=False):
    t_input = st.sidebar.text_area("Tickers (Comma Separated)", value=st.session_state["persisted_tickers"], key="ticker_input_field", height=250)
    st.session_state["persisted_tickers"] = t_input

watchlist = [t.strip().upper() for t in t_input.split(",") if t.strip()]
lookback = st.sidebar.selectbox("Lookback Data Window", ["6mo", "3mo", "1y"], index=0)

# ==============================================================================
# 3. QUANTITATIVE INDICATOR ENGINE
# ==============================================================================
def calculate_indicators(df):
    if len(df) < 55: return None
    df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    df['RSI'] = 100 - (100 / (1 + (gain.ewm(com=13, adjust=False).mean() / (loss.ewm(com=13, adjust=False).mean() + 1e-10))))
    
    e12, e26 = df['Close'].ewm(span=12, adjust=False).mean(), df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Hist'] = (e12 - e26) - (e12 - e26).ewm(span=9, adjust=False).mean()
    
    df['TR'] = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift()).abs(), (df['Low']-df['Close'].shift()).abs()], axis=1).max(axis=1)
    up, down = df['High'].diff(), -df['Low'].diff()
    df['+DM'] = np.where((up > down) & (up > 0), up, 0)
    df['-DM'] = np.where((down > up) & (down > 0), down, 0)
    
    tr_s = df['TR'].ewm(com=13, adjust=False).mean()
    df['+DI'] = 100 * (df['+DM'].ewm(com=13, adjust=False).mean() / (tr_s + 1e-10))
    df['-DI'] = 100 * (df['-DM'].ewm(com=13, adjust=False).mean() / (tr_s + 1e-10))
    df['ADX'] = (100 * (df['+DI'] - df['-DI']).abs() / (df['+DI'] + df['-DI'] + 1e-10)).ewm(com=13, adjust=False).mean()
    return df

def get_score_from_row(r):
    return (1 if r['Close'] > r['EMA5'] else -1) + (1 if r['Close'] > r['EMA20']
