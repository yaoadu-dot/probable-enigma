import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime

# Page Layout Initialization
st.set_page_config(
    page_title="Alpha Engine", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("🛡️ Alpha Engine: Premium Momentum Scanner")
st.markdown("Multi-asset quantitative confluence tracking dashboard.")

# Improved Theme Styles
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
# 2. SIDEBAR INTERACTIVE FILTERS & CACHE CONTROL
# ==============================================================================
st.sidebar.header("🎛️ Control Panel")

# Force Cache Clear Action Button
if st.sidebar.button("🔄 Force Hard Reload", use_container_width=True):
    st.cache_data.clear()
    st.toast("Data matrix cache purged! Fetching live stream...")

# Confluence filter slider scaled exactly from -6 to +6
score_selection = st.sidebar.slider(
    "Filter Confluence Score", 
    min_value=-6, 
    max_value=6, 
    value=(-6, 6), 
    step=1
)

if isinstance(score_selection, tuple):
    min_score, max_score = score_selection
else:
    min_score, max_score = score_selection, score_selection

with st.sidebar.expander("📝 Edit Watchlist Assets", expanded=False):
    t_input = st.sidebar.text_area(
        "Tickers (Comma Separated)", 
        value=st.session_state["persisted_tickers"], 
        key="ticker_input_field",
        height=250
    )
    st.session_state["persisted_tickers"] = t_input

watchlist = [t.strip().upper() for t in t_input.split(",") if t.strip()]

lookback = st.sidebar.selectbox(
    "Lookback Data Window", 
    ["6mo", "3mo", "1y"], 
    index=0
)

# ==============================================================================
# 3. QUANTITATIVE INDICATOR ENGINE
# ==============================================================================
def calculate_indicators(df):
    if len(df) < 55: return None
    
    df['EMA5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ema_g = gain.ewm(com=13, adjust=False).mean()
    ema_l = loss.ewm(com=13, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (ema_g / (ema_l + 1e-10))))
    
    e12 = df['Close'].ewm(span=12, adjust=False).mean()
    e26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Hist'] = (e12 - e26) - (e12 - e26).ewm(span=9, adjust=False).mean()
    
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
    df['+DI'] = 100 * (df['+DM'].ewm(com=13, adjust=False).mean() / (tr_smooth + 1e-10))
    df['-DI'] = 100 * (df['-DM'].ewm(com=13, adjust=False).mean() / (tr_smooth + 1e-10))
    
    dx = 100 * (df['+DI'] - df['-DI']).abs() / (df['+DI'] + df['-DI'] + 1e-10)
    df['ADX'] = dx.ewm(com=13, adjust=False).mean()
    
    return df

def process_state_and_score(df):
    if len(df) < 3: return 0, "⚪ Neutral"
        
    r = df.iloc[-1]   
    p = df.iloc[-2]   
    
    s = (1 if r['Close'] > r['EMA5'] else -1) + \
        (1 if r['Close'] > r['EMA20'] else -1) + \
        (1 if r['Close'] > r['EMA50'] else -1) + \
        (1 if r['RSI'] > 50 else -1) + \
        (1 if r['MACD_Hist'] > 0 else -1) + \
        (1 if r['+DI'] > r['-DI'] else -1)
        
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
        
    return s, status

# ==============================================================================
# 4. HIGH-AVAILABILITY CACHED NETWORK DATA PIPELINE
# ==============================================================================
@st.cache_data(ttl=600)
def fetch_and_build_matrix(tickers_tuple, selected_lookback):
    processed_nodes, failed_nodes = [], []
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
    })
    
    tickers_list = list(tickers_tuple)
    
    try:
        data = yf.download(
            tickers_list, 
            period=selected_lookback, 
            session=session, 
            group_by='ticker', 
            progress=False, 
            timeout=15
        )
        
        if data.empty:
            return processed_nodes, [{"Asset": "ALL", "Reason": "Empty stream frame."}], datetime.datetime.now()
            
        if isinstance(data.columns, pd.MultiIndex):
            valid_tickers = data.columns.get_level_values(0).unique()
        else:
            valid_tickers = [tickers_list] if 'Close' in data.columns else []

        for t in tickers_list:
            if t not in valid_tickers:
                failed_nodes.append({"Asset": t, "Reason": "Omitted from layout"})
                continue
            try:
                h = data[t].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
                h = h.dropna(subset=['Close'])
                
                if len(h) < 55:
                    failed_nodes.append({"Asset": t, "Reason": f"Insufficient history ({len(h)} bars)"})
                    continue
                    
                h = calculate_indicators(h)
                if h is not None:
                    score, status = process_state_and_score(h)
                    processed_nodes.append({
                        "Asset": t, "Score": score, "Status": status,
                        "Price": round(h.iloc[-1]['Close'], 4),
                        "RSI": round(h.iloc[-1]['RSI'], 1), "ADX": round(h.iloc[-1]['ADX'], 1)
                    })
                else:
                    failed_nodes.append({"Asset": t, "Reason": "Math logic failure"})
            except Exception as inner_ex:
