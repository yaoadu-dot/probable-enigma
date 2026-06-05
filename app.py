import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# 1. IMMEDIATE PAGE INITIALIZATION (Prevents Blank Screen Hangs)
st.set_page_config(page_title="Alpha Engine", layout="wide", initial_sidebar_state="expanded")

st.title("🛡️ Alpha Engine: Premium Momentum Scanner")
st.markdown("Multi-asset quantitative confluence tracking dashboard.")

# Custom Premium Styling Layer
st.markdown("""
    <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 1.5rem;}
        h1 {font-weight: 800; letter-spacing: -1px;}
        .stMetric {background-color: #1e2230; padding: 15px; border-radius: 10px; border: 1px solid #2e3440;}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. WATCHLIST DEFINITION
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
# 3. SIDEBAR INTERACTIVE FILTERS
# ==============================================================================
st.sidebar.header("🎛️ Control Panel")

# Score Filtering Slider
score_range = st.sidebar.slider("Filter Confluence Score", min_value=-4, max_value=4, value=(-4, 4), step=1)

with st.sidebar.expander("📝 Edit Watchlist Assets", expanded=False):
    t_input = st.sidebar.text_area("Tickers (Comma Separated)", ", ".join(default_tickers), height=250)
watchlist = [t.strip().upper() for t in t_input.split(",") if t.strip()]

lookback = st.sidebar.selectbox("Lookback Data Window", ["6mo", "3mo", "1y"], index=0)

# ==============================================================================
# 4. MATHEMATICAL INDICATOR ENGINE
# ==============================================================================
def calculate_indicators(df):
    if len(df) < 30: return None
    
    # Trend Overlays
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
    
    # ADX Directional Engine
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
    if len(df) < 3: return 0, "⚪ Neutral"
        
    r = df.iloc[-1]   # Current candle
    p = df.iloc[-2]   # Previous candle
    
    # Pure Multi-Signal Score Confluence (-4 to +4)
    s = (1 if r['Close'] > r['EMA20'] else -1) + \
        (1 if r['Close'] > r['EMA5'] else -1) + \
        (1 if r['RSI'] > 50 else -1) + \
        (1 if r['MACD_Hist'] > 0 else -1)
        
    # State-Machine Status Flow Matrix
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
# 5. DATA INGESTION PIPELINE WITH TIMEOUT PROTECTION
# ==============================================================================
processed, failed = [], []

# Establish highly resilient network session wrapper
session = requests.Session()
retries = Retry(total=2, backoff_factor=0.3, status_forcelist=)
session.mount('https://', HTTPAdapter(max_retries=retries))
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})

with st.spinner("Executing real-time structural matrix scan..."):
    try:
        # Enforce strict 10s download timeout parameter to avoid infinite network hangs
        data = yf.download(watchlist, period=lookback, session=session, group_by='ticker', progress=False, timeout=10)
        
        if data.empty:
            st.error("The data provider stream returned empty. Yahoo may be throttling connections.")
        else:
            # Safely read multi-index layer structure
            if isinstance(data.columns, pd.MultiIndex):
                valid_tickers = data.columns.get_level_values(0).unique()
            else:
                valid_tickers = [watchlist] if 'Close' in data.columns else []

            for t in watchlist:
                if t not in valid_tickers:
                    failed.append({"Asset": t, "Reason": "Omitted from streaming response"})
                    continue
                try:
                    h = data[t].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
                    h = h.dropna(subset=['Close'])
                    
                    if len(h) < 30:
                        failed.append({"Asset": t, "Reason": f"Insufficient history ({len(h)} rows)"})
                        continue
                        
                    h = calculate_indicators(h)
                    if h is not None:
                        score, status = process_state_and_score(h)
                        processed.append({
                            "Asset": t, "Score": score, "Status": status,
                            "Price": round(h.iloc[-1]['Close'], 4),
                            "RSI": round(h.iloc[-1]['RSI'], 1), "ADX": round(h.iloc[-1]['ADX'], 1)
                        })
                    else:
                        failed.append({"Asset": t, "Reason": "Indicator math calculation breakdown"})
                except Exception as inner_ex:
                    failed.append({"Asset": t, "Reason": str(inner_ex)})
    except Exception as e:
        st.error(f"Network Connection Interrupted: {e}")

# ==============================================================================
# 6. UI PRESENTATION ENGINE
# ==============================================================================
if processed:
    df_raw = pd.DataFrame(processed)
    
    # Filter output live dynamically based on the sidebar control slider position
    df_final = df_raw[(df_raw['Score'] >= score_range) & (df_raw['Score'] <= score_range)].sort_values("Score", ascending=False)
    
    # Executive KPI Metric Dashboard Blocks
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1: st.metric("Total Online Assets", len(df_raw))
    with c_m2: st.metric("Matches Filter View", len(df_final))
    with c_m3: st.metric("Bullish Flips 🚀", len(df_raw[df_raw['Status'] == "🚀 Bullish Flip"]))
    with c_m4: st.metric("Bearish Flips 🩸", len(df_raw[df_raw['Status'] == "🩸 Bearish Flip"]))
    
    st.markdown("---")
    
    # Modern Tabbed Display Layout
    tab1, tab2, tab3 = st.tabs(["📊 Main Engine Matrix", "🎯 High-Velocity Alerts", "🔍 System Logs"])
    
    with tab1:
        st.markdown("### Active Watchlist Matrix")
        st.dataframe(
            df_final[["Asset", "Score", "Status", "Price", "RSI", "ADX"]], 
            width="stretch", 
            hide_index=True
        )
        
    with tab2:
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### 🔥 Structural Momentum (+4 Score & ADX ≥ 25)")
            breakouts = df_final[(df_final['Score'] == 4) & (df_final['ADX'] >= 25)]
            if not breakouts.empty:
                for _, row in breakouts.iterrows():
                    st.success(f"**{row['Asset']}** | Price: {row['Price']} | ADX: {row['ADX']} | Status: {row['Status']}")
            else:
                st.info("No breakout assets meeting criteria in this data slice.")
                
        with col_right:
            st.markdown("#### 💤 Range Traps (+4 Score but ADX < 15)")
            traps = df_final[(df_final['Score'] == 4) & (df_final['ADX'] < 15)]
            if not traps.empty:
                for _, row in traps.iterrows():
                    st.warning(f"**{row['Asset']}** | Price: {row['Price']} | ADX: {row['ADX']} | Status: {row['Status']}")
            else:
                st.info("No range compression grinds captured.")

    with tab3:
        st.markdown("### Matrix Performance Logs")
        if failed:
            st.dataframe(pd.DataFrame(failed), width="stretch", hide_index=True)
        else:
            st.success("All systems green. Zero processing faults reported.")
else:
    st.error("🚨 Critical Error: Could not compute data nodes for rendering.")
    if failed:
        st.markdown("### 🔍 Engine Diagnostic Debug Logs")
        st.dataframe(pd.DataFrame(failed), width="stretch", hide_index=True)
