import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests

# App Layout Configuration
st.set_page_config(page_title="Alpha Engine Engine", layout="wide", initial_sidebar_state="expanded")

# Custom Modern CSS Tweaks
st.markdown("""
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
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
# 3. QUANTITATIVE MATHEMATICAL CALCULATIONS
# ==============================================================================
def calculate_indicators(df):
    if len(df) < 30: return None
    
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
    
    # True Range & ADX Engine
    df['TR'] = pd.concat([
        df['High'] - df['Low'], 
        (df['High'] - df['Close'].shift()).abs(), 
        (df['Low'] - df['Close'].shift()).abs()
    ], axis=1).max(axis=1)
    
    up_move = df['High'].diff()
    down_move = -df['Low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_smooth = df['TR'].ewm(com=13, adjust=False).mean()
    p_di = 100 * (pd.Series(plus_dm).ewm(com=13, adjust=False).mean() / (tr_smooth + 1e-10))
    m_di = 100 * (pd.Series(minus_dm).ewm(com=13, adjust=False).mean() / (tr_smooth + 1e-10))
    
    dx = 100 * (p_di - m_di).abs() / (p_di + m_di + 1e-10)
    df['ADX'] = dx.ewm(com=13, adjust=False).mean().values
    
    # Trend Base Baseline
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    return df

def process_state_and_score(df):
    if len(df) < 3: 
        return 0, "⚪ Neutral"
        
    r = df.iloc[-1]   # Current Row
    p = df.iloc[-2]   # Previous Row
    
    # Core Confluence Math System (-4 to +4)
    s = (1 if r['Close'] > r['EMA20'] else -1) + \
        (1 if r['RSI'] > 50 else -1) + \
        (1 if r['MACD_Hist'] > 0 else -1) + \
        (1 if r['Close'] > r['EMA20'] else -1)
        
    # State-Machine Status Logic
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
# 4. ROBUST MATRIX PARSING PIPELINE
# ==============================================================================
processed, failed = [], []
session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})

with st.spinner("Executing secure market matrix scan..."):
    try:
        data = yf.download(watchlist, period=lookback, session=session, group_by='ticker', progress=False)
        if data.empty:
            st.error("Data stream returned empty. Provider rate-limiting active.")
        else:
            valid_tickers = data.columns.get_level_values(0).unique() if isinstance(data.columns, pd.MultiIndex) else [watchlist]
            
            for t in watchlist:
                if t not in valid_tickers: continue
                try:
                    h = data[t].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
                    h = h.dropna(subset=['Close'])
                    h = calculate_indicators(h)
                    
                    if h is not None:
                        score, status = process_state_and_score(h)
                        processed.append({
                            "Asset": t, 
                            "Score": score, 
                            "Price": round(h.iloc[-1]['Close'], 4),
                            "RSI": round(h.iloc[-1]['RSI'], 1), 
                            "ADX": round(h.iloc[-1]['ADX'], 1),
                            "Status": status
                        })
                except: continue
    except Exception as e:
        st.error(f"Execution Error: {e}")

# ==============================================================================
# 5. MODERN UI VIEWPORTS & LAYOUT
# ==============================================================================
if processed:
    df_raw = pd.DataFrame(processed)
    
    # Apply Sidebar Score Filtering
    df_final = df_raw[(df_raw['Score'] >= score_range) & (df_raw['Score'] <= score_range)].sort_values("Score", ascending=False)
    
    # UI Top Metrics Dashboard Layer
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1: st.metric("Total Scanned", len(df_raw))
    with c_m2: st.metric("Filtered View", len(df_final))
    with c_m3: st.metric("Bullish Flips 🚀", len(df_raw[df_raw['Status'] == "🚀 Bullish Flip"]))
    with c_m4: st.metric("Bearish Flips 🩸", len(df_raw[df_raw['Status'] == "🩸 Bearish Flip"]))
    
    st.markdown("---")
    
    # Modern View Division Using Tabs
    tab1, tab2 = st.tabs(["📊 Main Engine Matrix", "🎯 High-Velocity Alerts"])
    
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
                    st.success(f"**{row['Asset']}** | Price: {row['Price']} | ADX: {row['ADX']} | RSI: {row['RSI']}")
            else:
                st.info("No validated high-velocity structural breakouts visible.")
                
        with col_right:
            st.markdown("#### 💤 Low Velocity Range Traps (Score +4 but ADX < 15)")
            traps = df_final[(df_final['Score'] == 4) & (df_final['ADX'] < 15)]
            if not traps.empty:
                for _, row in traps.iterrows():
                    st.warning(f"**{row['Asset']}** | Price: {row['Price']} | ADX: {row['ADX']} | Status: {row['Status']}")
            else:
                st.info("No range traps caught in this cycle loop.")
else:
    st.info("No asset nodes could be rendered. Adjust configurations or check logs.")
