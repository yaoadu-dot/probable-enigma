import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime

# ==========================================
# STREAMLIT PAGE SETUP
# ==========================================
st.set_page_config(
    page_title="Money Line Scanner", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Daily Money Line Scanner")
st.markdown("Track trend flips and structural momentum confluences in real-time.")

# ==========================================
# SIDEBAR CONFIGURATION
# ==========================================
st.sidebar.header("Scanner Settings")

bullish_thresh = st.sidebar.slider("Bullish Threshold (Bright Green)", 1, 5, 3)
bearish_thresh = st.sidebar.slider("Bearish Threshold (Bright Red)", -5, -1, -3)

default_tickers = "BTC-USD, SOL-USD, SUI-USD, AAPL, NVDA"
ticker_input = st.sidebar.text_area("Watchlist Tickers (comma separated)", default_tickers)
watchlist = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

# ==========================================
# NATIVE MATHEMATICAL INDICATOR ENGINES
# ==========================================
def calculate_money_line(df):
    # Flatten MultiIndex columns if present to avoid indexing mismatches
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    close_ser = df['Close']
    high_ser = df['High']
    low_ser = df['Low']
    vol_ser = df['Volume']

    # 1. Core EMA 20 Filter
    df['EMA_20'] = close_ser.ewm(span=20, adjust=False).mean()

    # 2. RSI 14 Momentum
    delta = close_ser.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ema_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    ema_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = ema_gain / (ema_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. MACD (12, 26, 9)
    ema12 = close_ser.ewm(span=12, adjust=False).mean()
    ema26 = close_ser.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = macd_line - signal_line

    # 4. Volume SMA 20
    df['Vol_SMA20'] = vol_ser.rolling(window=20).mean()

    # 5. Native SuperTrend Engine (7, 3.0)
    hl = high_ser - low_ser
    hc = (high_ser - close_ser.shift()).abs()
    lc = (low_ser - close_ser.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/7, adjust=False).mean()
    
    hl2 = (high_ser + low_ser) / 2
    upperband = hl2 + (3.0 * atr)
    lowerband = hl2 - (3.0 * atr)
    
    cl_arr = close_ser.to_numpy().copy()
    ub_arr = upperband.to_numpy().copy()
    lb_arr = lowerband.to_numpy().copy()
    dir_arr = np.ones(len(df))
    
    for i in range(1, len(df)):
        if cl_arr[i] > ub_arr[i-1]:
            dir_arr[i] = 1
        elif cl_arr[i] < lb_arr[i-1]:
            dir_arr[i] = -1
        else:
            dir_arr[i] = dir_arr[i-1]
            if dir_arr[i] == 1 and lb_arr[i] < lb_arr[i-1]:
                lb_arr[i] = lb_arr[i-1]
            if dir_arr[i] == -1 and ub_arr[i] > ub_arr[i-1]:
                ub_arr[i] = ub_arr[i-1]
                
    df['SuperTrend_Dir'] = dir_arr

    # Confluence Scoring Loop
    scores = []
    for i in range(len(df)):
        if pd.isna(df['EMA_20'].iloc[i]) or pd.isna(df['RSI'].iloc[i]) or pd.isna(df['SuperTrend_Dir'].iloc[i]):
            scores.append(0)
            continue
            
        score = 0
        score += 1 if df['SuperTrend_Dir'].iloc[i] == 1 else -1
        score += 1 if cl_arr[i] > df['EMA_20'].iloc[i] else -1
        score += 1 if df['RSI'].iloc[i] > 50 else -1
        score += 1 if df['MACD_Hist'].iloc[i] > 0 else -1
        if vol_ser.iloc[i] > df['Vol_SMA20'].iloc[i]:
            score += 1 if cl_arr[i] > df['EMA_20'].iloc[i] else -1
            
        scores.append(score)
        
    df['Money_Line_Score'] = scores
    return df

# ADVANCED FIX: Spoof a real browser configuration to bypass cloud bot firewalls
@st.cache_data(ttl=600)
def fetch_data(ticker):
    try:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
        })
        ticker_obj = yf.Ticker(ticker, session=session)
        df = ticker_obj.history(period="6mo", interval="1d", progress=False)
        return df
    except Exception as e:
        # Pass the raw structural exception backward to display natively
        return e

# ==========================================
# DASHBOARD EXECUTION
# ==========================================
if st.sidebar.button("🔄 Force Refresh Scanner"):
    st.cache_data.clear()

st.subheader(f"System Snapshot — {datetime.now().strftime('%Y-%m-%d %H:%M')} GMT")

cols = st.columns(3)
col_idx = 0

for ticker in watchlist:
    result = fetch_data(ticker)
    
    # Diagnostic Check: If the request raised an error, show it clearly on screen
    if isinstance(result, Exception):
        st.error(f"⚠️ Yahoo Connection Error for {ticker}: {str(result)}")
        continue
        
    if result is None or result.empty or len(result) < 25:
        st.warning(f"Could not load sufficient historical data for {ticker}. (Data came back empty)")
        continue
        
    # Isolate data safely 
    df = result.copy()
    df = calculate_money_line(df)
    
    prev_score = int(df['Money_Line_Score'].iloc[-3])
    curr_score = int(df['Money_Line_Score'].iloc[-2])
    last_price = float(df['Close'].iloc[-2])
    rsi_val = float(df['RSI'].iloc[-2])
    
    if curr_score >= bullish_thresh and prev_score < bullish_thresh:
        status_text = "🟢 BULLISH FLIP"
        card_bg = "#d4edda"
        text_color = "#155724"
    elif curr_score <= bearish_thresh and prev_score > bearish_thresh:
        status_text = "🔴 BEARISH FLIP"
        card_bg = "#f8d7da"
        text_color = "#721c24"
    elif curr_score >= bullish_thresh:
        status_text = "Sustained Bullish State"
        card_bg = "#e2f0d9"
        text_color = "#2e7d32"
    elif curr_score <= bearish_thresh:
        status_text = "Sustained Bearish State"
        card_bg = "#fce4d6"
        text_color = "#c65911"
    else:
        status_text = "Neutral / Low Conviction"
        card_bg = "#f8f9fa"
        text_color = "#6c757d"

    with cols[col_idx % 3]:
        st.markdown(
            f"""
            <div style="background-color: {card_bg}; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #ddd;">
                <h3 style="margin: 0; color: #333;">{ticker}</h3>
                <p style="margin: 5px 0; font-size: 1.2em; font-weight: bold; color: {text_color};">{status_text}</p>
                <hr style="margin: 10px 0; border: 0; border-top: 1px solid #ccc;">
                <table style="width:100%; font-size: 0.9em; color: #333;">
                    <tr><td><b>Close Price:</b></td><td style="text-align:right;">${last_price:,.2f}</td></tr>
                    <tr><td><b>Current Score:</b></td><td style="text-align:right; font-weight:bold;">{curr_score}</td></tr>
                    <tr><td><b>Previous Score:</b></td><td style="text-align:right;">{prev_score}</td></tr>
                    <tr><td><b>RSI (14d):</b></td><td style="text-align:right;">{rsi_val:.1f}</td></tr>
                </table>
            </div>
            """, 
            unsafe_allowed_html=True
        )
    col_idx += 1
